#!/usr/bin/env python
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
FlowMalTrans training entry point (unsupervised cross-ISA assembly translation).

This single script replaces the former train.py / train_flow.py / train_arm32_x86.py.
With the default values of its options it behaves exactly like the old train_flow.py;
the schedule that was hard-coded in the train_arm32_x86.py notebook export (the paper
recipe) is available through three switches:

    --two_stage true      two-stage schedule
                            epoch <  bt_begin_epoch : CLM / MLM / PC / AE / MT steps only
                            epoch >= bt_begin_epoch : back-translation (BT) steps ONLY
                          (default false = train_flow.py: the non-BT steps never stop, BT is
                          added on top from bt_begin_epoch on)
    --eval_periodic N     in every iteration of every N-th epoch (epoch 0 included) run ONE
                          EXTRA pass over --bt_steps. Despite the historical name this is
                          a training step (bt_step calls optimize()), not an evaluation;
                          its BT loss shows up in the log. 0 = off.
    --eval_every N        run the validation / test evaluation (perplexity, accuracy, BLEU
                          if --eval_bleu) every N epochs and call save_best_model.
                          1 = every epoch (train_flow.py behaviour), 0 = never (notebook).

Minimal example (see scripts/train_arm32_x86.sh for the recorded arm32->x86 configuration):

    python train.py --exp_name arm32-x86 --data_path data/processed/arm32-x86 \\
        --lgs arm32-x86 --encoder_only false --separate_decoders true \\
        --ae_steps arm32,x86 --bt_steps arm32-x86-arm32,x86-arm32-x86 ...
"""

import os
import sys
import json
import random
import argparse
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from xlm.slurm import init_signal_handler, init_distributed_mode
from xlm.data.loader import check_data_params, load_data
from xlm.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from xlm.model import check_model_params, build_model, build_model_separate
from xlm.model.memory import HashingMemory
from xlm.trainer import SingleTrainer, EncDecTrainer
from xlm.evaluation.evaluator import SingleEvaluator, EncDecEvaluator
from xlm.model.flows import create_normalizing_flows, FLOW_TYPES



def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="FlowMalTrans: unsupervised cross-ISA translation (training)")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name (required, except with --debug)")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model every N epochs as periodic-<epoch>.pth (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID (random if empty)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed python / numpy / torch (-1: do not seed, original behaviour). Not bit-exact: the AE/BT "
                             "batch order comes from an unseeded RandomState in xlm/data/dataset.py")

    # float16 / AMP API (requires NVIDIA apex)
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=True,
                        help="Only use an encoder (plain language model). Use false for translation.")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    # memory / adaptive softmax: these options only exist when enabled, so that --help stays short.
    # They are looked up with a tiny pre-parser (add_help=False) so that --help still prints everything.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--use_memory", type=bool_flag, default=False)
    pre.add_argument("--asm", type=bool_flag, default=False)
    pre_args = pre.parse_known_args()[0]

    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")
    if pre_args.use_memory:
        HashingMemory.register_args(parser)
        parser.add_argument("--mem_enc_positions", type=str, default="",
                            help="Memory positions in the encoder ('4' for inside layer 4, '7,10+' for inside layer 7 and after layer 10)")
        parser.add_argument("--mem_dec_positions", type=str, default="",
                            help="Memory positions in the decoder. Same syntax as `mem_enc_positions`.")

    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if pre_args.asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise (used by the denoising auto-encoder steps)
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Folder with the binarized data (train/valid/test.<lang>.pth, valid/test.<l1>-<l2>.<l>.pth)")
    parser.add_argument("--lgs", type=str, required=True,
                        help="Languages (lg1-lg2-lg3 .. ex: arm32-x86). With separate decoders / flows the FIRST one is the 'source' side")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length of the CLM / MLM streams")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch (CLM/MLM streams) and unit used to count 'sentences' per epoch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch for AE / MT / BT batches (-1: use batch_size sentences)")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size: the number of 'sentences' (counted in --batch_size units, times "
                             "--accumulate_gradients) trained per epoch; must be > 0")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Number of epochs run by THIS invocation (after --reload_checkpoint it is the number of "
                             "additional epochs, not an absolute epoch number)")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Early stopping: '<metric>,<patience>'. A metric is maximised unless its name starts with "
                             "'_' (then it is minimised): valid_arm32-x86_mt_bleu,10 or _valid_arm32-x86_mt_ppl,10. "
                             "Needs --eval_every 1. Empty = never stop early")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Comma-separated metrics for which save_best_model keeps a best-<metric>.pth (maximised unless "
                             "prefixed with '_', e.g. _valid_arm32-x86_mt_ppl). Needs --eval_every > 0")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")

    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM), e.g. arm32,x86")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM), e.g. arm32,x86")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps (supervised), e.g. arm32-x86")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps, e.g. arm32,x86")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps, e.g. arm32-x86-arm32,x86-arm32-x86")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model ('enc.pth,dec.pth' for encoder-decoder models)")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Resume from a periodic-<epoch>.pth checkpoint: weights, epoch, iteration and (if the file "
                             "contains optimizer state) the LR schedule step; Adam moments are NOT restored")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Beam search: stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Run the validation/test evaluation every N epochs (1: every epoch, 0: never)")
    parser.add_argument("--is_checkpoint", type=bool_flag, default=True,
                        help="--reload_checkpoint also restores epoch / iteration / LR step (false: weights only)")
    parser.add_argument("--save_last_checkpoint", type=bool_flag, default=False,
                        help="Write checkpoint.pth (with optimizer state) at the end of every epoch; it is picked up "
                             "automatically when the same --exp_id is restarted")

    # schedule (paper recipe, see module docstring)
    parser.add_argument('--bt_begin_epoch', type=int, default=1,
                        help='The epoch for begining Back-translation')
    parser.add_argument('--two_stage', type=bool_flag, default=False,
                        help='Stop CLM/MLM/PC/AE/MT steps at bt_begin_epoch and train with BT only afterwards')
    parser.add_argument('--eval_periodic', type=int, default=0,
                        help='In every iteration of every N-th epoch (epoch 0 included) run one extra pass over --bt_steps. '
                             'NOTE: this is a training step (it updates the parameters), not an evaluation. 0 = off')

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Ignore SLURM environment variables (use true for a single process inside a SLURM allocation)")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")
    parser.add_argument("--cuda_launch_blocking", type=bool_flag, default=False,
                        help="Set CUDA_LAUNCH_BLOCKING=1 (accurate CUDA error locations, slower)")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # flows / sentence-level latent code
    parser.add_argument('--use_latent_code', type=bool_flag, default=False,
                        help='whether a sentence-level latent code should be used for generation')
    parser.add_argument('--z_size', type=int, default=256,
                        help='Dimension of the sentence-level latent code')
    parser.add_argument('--flow_type', type=str, default='none',
                        help="The type of flow to be used: %s (needs --use_latent_code true)" % ", ".join(FLOW_TYPES))
    parser.add_argument('--hiddenflow_layers', type=int, default=3,
                        help='Number of MLP layers in flow')
    parser.add_argument('--hiddenflow_units', type=int, default=256,
                        help='Size of MLP in flows')
    parser.add_argument('--hiddenflow_flow_nums', type=int, default=3,
                        help='Number of flows in the NF')
    parser.add_argument('--use_mean_pooling', type=bool_flag, default=True,
                        help='Use mean pooling to generate the sentence-level latent code')
    parser.add_argument('--use_max_pooling', type=bool_flag, default=True,
                        help='Use max pooling to generate the sentence-level latent code')
    parser.add_argument('--use_CSL', type=bool_flag, default=True,
                        help='Use CSL embedding to generate the sentence-level latent code')
    parser.add_argument('--nll_weight', type=float, default=0.1,
                        help='The weight for the NLL / density estimation loss from the NFs')
    parser.add_argument('--latent_concat', type=bool_flag, default=True,
                        help='Whether concat or add z to the tensor')
    parser.add_argument('--separate_decoders', type=bool_flag, default=False,
                        help='Whether use separate decoders (one per language, needs exactly two languages)')
    parser.add_argument('--freeze_encoder', type=bool_flag, default=False,
                        help='Freeze the encoder (except its word, position and language embeddings)')
    parser.add_argument('--use_weight_network', type=bool_flag, default=False,
                        help='Whether use weight network for incorporating latent code z (only if --latent_concat false)')
    parser.add_argument('--concat_h', type=bool_flag, default=True,
                        help='Whether concat instead add the middle representations h of max-pooling, ...')
    parser.add_argument('--use_decoder_lang_emb', type=bool_flag, default=True,
                        help='Whether use langauge embeddings for the decoder.')
    parser.add_argument('--detach_type', type=str, default='none', choices=['none', 'initial_z', 'all'],
                        help='The detach type for the pair of flows, none, initial_z, all')
    parser.add_argument('--consider_nll_in_bt', type=bool_flag, default=False,
                        help='whether considers the nll loss in the BT step')

    # MASS related (kept for compatibility with the model code)
    parser.add_argument("--attention_setting", type=str, default="none", choices=["none", "v1"],
                        help="Setting for attention module, benefits for distinguish language")
    parser.add_argument('--load_mass', type=bool_flag, default=False,
                        help='whether the loadeded model is MASS (must be true iff --attention_setting v1).')

    return parser


def check_args(params, parser):
    """
    Validate option combinations that would otherwise fail late (after data loading) or silently do nothing.
    Must be called after the --debug override and before check_data_params.
    """
    if params.exp_name.strip() == '':
        parser.error("--exp_name must be set")

    # normalise the flow type ('None' used to be accepted by the argparse default but crashed in flows.py)
    params.flow_type = params.flow_type.lower()
    if params.flow_type not in FLOW_TYPES:
        parser.error("--flow_type must be one of %s" % ", ".join(FLOW_TYPES))

    # MASS settings
    if (params.attention_setting == 'v1') != params.load_mass:
        parser.error("--attention_setting v1 and --load_mass true must be used together")

    if params.use_latent_code:
        if params.encoder_only:
            parser.error("--use_latent_code needs an encoder-decoder model (--encoder_only false)")
        if not (params.use_mean_pooling or params.use_max_pooling or params.use_CSL):
            parser.error("--use_latent_code needs at least one of --use_mean_pooling / --use_max_pooling / --use_CSL")
        if params.z_size <= 0:
            parser.error("--z_size must be > 0")
    elif params.flow_type != 'none':
        print("WARNING: --flow_type %s is ignored because --use_latent_code is false" % params.flow_type, file=sys.stderr)

    if params.separate_decoders:
        if params.encoder_only:
            parser.error("--separate_decoders needs --encoder_only false")
        if len(params.lgs.split('-')) != 2:
            parser.error("--separate_decoders needs exactly two languages in --lgs (got %r)" % params.lgs)
    if not params.encoder_only and len(params.lgs.split('-')) != 2:
        parser.error("encoder-decoder training needs exactly two languages in --lgs (got %r)" % params.lgs)

    if params.eval_every < 0 or params.eval_periodic < 0:
        parser.error("--eval_every and --eval_periodic must be >= 0")
    if params.stopping_criterion != '' and params.eval_every != 1:
        parser.error("--stopping_criterion needs the evaluation scores of every epoch (--eval_every 1)")
    if params.epoch_size <= 0:
        parser.error("--epoch_size must be > 0")
    if params.use_latent_code and params.flow_type in ('scf', 'glow') and params.z_size % 2:
        parser.error("--z_size must be even for --flow_type %s" % params.flow_type)
    if params.use_memory and params.separate_decoders:
        parser.error("--use_memory is not supported together with --separate_decoders")

    # every language of every *_steps option must be one of --lgs
    langs = params.lgs.split('-')
    for name in ('clm_steps', 'mlm_steps', 'pc_steps', 'ae_steps', 'mt_steps', 'bt_steps'):
        for step in getattr(params, name).split(','):
            for lang in step.split('-') if step else []:
                if lang not in langs:
                    parser.error("--%s: language %r is not in --lgs %s" % (name, lang, params.lgs))


def check_steps(params):
    """
    Checks that need the parsed *_steps (call after check_data_params).
    """
    if params.eval_periodic > 0 and len(params.bt_steps) == 0:
        raise ValueError("--eval_periodic needs --bt_steps")
    if params.two_stage and len(params.bt_steps) == 0:
        raise ValueError("--two_stage needs --bt_steps")


def resolve_latent_params(params):
    """
    Decide, from the options only (no CUDA, no data), whether a h->z projection and normalizing flows are
    needed. Done before initialize_exp so that the values end up in params.pkl.
    """
    params.use_flow = False
    params.use_h2z_matrix = False
    params.h2z_in_dim = None
    if params.use_latent_code:
        if params.concat_h:
            in_dim = params.emb_dim * sum([params.use_mean_pooling, params.use_max_pooling, params.use_CSL])
        else:
            in_dim = params.emb_dim
        if in_dim != params.z_size:
            params.use_h2z_matrix = True
            params.h2z_in_dim = in_dim
        params.use_flow = params.flow_type != 'none'


def build_latent_modules(params):
    """
    Create the modules that live outside the Transformer and that are registered in the trainer
    (they are optimised and checkpointed under the names shared_h2z_projection / src_flow / tgt_flow).
    """
    shared_h2z_projection, src_flow, tgt_flow = None, None, None
    if params.use_h2z_matrix:
        shared_h2z_projection = nn.Linear(params.h2z_in_dim, params.z_size).cuda()
    if params.use_flow:
        flow_kwargs = {'hiddenflow_layers': params.hiddenflow_layers,
                       'hiddenflow_units': params.hiddenflow_units,
                       'hiddenflow_flow_nums': params.hiddenflow_flow_nums}
        src_flow = create_normalizing_flows(flow_type=params.flow_type, z_size=params.z_size, dropout=0, kwargs=flow_kwargs).cuda()
        tgt_flow = create_normalizing_flows(flow_type=params.flow_type, z_size=params.z_size, dropout=0, kwargs=flow_kwargs).cuda()
    return shared_h2z_projection, src_flow, tgt_flow


def set_train_mode(trainer, params):
    """
    Put the trained networks back in training mode (only matters after an evaluation pass).
    """
    if params.encoder_only:
        trainer.model.train()
        return
    trainer.encoder.train()
    if params.separate_decoders:
        trainer.src_decoder.train()
        trainer.tgt_decoder.train()
    else:
        trainer.decoder.train()


def run_steps(trainer, params):
    """
    One training iteration (the body of the `while trainer.n_sentences < trainer.epoch_size` loop).
    """
    epoch = trainer.epoch
    bt_phase = epoch >= params.bt_begin_epoch
    pretrain_phase = not (params.two_stage and bt_phase)

    if pretrain_phase:
        # CLM steps
        for lang1, lang2 in shuf_order(params.clm_steps, params):
            trainer.clm_step(lang1, lang2, params.lambda_clm)

        # MLM steps (also includes TLM if lang2 is not None)
        for lang1, lang2 in shuf_order(params.mlm_steps, params):
            trainer.mlm_step(lang1, lang2, params.lambda_mlm)

        # parallel classification steps
        for lang1, lang2 in shuf_order(params.pc_steps, params):
            trainer.pc_step(lang1, lang2, params.lambda_pc)

        # denoising auto-encoder steps
        for lang in shuf_order(params.ae_steps):
            trainer.mt_step(lang, lang, params.lambda_ae)

        # machine translation steps
        for lang1, lang2 in shuf_order(params.mt_steps, params):
            trainer.mt_step(lang1, lang2, params.lambda_mt)

    # back-translation steps, only after bt_begin_epoch
    if bt_phase:
        for lang1, lang2, lang3 in shuf_order(params.bt_steps):
            trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)

    # extra BT step every `eval_periodic` epochs (notebook behaviour; it also trains the model)
    if params.eval_periodic > 0 and epoch % params.eval_periodic == 0:
        for lang1, lang2, lang3 in shuf_order(params.bt_steps):
            trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)


def main(params):

    if params.cuda_launch_blocking:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA device found; this code base only runs on GPU.")

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # seeds
    if params.seed >= 0:
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)

    # decide about projection / flows (pure python) before the parameters are dumped
    resolve_latent_params(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler(params)

    # load data
    data = load_data(params)

    # latent-code modules (created before the Transformer, as in the original scripts)
    shared_h2z_projection, src_flow, tgt_flow = build_latent_modules(params)

    # build model
    if params.encoder_only:
        model = build_model(params, data['dico'])
    else:
        if not params.separate_decoders:
            encoder, decoder = build_model(params, data['dico'])
        else:
            encoder, src_decoder, tgt_decoder = build_model_separate(params, data['dico'])
            decoder = [src_decoder, tgt_decoder]

    # build trainer, reload potential checkpoints / build evaluator
    if params.encoder_only:
        trainer = SingleTrainer(model, data, params)
        evaluator = SingleEvaluator(trainer, data, params)
    else:
        trainer = EncDecTrainer(encoder, decoder, data, params,
                                src_flow=src_flow, tgt_flow=tgt_flow,
                                shared_h2z_projection=shared_h2z_projection)
        evaluator = EncDecEvaluator(trainer, data, params)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        return

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0
        if params.eval_periodic > 0 and trainer.epoch % params.eval_periodic == 0:
            logger.info("============ Evaluation Epoch %i ============" % trainer.epoch)

        while trainer.n_sentences < trainer.epoch_size:

            run_steps(trainer, params)

            set_train_mode(trainer, params)
            trainer.iter()

            # if no steps are performed (e.g. only BT steps are given and the epoch is < bt_begin_epoch, or all
            # lambdas are 0), n_sentences never grows: leave the loop instead of spinning forever. The epoch is
            # then just "evaluate / save / next epoch", which is how a loaded model can be evaluated at epoch 0.
            if trainer.n_sentences == 0:
                break

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluation (perplexity / accuracy / optional BLEU on valid and test)
        if params.eval_every > 0 and trainer.epoch % params.eval_every == 0:
            scores = evaluator.run_all_evals(trainer)

            # print / JSON log
            for k, v in scores.items():
                logger.info("%s -> %.6f" % (k, v))
            if params.is_master:
                logger.info("__log__:%s" % json.dumps(scores))

            trainer.save_best_model(scores)
        else:
            scores = OrderedDict([('epoch', trainer.epoch)])

        # end of epoch
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True
        params.debug_train = True

    # check parameters
    check_args(params, parser)
    check_data_params(params)
    check_model_params(params)
    check_steps(params)

    # run experiment
    main(params)
