import os
import json
import random
import argparse
import xlm
from xlm.slurm import init_signal_handler, init_distributed_mode
from xlm.data.loader import check_data_params, load_data
from xlm.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from xlm.model import check_model_params, build_model, build_model_separate
from xlm.model.memory import HashingMemory
from xlm.trainer import SingleTrainer, EncDecTrainer
from xlm.evaluation.evaluator import SingleEvaluator, EncDecEvaluator
from xlm.model.flows import create_normalizing_flows
from torch import nn

from collections import OrderedDict

def get_parser():
    """
    Generate a parameters parser.
    """
    
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="dumped",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="arm32-x86",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=50,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument('--eval_periodic', type=int, default=20, help='Evaluate every N epochs')
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    #parser.add_argument("--fp16", type=bool_flag, default=False,
    #                    help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=False,
                        help="Only use an encoder")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=1024,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    # memory parameters
    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")
    if parser.parse_known_args()[0].use_memory:
        HashingMemory.register_args(parser)
        parser.add_argument("--mem_enc_positions", type=str, default="",
                            help="Memory positions in the encoder ('4' for inside layer 4, '7,10+' for inside layer 7 and after layer 10)")
        parser.add_argument("--mem_dec_positions", type=str, default="",
                            help="Memory positions in the decoder. Same syntax as `mem_enc_positions`.")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
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

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=3,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0.1,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0.1,
                        help="Randomly blank input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="data/processed/en-fr/",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="en-fr",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=2000,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0003",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=10000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=10000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="valid_en-de_mt_bleu,100",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="valid_en-de_mt_bleu",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=8,
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
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="en,fr",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="en-fr-en,fr-en-fr",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=True,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # flows related
    parser.add_argument('--use_latent_code', type=bool_flag, default=False, help='whether a sentence-level latent code should be used for generation')
    parser.add_argument('--z_size', type=int, default=256, help='Dimension of the sentence-level latent code')
    parser.add_argument('--flow_type', type=str, default="scf", help="The type of flow to be used, none, scf or glow")
    parser.add_argument('--hiddenflow_layers', type=int, default=3, help='Number of MLP layers in flow')
    parser.add_argument('--hiddenflow_units', type=int, default=256, help='Size of MLP in flows')
    parser.add_argument('--hiddenflow_flow_nums', type=int, default=3, help='Number of flows in the NF')
    parser.add_argument('--use_mean_pooling', type=bool_flag, default=True,
                        help='Use mean pooling to generate the sentence-level latent code')
    parser.add_argument('--use_max_pooling', type=bool_flag, default=True,
                        help='Use max pooling to generate the sentence-level latent code')
    parser.add_argument('--use_CSL', type=bool_flag, default=True,
                        help='Use CSL embedding to generate the sentence-level latent code')
    parser.add_argument('--nll_weight', type=float, default=0.1,
                        help='The weight for the NLL / density estimation loss from the NFs')
    parser.add_argument('--bt_begin_epoch', type=int, default=1, 
                        help='The epoch for begining Back-translation')
    parser.add_argument('--latent_concat', type=bool_flag, default=True, 
                        help='Whether concat or add z to the tensor')
    parser.add_argument('--separate_decoders', type=bool_flag, default=False,
                        help='Whether use separate decoders')
    parser.add_argument('--freeze_encoder', type=bool_flag, default=False,
                        help='Whether freeze the parameteres of the encoder')#This parameter is unused
    parser.add_argument('--use_weight_network', type=bool_flag, default=True,
                        help='Whether use weight network for incorporating latent code z')
    parser.add_argument('--concat_h', type=bool_flag, default=True,
                        help='Whetehr concat instead add the middle representations h of max-pooling, ...')
    parser.add_argument('--use_decoder_lang_emb', type=bool_flag, default=True,
                        help='Whether use langauge embeddings for the decoder.')
    parser.add_argument('--detach_type', type=str, default='none',
                        help='The detach type for the pair of flows, none, initial_z, all')

    # MASS related
    parser.add_argument("--attention_setting", type=str, default="none", choices=["none", "v1"],
                        help="Setting for attention module, benefits for distinguish language")
    parser.add_argument('--load_mass', type=bool_flag, default=False,
                        help='whether the loadeded model is MASS.')
    parser.add_argument('--consider_nll_in_bt', type=bool_flag, default=False,
                        help='whether considers the nll loss in the BT step')

    # evaluation
    parser.add_argument("--is_checkpoint", type=bool_flag, default=True)

    return parser

def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    data = load_data(params)

    # check MASS parameters
    if params.attention_setting == 'v1':
        assert params.load_mass is True
    if params.load_mass is True:
        assert params.attention_setting == 'v1'

    # check latent codes and flow params
    params.use_flow = False
    params.use_h2z_matrix = False
    if params.use_latent_code:
        if params.concat_h:
            latent_code_options = [params.use_mean_pooling, params.use_max_pooling, params.use_CSL]
            if params.emb_dim * sum(latent_code_options) != params.z_size:
                shared_h2z_projection = nn.Linear(params.emb_dim * sum(latent_code_options), params.z_size).cuda()
                params.use_h2z_matrix = True
            else:
                shared_h2z_projection = None
        else:
            if params.emb_dim != params.z_size:
                shared_h2z_projection = nn.Linear(params.emb_dim, params.z_size).cuda()
                params.use_h2z_matrix = True
            else:
                shared_h2z_projection = None
        if params.flow_type != 'none':
            assert params.detach_type in ['none', 'initial_z', 'all']
            flow_kwargs = {'hiddenflow_layers': params.hiddenflow_layers, 'hiddenflow_units': params.hiddenflow_units, 'hiddenflow_flow_nums': params.hiddenflow_flow_nums}
            src_flow = create_normalizing_flows(flow_type=params.flow_type, z_size=params.z_size, dropout=0, kwargs=flow_kwargs).cuda()
            tgt_flow = create_normalizing_flows(flow_type=params.flow_type, z_size=params.z_size, dropout=0, kwargs=flow_kwargs).cuda()
            params.use_flow = True

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
        if params.use_latent_code:
            if params.use_flow:
                trainer = EncDecTrainer(encoder, decoder, data, params, src_flow=src_flow, tgt_flow=tgt_flow, shared_h2z_projection=shared_h2z_projection)
            else:
                trainer = EncDecTrainer(encoder, decoder, data, params, shared_h2z_projection=shared_h2z_projection)
        else:
            trainer = EncDecTrainer(encoder, decoder, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # language model training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0
        if trainer.epoch % params.eval_periodic == 0:
            logger.info("============ Evaluation Epoch %i ============" % trainer.epoch)
        while trainer.n_sentences < trainer.epoch_size:

            # parallel classification steps
            #for lang1, lang2 in shuf_order(params.pc_steps, params):
            #    trainer.pc_step(lang1, lang2, params.lambda_pc)

            # denoising auto-encoder steps
            if trainer.epoch < params.bt_begin_epoch:
                # CLM steps
                for lang1, lang2 in shuf_order(params.clm_steps, params):
                    trainer.clm_step(lang1, lang2, params.lambda_clm)    

                # MLM steps (also includes TLM if lang2 is not None)
                for lang1, lang2 in shuf_order(params.mlm_steps, params):
                    trainer.mlm_step(lang1, lang2, params.lambda_mlm)    

                for lang in shuf_order(params.ae_steps):
                    trainer.mt_step(lang, lang, params.lambda_ae)

            # machine translation steps
            # for lang1, lang2 in shuf_order(params.mt_steps, params):
            #     trainer.mt_step(lang1, lang2, params.lambda_mt)

            # here doing a small revision, only after certain epochs to start the BT
            if trainer.epoch >= params.bt_begin_epoch:
                # back-translation steps
                for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                    trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)

            if trainer.epoch % params.eval_periodic == 0:
                
                trainer.encoder.eval()
                trainer.tgt_decoder.eval()
                trainer.src_decoder.eval()
                for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                    trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)

            trainer.encoder.train()
            trainer.tgt_decoder.train()
            trainer.src_decoder.train()
            trainer.iter()

            # if no steps are performed (usually for the first epoch if we want to see how the performance for the loaded model)
            if trainer.n_sentences == 0:
                break

        logger.info("============ End of epoch %i ============" % trainer.epoch)
        # evaluate perplexity
        #scores = evaluator.run_all_evals(trainer)
        scores = OrderedDict({'epoch': trainer.epoch})

        #if (trainer.epoch >params.bt_begin_epoch) and (trainer.epoch %10==0):
           #scores=evaluator.evaluate_mt(scores, 'valid', lang1, lang2, False)
        #scores=None #a dummy variable
        # print / JSON log
            #for k, v in scores.items():
            #    logger.info("%s -> %.6f" % (k, v))
        # if params.is_master:
        #     logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        #trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)

if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # generate parser / parse parameters
    
    parser = get_parser()
    params = parser.parse_args(args=[]) #the content args=[] is necessary to use in ipython notebook
    params.flow_type="None"
    params.fp16=False
    params.exp_name="arm32-x86"
    #params.exp_name="test"
    params.data_path="data/processed/arm32-x86/"
    params.reload_model=""
    params.reload_checkpoint=""
    params.lgs="arm32-x86"
    params.ae_steps="arm32,x86"
    params.bt_steps="arm32-x86-arm32,x86-arm32-x86"
    params.pc_steps=""
    params.mt_steps=""
    params.mlm_steps="arm32,x86"
    params.clm_steps="arm32,x86"

    params.separate_decoders=True

    params.evaluation_metrics=""
    params.stopping_criterion=""
    params.save_periodic=100

    params.epoch_size=100000
    params.batch_size=64
    params.bt_begin_epoch=1000
    params.max_len=200
    params.emb_dim=32
    params.n_layers=4
    params.n_heads=4
    params.optimizer="adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0003"
    params.reload_checkpoint="dumped/arm32-x86/adlui4fvq4/periodic-300.pth"


    # debug mode
    # if params.debug:
    #     params.exp_name = 'debug'
    #     params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
    #     params.debug_slurm = True
    #     params.debug_train = True
    # to adjust to the en-de pre-trained model
    # check parameters
    #params.reload_model='ende/mlm_enfr_1024_new.pth,ende/mlm_enfr_1024_new.pth'
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)






