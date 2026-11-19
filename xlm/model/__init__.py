# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .flows import FLOW_TYPES
from .pretrain import load_embeddings
from .transformer import TransformerModel
from .memory import HashingMemory


logger = getLogger()

# Parameters of the output layer that only exist when the sentence-level latent
# code z (flow adapter) is used. They are absent from checkpoints of plain
# (vanilla XLM / MLM) models and are therefore kept at their random
# initialisation when such a checkpoint is reloaded.
LATENT_PARAM_PREFIXES = ('pred_layer.z_proj.', 'pred_layer.weight_network.')



def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # sentence-level latent code / normalizing flows (flow adapter)
    if getattr(params, 'use_latent_code', False):
        assert params.z_size > 0
        params.flow_type = params.flow_type.lower()
        assert params.flow_type in FLOW_TYPES, \
            "flow_type must be one of %s, got '%s'" % (', '.join(FLOW_TYPES), params.flow_type)

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
        s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
        assert len(s_enc) == len(set(s_enc))
        assert len(s_dec) == len(set(s_dec))
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
        params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]
        params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]
        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max([x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max([x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx]
            model.pred_layer.proj.weight[i] = embeddings[idx]
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def to_device(model):
    """
    Move a model to the GPU if one is available (building / reloading models
    itself also works on a CPU-only machine).
    """
    return model.cuda() if torch.cuda.is_available() else model


def reload_state_dict(path, keys):
    """
    Read a checkpoint on CPU and return the state dict stored under the first
    key of `keys` that is present ('model' for a single-model checkpoint,
    'encoder' / 'decoder' for an encoder-decoder checkpoint).
    """
    # weights_only=False: checkpoints also store the training parameters (numpy values), which the
    # default weights_only=True of torch >= 2.6 refuses to unpickle. Only load checkpoints you trust.
    reloaded = torch.load(path, map_location='cpu', weights_only=False)
    for key in keys:
        if key in reloaded:
            state_dict = reloaded[key]
            break
    else:
        raise KeyError("None of the keys %s found in %s (found: %s)" % (list(keys), path, list(reloaded.keys())))
    if all([k.startswith('module.') for k in state_dict.keys()]):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def keep_missing_latent_params(model, state_dict):
    """
    Add to `state_dict` the latent-code related parameters of `model`
    (pred_layer.z_proj.*, pred_layer.weight_network.*) that the reloaded
    checkpoint does not contain, so that they keep their random initialisation.
    """
    for name, value in model.state_dict().items():
        if name.startswith(LATENT_PARAM_PREFIXES) and name not in state_dict:
            logger.warning("Parameter %s not found in the reloaded model, keeping its initial value." % name)
            state_dict[name] = value


def reload_encoder(encoder, path):
    logger.info("Reloading encoder from %s ..." % path)
    state_dict = reload_state_dict(path, ('model', 'encoder'))
    keep_missing_latent_params(encoder, state_dict)
    encoder.load_state_dict(state_dict)


def reload_decoder(decoder, path, name='decoder'):
    logger.info("Reloading %s from %s ..." % (name, path))
    state_dict = reload_state_dict(path, ('model', 'decoder'))
    keep_missing_latent_params(decoder, state_dict)
    # strict=False: a decoder has extra parameters (encoder attention, layer_norm15)
    # that an MLM-pretrained model does not provide
    result = decoder.load_state_dict(state_dict, strict=False)
    if len(result.missing_keys) > 0:
        logger.info("%s: %i parameters not found in the reloaded model (kept at initial value)." % (name, len(result.missing_keys)))
    if len(result.unexpected_keys) > 0:
        logger.warning("%s: ignoring %i unexpected parameters in the reloaded model." % (name, len(result.unexpected_keys)))


def freeze_encoder(encoder):
    """
    Freeze all the parameters of the encoder, except those whose name ends in 'embeddings.weight': the word,
    position and language embeddings stay trainable (behaviour inherited from XLM / the earlier runs).
    """
    for p in encoder.parameters():
        p.requires_grad = False
    for name, p in encoder.named_parameters():
        if name.endswith('embeddings.weight'):
            p.requires_grad = True


def n_trainable_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def build_model(params, dico):
    """
    Build model: a single encoder-only model (MLM / CLM pretraining), or an
    (encoder, decoder) pair with one decoder shared by all languages.
    """
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            model.load_state_dict(reload_state_dict(params.reload_model, ('model',)))

        logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % n_trainable_params(model))

        return to_device(model)

    encoder, decoders = _build_encoder_decoders(params, dico, n_decoders=1)
    return to_device(encoder), to_device(decoders[0])


def build_model_separate(params, dico):
    """
    Build an encoder and two decoders: a source decoder (for the first language
    of --lgs) and a target decoder (for the second one).
    """
    encoder, (src_decoder, tgt_decoder) = _build_encoder_decoders(params, dico, n_decoders=2)
    return to_device(encoder), to_device(src_decoder), to_device(tgt_decoder)


def _build_encoder_decoders(params, dico, n_decoders):
    # build (the order of the constructions is kept, so random initialisations are unchanged)
    encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
    decoders = [TransformerModel(params, dico, is_encoder=False, with_output=True) for _ in range(n_decoders)]

    # reload pretrained word embeddings
    if params.reload_emb != '':
        word2id, embeddings = load_embeddings(params.reload_emb, params)
        for model in [encoder] + decoders:
            set_pretrain_emb(model, dico, word2id, embeddings)

    # reload a pretrained model (the same decoder checkpoint initialises every decoder)
    if params.reload_model != '':
        enc_path, dec_path = params.reload_model.split(',')
        assert not (enc_path == '' and dec_path == '')
        if enc_path != '':
            reload_encoder(encoder, enc_path)
        if dec_path != '':
            names = ['decoder'] if n_decoders == 1 else ['source decoder', 'target decoder']
            for decoder, name in zip(decoders, names):
                reload_decoder(decoder, dec_path, name)

    # freeze the params in the encoder (the word / position / language embeddings stay trainable)
    if params.freeze_encoder:
        freeze_encoder(encoder)

    logger.debug("Encoder: {}".format(encoder))
    logger.info("Number of parameters (encoder): %i" % n_trainable_params(encoder))
    if n_decoders == 1:
        logger.debug("Decoder: {}".format(decoders[0]))
        logger.info("Number of parameters (decoder): %i" % n_trainable_params(decoders[0]))
    else:
        for decoder, name in zip(decoders, ['source', 'target']):
            logger.debug("Decoder: {}".format(decoder))
            logger.info("Number of parameters (%s decoder): %i" % (name, n_trainable_params(decoder)))

    return encoder, decoders
