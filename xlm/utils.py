# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import sys
import pickle
import random
import argparse
import numpy as np
import torch
from torch.nn import functional as F

from .logger import create_logger


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = './dumped'  # used when --dump_path is empty
DYNAMIC_COEFF = ['lambda_clm', 'lambda_mlm', 'lambda_pc', 'lambda_ae', 'lambda_mt', 'lambda_bt']


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    with open(os.path.join(params.dump_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(dump_path, params.exp_name)
    os.makedirs(sweep_path, exist_ok=True)

    # create an ID for the job if it is not given in the parameters.
    # inside a SLURM job it is the SLURM job ID, otherwise it is randomly generated
    if params.exp_id == '':
        exp_id = os.environ.get('SLURM_JOB_ID')
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    os.makedirs(params.dump_path, exist_ok=True)


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]


def restore_segmentation(path):
    """
    Take a file segmented with BPE and restore it to its original segmentation.
    """
    assert os.path.isfile(path)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(re.sub(r'(@@ )|(@@ ?$)', '', line) for line in lines))


def parse_lambda_config(params):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000 iterations, then will linearly increase to 1 until iteration 2000
    """
    for name in DYNAMIC_COEFF:
        x = getattr(params, name)
        split = x.split(',')
        if len(split) == 1:
            setattr(params, name, float(x))
            setattr(params, name + '_config', None)
        else:
            split = [s.split(':') for s in split]
            assert all(len(s) == 2 for s in split)
            assert all(k.isdigit() for k, _ in split)
            assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
            setattr(params, name, float(split[0][1]))
            setattr(params, name + '_config', [(int(k), float(v)) for k, v in split])


def get_lambda_value(config, n_iter):
    """
    Compute a lambda value according to its schedule configuration.
    """
    ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
    if len(ranges) == 0:
        assert n_iter >= config[-1][0]
        return config[-1][1]
    assert len(ranges) == 1
    i = ranges[0]
    x_a, y_a = config[i]
    x_b, y_b = config[i + 1]
    return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)


def update_lambdas(params, n_iter):
    """
    Update all lambda coefficients.
    """
    for name in DYNAMIC_COEFF:
        config = getattr(params, name + '_config')
        if config is not None:
            setattr(params, name, get_lambda_value(config, n_iter))


def set_sampling_probs(data, params):
    """
    Set the probability of sampling specific languages / language pairs during training.
    """
    coeff = params.lg_sampling_factor
    if coeff == -1:
        return
    assert coeff > 0

    # monolingual data
    params.mono_list = [k for k, v in data['mono_stream'].items() if 'train' in v]
    if len(params.mono_list) > 0:
        probs = np.array([1.0 * len(data['mono_stream'][lang]['train']) for lang in params.mono_list])
        probs /= probs.sum()
        probs = np.array([p ** coeff for p in probs])
        probs /= probs.sum()
        params.mono_probs = probs

    # parallel data
    params.para_list = [k for k, v in data['para'].items() if 'train' in v]
    if len(params.para_list) > 0:
        probs = np.array([1.0 * len(data['para'][(l1, l2)]['train']) for (l1, l2) in params.para_list])
        probs /= probs.sum()
        probs = np.array([p ** coeff for p in probs])
        probs /= probs.sum()
        params.para_probs = probs


def concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, pad_idx, eos_idx, reset_positions):
    """
    Concat batches with different languages.
    """
    assert reset_positions is False or lang1_id != lang2_id
    lengths = len1 + len2
    if not reset_positions:
        lengths -= 1
    slen, bs = lengths.max().item(), lengths.size(0)

    x = x1.new(slen, bs).fill_(pad_idx)
    x[:len1.max().item()].copy_(x1)
    positions = torch.arange(slen)[:, None].repeat(1, bs).to(x1.device)
    langs = x1.new(slen, bs).fill_(lang1_id)

    for i in range(bs):
        l1 = len1[i] if reset_positions else len1[i] - 1
        x[l1:l1 + len2[i], i].copy_(x2[:len2[i], i])
        if reset_positions:
            positions[l1:, i] -= len1[i]
        langs[l1:, i] = lang2_id

    assert (x == eos_idx).long().sum().item() == (4 if reset_positions else 3) * bs

    return x, lengths, positions, langs


def truncate(x, lengths, max_len, eos_index):
    """
    Truncate long sentences.
    """
    if lengths.max().item() > max_len:
        x = x[:max_len].clone()
        lengths = lengths.clone()
        for i in range(len(lengths)):
            if lengths[i] > max_len:
                lengths[i] = max_len
                x[max_len - 1, i] = eos_index
    return x, lengths


def shuf_order(langs, params=None, n=5):
    """
    Randomize training order.
    """
    if len(langs) == 0:
        return []

    if params is None:
        return [langs[i] for i in np.random.permutation(len(langs))]

    # sample monolingual and parallel languages separately
    mono = [l1 for l1, l2 in langs if l2 is None]
    para = [(l1, l2) for l1, l2 in langs if l2 is not None]

    # uniform / weighted sampling
    if params.lg_sampling_factor == -1:
        p_mono = None
        p_para = None
    else:
        p_mono = np.array([params.mono_probs[params.mono_list.index(k)] for k in mono])
        p_para = np.array([params.para_probs[params.para_list.index(tuple(sorted(k)))] for k in para])
        p_mono = p_mono / p_mono.sum()
        p_para = p_para / p_para.sum()

    s_mono = [mono[i] for i in np.random.choice(len(mono), size=min(n, len(mono)), p=p_mono, replace=True)] if len(mono) > 0 else []
    s_para = [para[i] for i in np.random.choice(len(para), size=min(n, len(para)), p=p_para, replace=True)] if len(para) > 0 else []

    assert len(s_mono) + len(s_para) > 0
    return [(lang, None) for lang in s_mono] + s_para


def get_latent_code(enc, params, h2z_projection=None):
    """
    Sentence-level latent code z from encoder states.
    enc: (batch_size, seq_len, emb_dim). The enabled representations (mean pooling, max pooling,
    first-token / CLS embedding, always in this order) are concatenated (params.concat_h) or summed,
    then optionally projected to z_size with `h2z_projection` (params.use_h2z_matrix).
    """
    if not (params.use_mean_pooling or params.use_max_pooling or params.use_CSL):
        raise ValueError('There must at least be one method to obtain the sentence level latent code')
    hs = []
    if params.use_mean_pooling:
        hs.append(F.avg_pool1d(enc.transpose(1, 2).contiguous(), enc.size(1)).squeeze(2))  # [B, emb_dim]
    if params.use_max_pooling:
        hs.append(F.max_pool1d(enc.transpose(1, 2).contiguous(), enc.size(1)).squeeze(2))
    if params.use_CSL:
        hs.append(enc[:, 0])
    h_enc = torch.cat(hs, dim=-1) if params.concat_h else sum(hs)
    return h2z_projection(h_enc) if h2z_projection is not None else h_enc


def find_modules(module, module_name, module_instance, found):
    """
    Recursively find all instances of a specific module inside a module.
    """
    if isinstance(module, module_instance):
        found.append((module_name, module))
    else:
        for name, child in module.named_children():
            name = ('%s[%s]' if name.isdigit() else '%s.%s') % (module_name, name)
            find_modules(child, name, module_instance, found)
