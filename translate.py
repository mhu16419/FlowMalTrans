#!/usr/bin/env python
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Translate (BPE-encoded) assembly text with a trained FlowMalTrans / XLM
encoder-decoder checkpoint (periodic-*.pth written by train*.py).

This single script replaces the former translate.py (stdin), translate_batch.py
and translate_faster.py (folder based, hard-coded paths).

Input must be tokenised/BPE-encoded exactly like the training data, one
sentence (assembly basic block) per line, tokens separated by white space.

Three modes
-----------
  1. one file
       python translate.py --model_path dumped/arm32-x86/<id>/periodic-100.pth \
           --src_lang arm32 --tgt_lang x86 \
           --src_path in.bpe --output_path out.bpe

  2. a folder (one output file per input file, same file name; existing
     outputs are skipped, so an interrupted run can simply be restarted)
       python translate.py --model_path ... --src_lang arm32 --tgt_lang x86 \
           --src_folder data/arm32_bpe --output_folder output/arm32_translation/arm32_benign \
           --batch_size 300 --max_len 200

  3. stdin -> stdout (or --output_path)
       cat in.bpe | python translate.py --model_path ... --src_lang arm32 --tgt_lang x86

  Show what a checkpoint contains (languages, decoders, flow modules) and exit:
       python translate.py --model_path ... --info

Which decoder is used
---------------------
Checkpoints trained with --separate_decoders true contain `src_decoder` and
`tgt_decoder`.  In EncDecTrainer these names refer to the POSITION of the
language in `--lgs`, not to the direction of a particular translation:
`src_decoder` generates lgs[0] and `tgt_decoder` generates lgs[1]
(xlm/trainer.py, mt_step / bt_step).  With `--lgs arm32-x86`:
    arm32 -> x86   uses tgt_decoder
    x86 -> arm32   uses src_decoder
This script selects the decoder from `--tgt_lang` and the `lgs` stored in the
checkpoint, so both directions work.  Checkpoints with a single shared decoder
(key `decoder`) are supported as well.

Checkpoints trained with a sentence-level latent code (`use_latent_code`, flows)
additionally store `shared_h2z_projection`, `src_flow` and `tgt_flow`.  The
latent code z is recomputed at translation time exactly as in
EncDecTrainer.bt_step (pooling -> projection -> src flow -> tgt flow) and is
passed to the decoder.

Notes
-----
* Lines with more than --max_len tokens cannot be translated reliably (the model
  was trained with --max_len 200).  --long_lines selects what happens to them:
  `drop` (default, behaviour of the original batch scripts: the line is removed
  from the output, so the output has fewer lines than the input), `empty` (an
  empty line is written, line numbers stay aligned -- use this when you want to
  compute BLEU against a reference), `truncate` (only the first --max_len tokens
  are translated).  Blank input lines always yield blank output lines.
* Input files are never modified or deleted.
* Output files are written atomically (temp file + rename), so a crashed run
  never leaves a partial file that would be skipped on restart.
* Checkpoints are python pickles: only load checkpoints you trust.
"""

import argparse
import contextlib
import logging
import os
import sys
import time

import torch
import torch.nn as nn

# the xlm package prints a FAISS notice on import; keep stdout clean for the stdin->stdout mode
with contextlib.redirect_stdout(sys.stderr):
    from xlm.utils import AttrDict, get_latent_code
    from xlm.data.dictionary import Dictionary
    from xlm.model.transformer import TransformerModel, N_MAX_POSITIONS
    from xlm.model.flows import create_normalizing_flows

logger = logging.getLogger("translate")

# attributes that TransformerModel / PredLayer read but which checkpoints written by
# older versions of the code do not contain
PARAM_DEFAULTS = {
    'attention_setting': 'none',
    'use_decoder_lang_emb': True,
    'use_lang_emb': True,
    'use_latent_code': False,
    'latent_concat': True,
    'use_weight_network': False,
    'use_flow': False,
    'use_h2z_matrix': False,
    'separate_decoders': False,
    'use_memory': False,
    'asm': False,
}


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #

class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults in --help, but not empty / None / False ones."""

    def _get_help_string(self, action):
        if action.default in (None, '', False):
            return action.help
        return super()._get_help_string(action)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Translate BPE-encoded assembly with a trained FlowMalTrans / XLM checkpoint",
        formatter_class=_HelpFormatter)

    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint (periodic-*.pth)")
    parser.add_argument("--src_lang", type=str, default="", help="Source language, e.g. arm32")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language, e.g. x86")
    parser.add_argument("--info", action="store_true", help="Print what the checkpoint contains and exit")

    io_group = parser.add_argument_group(
        "input / output (choose one: --src_path/--output_path, --src_folder/--output_folder, or stdin)")
    io_group.add_argument("--src_path", type=str, default="", help="Source file to translate")
    io_group.add_argument("--output_path", type=str, default="",
                          help="Output file (single-file mode; with stdin input it defaults to stdout)")
    io_group.add_argument("--src_folder", type=str, default="", help="Folder with source files (one per binary)")
    io_group.add_argument("--output_folder", type=str, default="", help="Folder for the translated files")
    io_group.add_argument("--overwrite", action="store_true",
                          help="Overwrite existing outputs instead of skipping / refusing")

    parser.add_argument("--batch_size", type=int, default=64, help="Sentences per batch")
    parser.add_argument("--max_len", type=int, default=200,
                        help="Maximum number of tokens of an input line (use the --max_len of training)")
    parser.add_argument("--long_lines", type=str, default="drop", choices=["drop", "empty", "truncate"],
                        help="What to do with lines longer than --max_len")
    parser.add_argument("--no_sort", action="store_true",
                        help="Keep the input order when batching (default: sort by length, which is much faster; "
                             "results are identical except for runaway generations that hit the length cap, and "
                             "for models with a latent code, whose sentence code pools over the padded batch and "
                             "therefore depends on the batch composition)")

    parser.add_argument("--beam_size", type=int, default=1, help="1 = greedy decoding (default)")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Beam search length penalty")
    parser.add_argument("--early_stopping", action="store_true", help="Beam search early stopping")
    parser.add_argument("--max_gen_ratio", type=float, default=1.3,
                        help="Generation stops after ratio * (longest source in batch + 2) + max_gen_extra tokens")
    parser.add_argument("--max_gen_extra", type=int, default=10, help="See --max_gen_ratio")

    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--verbose", action="store_true", help="Also log debug information (e.g. unknown-token rate)")
    return parser


def check_args(parser, args):
    folder_mode = bool(args.src_folder or args.output_folder)
    file_mode = bool(args.src_path)
    if args.info:
        return
    if not args.src_lang or not args.tgt_lang:
        parser.error("--src_lang and --tgt_lang are required")
    if args.src_lang == args.tgt_lang:
        parser.error("--src_lang and --tgt_lang must differ")
    if folder_mode and file_mode:
        parser.error("use either --src_path or --src_folder, not both")
    if folder_mode and not (args.src_folder and args.output_folder):
        parser.error("--src_folder and --output_folder must be given together")
    if file_mode and not args.output_path:
        parser.error("--src_path needs --output_path")
    if args.output_path and folder_mode:
        parser.error("--output_path cannot be combined with --src_folder/--output_folder")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.beam_size < 1:
        parser.error("--beam_size must be >= 1")
    if not 1 <= args.max_len <= N_MAX_POSITIONS - 2:
        parser.error("--max_len must be between 1 and %d (positional embedding limit)" % (N_MAX_POSITIONS - 2))


# --------------------------------------------------------------------------- #
# checkpoint handling
# --------------------------------------------------------------------------- #

def load_checkpoint(path):
    """Load a checkpoint on the CPU (torch >= 2.6 defaults to weights_only=True, which cannot
    unpickle the params / dictionary stored in our checkpoints)."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location='cpu')


def strip_module_prefix(state_dict):
    if len(state_dict) > 0 and all(k.startswith('module.') for k in state_dict):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def describe_checkpoint(ck):
    p = ck['params']
    lines = ["epoch / n_total_iter : %s / %s" % (ck.get('epoch'), ck.get('n_total_iter')),
             "top-level keys       : %s" % ", ".join(k for k in ck if not k.endswith('_optimizer')),
             "languages (lgs)      : %s   (lang2id = %s)" % (p.get('lgs'), p.get('lang2id')),
             "vocabulary size      : %s" % p.get('n_words'),
             "emb_dim / layers     : %s / %s (heads %s)" % (p.get('emb_dim'), p.get('n_layers'), p.get('n_heads')),
             "separate decoders    : %s" % p.get('separate_decoders', False),
             "latent code / flow   : use_latent_code=%s use_flow=%s flow_type=%s z_size=%s" % (
                 p.get('use_latent_code', False), p.get('use_flow', False), p.get('flow_type'), p.get('z_size')),
             "max_len (training)   : %s" % p.get('max_len')]
    lgs = str(p.get('lgs', '')).split('-')
    if 'tgt_decoder' in ck and len(lgs) == 2:
        lines.append("decoders             : src_decoder generates %s, tgt_decoder generates %s" % (lgs[0], lgs[1]))
        lines.append("translation          : %s -> %s uses tgt_decoder, %s -> %s uses src_decoder" % (
            lgs[0], lgs[1], lgs[1], lgs[0]))
    return "\n".join(lines)


class Translator(object):
    """Encoder + (selected) decoder + optional latent-code modules, ready to translate."""

    def __init__(self, ck, src_lang, tgt_lang, device, args):
        if 'model' in ck and 'encoder' not in ck:
            raise ValueError("This is an encoder-only (MLM/CLM) checkpoint, it cannot translate.")
        for key in ('encoder', 'dico_id2word', 'dico_word2id', 'dico_counts', 'params'):
            if key not in ck:
                raise ValueError("Not a translation checkpoint: key '%s' is missing (found: %s)" % (key, list(ck.keys())))

        params = AttrDict(ck['params'])
        for k, v in PARAM_DEFAULTS.items():
            if k not in params:
                params[k] = v
        self.params = params
        self.device = device
        self.args = args

        lang2id = params.lang2id
        for lang in (src_lang, tgt_lang):
            if lang not in lang2id:
                raise ValueError("Language '%s' is not supported by this checkpoint (supported: %s)"
                                 % (lang, ", ".join(sorted(lang2id))))
        self.src_id = lang2id[src_lang]
        self.tgt_id = lang2id[tgt_lang]

        self.dico = Dictionary(ck['dico_id2word'], ck['dico_word2id'], ck['dico_counts'])
        self.pad_index = params.pad_index
        self.eos_index = params.eos_index
        self.unk_index = self.dico.unk_index

        # ---- which decoder / flow belongs to which language? ----
        # EncDecTrainer: src_decoder / src_flow <-> lgs[0],  tgt_decoder / tgt_flow <-> lgs[1]
        lgs = str(params.get('lgs', '')).split('-')
        src_flow_key = tgt_flow_key = None
        if len(lgs) == 2 and src_lang in lgs and tgt_lang in lgs:
            src_flow_key = 'src_flow' if src_lang == lgs[0] else 'tgt_flow'
            tgt_flow_key = 'src_flow' if tgt_lang == lgs[0] else 'tgt_flow'
        if 'tgt_decoder' in ck:
            if len(lgs) != 2:
                raise ValueError("separate decoders need exactly two languages in lgs, got '%s'" % params.get('lgs'))
            if src_lang not in lgs or tgt_lang not in lgs:
                raise ValueError("src/tgt language must be in lgs=%s" % params.lgs)
            dec_key = 'src_decoder' if tgt_lang == lgs[0] else 'tgt_decoder'
        elif 'decoder' in ck:
            dec_key = 'decoder'
        else:
            raise ValueError("No decoder found in checkpoint (keys: %s)" % list(ck.keys()))
        logger.info("Translating %s -> %s with %s (lang ids %d -> %d)", src_lang, tgt_lang, dec_key,
                    self.src_id, self.tgt_id)

        self.encoder = TransformerModel(params, self.dico, is_encoder=True, with_output=True)
        self.encoder.load_state_dict(strip_module_prefix(ck['encoder']))
        self.decoder = TransformerModel(params, self.dico, is_encoder=False, with_output=True)
        self.decoder.load_state_dict(strip_module_prefix(ck[dec_key]))
        self.encoder.to(device).eval()
        self.decoder.to(device).eval()

        # ---- sentence-level latent code (only for models trained with use_latent_code) ----
        self.use_z = bool(params.use_latent_code)
        self.h2z = self.src_flow = self.tgt_flow = None
        if self.use_z:
            if params.use_h2z_matrix:
                w = ck['shared_h2z_projection']['weight']
                self.h2z = nn.Linear(w.shape[1], w.shape[0])
                self.h2z.load_state_dict(ck['shared_h2z_projection'])
                self.h2z.to(device).eval()
            if params.use_flow:
                if src_flow_key is None:
                    raise ValueError("flows need a two-language checkpoint (lgs='%s') containing both languages"
                                     % params.get('lgs'))
                flow_kwargs = {'hiddenflow_layers': params.hiddenflow_layers,
                               'hiddenflow_units': params.hiddenflow_units,
                               'hiddenflow_flow_nums': params.hiddenflow_flow_nums}
                self.src_flow = self._load_flow(params, flow_kwargs, ck[src_flow_key])
                self.tgt_flow = self._load_flow(params, flow_kwargs, ck[tgt_flow_key])
            logger.info("Latent code enabled (z_size=%s, h2z=%s, flows=%s)", params.z_size,
                        self.h2z is not None, self.src_flow is not None)

    def _load_flow(self, params, kwargs, state):
        flow = create_normalizing_flows(flow_type=params.flow_type, z_size=params.z_size, dropout=0, kwargs=kwargs)
        flow.load_state_dict(strip_module_prefix(state))
        return flow.to(self.device).eval()

    # ---- latent code, mirrors EncDecTrainer.bt_step ----
    def latent_code(self, enc):
        """enc: (bs, slen, dim) encoder output -> z (bs, z_size) for the target language."""
        if 'concat_h' not in self.params:  # very old checkpoints
            self.params.concat_h = True
        z = get_latent_code(enc, self.params, self.h2z)
        if self.src_flow is not None:
            z_k, _ = self.src_flow(z)
            eps = z_k[-1]
            z_k_tgt, _ = self.tgt_flow(eps, direction='gen')
            z = z_k_tgt[-1]
        return z

    # ---- translation ----
    def _make_batch(self, id_lists):
        lengths = torch.LongTensor([len(s) + 2 for s in id_lists])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        batch[0] = self.eos_index
        for j, s in enumerate(id_lists):
            if len(s) > 0:
                batch[1:len(s) + 1, j] = torch.LongTensor(s)
            batch[len(s) + 1, j] = self.eos_index
        return batch, lengths

    @torch.no_grad()
    def _translate_batch(self, id_lists):
        args = self.args
        batch, lengths = self._make_batch(id_lists)
        langs = batch.clone().fill_(self.src_id)
        batch, lengths, langs = batch.to(self.device), lengths.to(self.device), langs.to(self.device)

        enc = self.encoder('fwd', x=batch, lengths=lengths, langs=langs, causal=False)
        enc = enc.transpose(0, 1)
        z = self.latent_code(enc) if self.use_z else None
        max_gen = min(int(args.max_gen_ratio * lengths.max().item() + args.max_gen_extra), N_MAX_POSITIONS)
        if args.beam_size > 1:
            decoded, _ = self.decoder.generate_beam(enc, lengths, self.tgt_id, beam_size=args.beam_size,
                                                    length_penalty=args.length_penalty,
                                                    early_stopping=args.early_stopping, max_len=max_gen, z=z)
        else:
            decoded, _ = self.decoder.generate(enc, lengths, self.tgt_id, max_len=max_gen, z=z)

        results = []
        for row in decoded.t().tolist():
            row = row[1:]  # drop the leading <eos> that serves as <bos>
            if self.eos_index in row:
                row = row[:row.index(self.eos_index)]
            results.append(" ".join(self.dico.id2word[t] for t in row))
        return results

    def translate(self, token_lists):
        """token_lists: list of list[str] (non-empty lists) -> list of translated strings, same order."""
        unk = self.unk_index
        w2i = self.dico.word2id
        ids = [[w2i.get(w, unk) for w in toks] for toks in token_lists]
        n_unk = sum(i == unk for s in ids for i in s)
        n_tok = sum(len(s) for s in ids)
        order = list(range(len(ids)))
        if not self.args.no_sort:
            order.sort(key=lambda i: -len(ids[i]))
        out = [None] * len(ids)
        bs = self.args.batch_size
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            for i, hyp in zip(idx, self._translate_batch([ids[i] for i in idx])):
                out[i] = hyp
        return out, n_unk, n_tok


# --------------------------------------------------------------------------- #
# file level helpers
# --------------------------------------------------------------------------- #

def prepare_lines(lines, args):
    """Split lines into the ones to translate and a per-line plan.

    Returns (to_translate, plan) where plan[i] is ('tr', k) for the k-th item of to_translate,
    ('blank',) or ('long', n_tokens).  Long lines under `truncate` are cut and become 'tr'.
    """
    to_translate, plan = [], []
    for line in lines:
        toks = line.split()
        if len(toks) == 0:
            plan.append(('blank',))
        elif len(toks) > args.max_len:
            if args.long_lines == 'truncate':
                plan.append(('tr', len(to_translate)))
                to_translate.append(toks[:args.max_len])
            else:
                plan.append(('long', len(toks)))
        else:
            plan.append(('tr', len(to_translate)))
            to_translate.append(toks)
    return to_translate, plan


def assemble_output(plan, translations, args):
    out = []
    for item in plan:
        if item[0] == 'tr':
            out.append(translations[item[1]])
        elif item[0] == 'blank':
            out.append("")
        elif args.long_lines == 'empty':  # 'long' under drop / empty
            out.append("")
        # under 'drop' the line is omitted
    return out


def read_lines(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return [line.rstrip('\n') for line in f]


def write_lines_atomic(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")
    os.replace(tmp, path)


def translate_lines(translator, lines, args, name="input"):
    to_translate, plan = prepare_lines(lines, args)
    n_long = sum(1 for p in plan if p[0] == 'long')
    n_trunc = sum(1 for l in lines if len(l.split()) > args.max_len) if args.long_lines == 'truncate' else 0
    if n_long:
        logger.info("%s: %d line(s) longer than --max_len=%d %s", name, n_long, args.max_len,
                    "dropped" if args.long_lines == 'drop' else "replaced by empty lines")
    if n_trunc:
        logger.info("%s: %d line(s) longer than --max_len=%d truncated", name, n_trunc, args.max_len)
    if to_translate:
        translations, n_unk, n_tok = translator.translate(to_translate)
        logger.debug("%s: %d tokens, %.2f%% unknown", name, n_tok, 100.0 * n_unk / max(n_tok, 1))
    else:
        translations, n_unk, n_tok = [], 0, 0
    return assemble_output(plan, translations, args), len(to_translate), n_unk, n_tok


def translate_file(translator, src_path, output_path, args):
    t0 = time.time()
    lines = read_lines(src_path)
    out, n_tr, n_unk, n_tok = translate_lines(translator, lines, args, name=os.path.basename(src_path))
    if n_tr == 0:
        logger.warning("%s: nothing to translate (%d lines), no output written", src_path, len(lines))
        return False
    write_lines_atomic(output_path, out)
    logger.info("%s: %d/%d lines translated, %.1f%% unknown source tokens, %.1fs", os.path.basename(src_path), n_tr,
                len(lines), 100.0 * n_unk / max(n_tok, 1), time.time() - t0)
    return True


def run_folder(translator, args):
    src_folder, output_folder = args.src_folder, args.output_folder
    if not os.path.isdir(src_folder):
        raise FileNotFoundError("--src_folder %s is not a directory" % src_folder)
    if os.path.realpath(src_folder) == os.path.realpath(output_folder):
        raise ValueError("--output_folder must differ from --src_folder")
    os.makedirs(output_folder, exist_ok=True)
    names = sorted(n for n in os.listdir(src_folder)
                   if not n.startswith('.') and os.path.isfile(os.path.join(src_folder, n)))
    n_done = n_skipped = n_empty = 0
    for k, name in enumerate(names, 1):
        output_path = os.path.join(output_folder, name)
        if os.path.exists(output_path) and not args.overwrite:
            logger.info("[%d/%d] skipping %s (output exists)", k, len(names), name)
            n_skipped += 1
            continue
        if translate_file(translator, os.path.join(src_folder, name), output_path, args):
            n_done += 1
        else:
            n_empty += 1
    logger.info("Done: %d translated, %d skipped (existing), %d without translatable lines, of %d files",
                n_done, n_skipped, n_empty, len(names))


def run_stdin(translator, args):
    lines = [line.rstrip('\n') for line in sys.stdin]
    logger.info("Read %d lines from stdin. Translating ...", len(lines))
    out, n_tr, _, _ = translate_lines(translator, lines, args, name="stdin")
    if args.output_path:
        if os.path.exists(args.output_path) and not args.overwrite:
            raise FileExistsError("%s exists (use --overwrite)" % args.output_path)
        write_lines_atomic(args.output_path, out)
    else:
        sys.stdout.write("".join(line + "\n" for line in out))
        sys.stdout.flush()


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("--device %s requested but CUDA is not available" % name)
    return device


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    check_args(parser, args)
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if not os.path.isfile(args.model_path):
        parser.error("--model_path %s does not exist" % args.model_path)
    ck = load_checkpoint(args.model_path)
    if args.info:
        print(describe_checkpoint(ck))
        return 0

    if args.src_path:
        if os.path.realpath(args.src_path) == os.path.realpath(args.output_path):
            parser.error("--output_path must differ from --src_path")
        if os.path.exists(args.output_path) and not args.overwrite:
            parser.error("%s already exists (use --overwrite)" % args.output_path)
    if args.src_folder and os.path.realpath(args.src_folder) == os.path.realpath(args.output_folder):
        parser.error("--output_folder must differ from --src_folder")

    # fail early (before the model is built and the input is translated) if the output cannot be written
    out_dir = None
    if args.src_path or (not args.src_folder and args.output_path):
        out_dir = os.path.dirname(os.path.abspath(args.output_path))
    elif args.src_folder:
        out_dir = os.path.abspath(args.output_folder)
    if out_dir is not None:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            parser.error("cannot create the output directory %s: %s" % (out_dir, e))

    try:
        return run(args, ck, parser)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        sys.exit("ERROR: %s" % e)


def run(args, ck, parser):
    device = resolve_device(args.device)
    translator = Translator(ck, args.src_lang, args.tgt_lang, device, args)
    del ck
    if args.max_len > translator.params.get('max_len', args.max_len):
        logger.warning("--max_len=%d is larger than the max_len=%s used for training; quality may degrade",
                       args.max_len, translator.params.get('max_len'))

    if args.src_folder:
        run_folder(translator, args)
    elif args.src_path:
        translate_file(translator, args.src_path, args.output_path, args)
    else:
        run_stdin(translator, args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
