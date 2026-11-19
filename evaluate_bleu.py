#!/usr/bin/env python
"""
BLEU between translated assembly and reference assembly, computed as in Section 5.3 of the paper:
SacreBLEU with tokenization 'none' (every line is split on white space, which is the right tokenization for
BPE-encoded assembly), add-one smoothing and otherwise default settings, one corpus-level BLEU per binary (= per
file; the lines of a file are its basic blocks), averaged over all binaries.

Replacement for BLEU_Evaluation.ipynb, whose bleu_score() received plain strings instead of token lists and therefore
computed a character-overlap score (see the git history).

Two modes
    python evaluate_bleu.py --hyp_path out.bpe --ref_path ref.bpe
    python evaluate_bleu.py --hyp_folder translated/ --ref_folder reference/        # same file names in both

Hypothesis and reference must be line aligned (block i of the translation <-> block i of the reference). Translate
with `translate.py --long_lines empty` so that lines dropped for being too long do not shift the alignment.
Files whose line counts differ are skipped (default) or compared up to the shorter length (--mismatch truncate);
they are always reported.

Scores are printed on the 0-1 scale of the paper (sacrebleu reports 0-100).
"""

import argparse
import os
import sys

from sacrebleu.metrics import BLEU


def read_lines(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return [line.rstrip('\n') for line in f]


def make_metric():
    # "tokenization to None, add-one smoothing" (paper, Sec. 5.3); sacrebleu's 'add-k' with k=1 is add-one smoothing
    return BLEU(tokenize='none', smooth_method='add-k', smooth_value=1, force=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hyp_path', default='', help='translated file')
    ap.add_argument('--ref_path', default='', help='reference file (same number of lines)')
    ap.add_argument('--hyp_folder', default='', help='folder with translated files')
    ap.add_argument('--ref_folder', default='', help='folder with reference files of the same names')
    ap.add_argument('--mismatch', choices=['skip', 'truncate', 'error'], default='skip',
                    help='what to do with files whose hypothesis / reference line counts differ')
    ap.add_argument('--per_file', action='store_true', help='also print the BLEU of every file')
    args = ap.parse_args()

    if bool(args.hyp_path) == bool(args.hyp_folder) or bool(args.ref_path) == bool(args.ref_folder) \
            or bool(args.hyp_path) != bool(args.ref_path):
        ap.error('give either --hyp_path/--ref_path or --hyp_folder/--ref_folder')

    if args.hyp_path:
        pairs = [(os.path.basename(args.hyp_path), args.hyp_path, args.ref_path)]
    else:
        names = sorted(n for n in os.listdir(args.hyp_folder)
                       if os.path.isfile(os.path.join(args.hyp_folder, n)) and not n.startswith('.'))
        pairs = []
        for n in names:
            ref = os.path.join(args.ref_folder, n)
            if os.path.isfile(ref):
                pairs.append((n, os.path.join(args.hyp_folder, n), ref))
            else:
                print('no reference for %s, skipped' % n, file=sys.stderr)

    bleu = make_metric()
    scores, all_hyp, all_ref, n_mismatch = [], [], [], 0
    for name, hyp_path, ref_path in pairs:
        hyp, ref = read_lines(hyp_path), read_lines(ref_path)
        if len(hyp) != len(ref):
            n_mismatch += 1
            print('line count mismatch in %s: hyp %d vs ref %d' % (name, len(hyp), len(ref)), file=sys.stderr)
            if args.mismatch == 'error':
                sys.exit(1)
            if args.mismatch == 'skip':
                continue
            m = min(len(hyp), len(ref))
            hyp, ref = hyp[:m], ref[:m]
        if not hyp:
            print('%s is empty, skipped' % name, file=sys.stderr)
            continue
        score = bleu.corpus_score(hyp, [ref]).score / 100.
        scores.append(score)
        all_hyp += hyp
        all_ref += ref
        if args.per_file:
            print('%s\t%d lines\tBLEU %.4f' % (name, len(hyp), score))

    if not scores:
        print('nothing to evaluate', file=sys.stderr)
        sys.exit(1)
    pooled = bleu.corpus_score(all_hyp, [all_ref]).score / 100.
    print('%d binaries (files), %d lines, %d file(s) skipped or truncated because of mismatching line counts'
          % (len(scores), len(all_hyp), n_mismatch))
    print('BLEU, mean over binaries (as in the paper): %.4f' % (sum(scores) / len(scores)))
    print('BLEU, all lines pooled into one corpus:     %.4f' % pooled)


if __name__ == '__main__':
    main()
