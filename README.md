# FlowMalTrans: Unsupervised Binary Code Translation for Malware Detection Using Flow-Adapter Architecture

[![EMNLP 2025 Findings](https://img.shields.io/badge/EMNLP%202025-Findings-blue)](https://arxiv.org/abs/2508.20212)
[![arXiv](https://img.shields.io/badge/arXiv-2508.20212-b31b1b)](https://arxiv.org/abs/2508.20212)

Official code of **FlowMalTrans** (EMNLP 2025 Findings). [Paper (arXiv)](https://arxiv.org/abs/2508.20212) · [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.173/)

Minghao Hu, Junzhe Wang, Weisen Zhao, Qiang Zeng, Lannan Luo — George Mason University

## Overview

Deep-learning malware detectors need many labelled samples *per instruction set architecture (ISA)*, but collecting them
for every ISA (ARM, MIPS, PowerPC, M68K, ...) is slow and expensive. FlowMalTrans avoids this: it learns an
**unsupervised neural translator** from assembly of a source ISA to X86-64, using only **benign, open-source
programs** (no malware, no parallel data). A malware detector is then trained **once on X86-64** and applied unchanged
to the translated code of any other ISA.

The translator is an unsupervised neural machine translation model (denoising auto-encoding + back-translation on top of
CLM / MLM pre-training) extended with a **flow adapter**: a normalizing flow per ISA maps the sentence-level latent code
of one ISA into the latent space of the other, so the target decoder is conditioned on a code that lives in *its own*
ISA's space. In the paper it is trained on the pairs {i386, ARM32, ARM64, MIPS32, PPC32, M68K} ⟷ X86-64 and evaluated in the
direction {ISA} → X86-64; it gives higher BLEU than UnsuperBinTrans and better detection AUC than UnsuperBinTrans,
UniMap, CrossIns2Vec and an IR-based baseline.

<p align="center">
  <img src="assets/overview.png" alt="FlowMalTrans workflow" width="1100"/>
</p>
<p align="center">
  <em>Figure 1: Applying FlowMalTrans. (1) Train the translator on benign mono-ISA data, (2) train a malware detector on
  X86-64 malware, (3) translate binaries of another ISA to X86-64 and classify them with the X86-64 detector.</em>
</p>

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [1. Prepare the data](#1-prepare-the-data)
- [2. Train the translator](#2-train-the-translator)
- [3. Translate](#3-translate)
- [4. Evaluate the translation (BLEU)](#4-evaluate-the-translation-bleu)
- [5. Malware detection](#5-malware-detection)
- [Smoke test](#smoke-test)
- [Results reported in the paper](#results-reported-in-the-paper)
- [Scope, limitations and known gaps](#scope-limitations-and-known-gaps)
- [Responsible use](#responsible-use)
- [Citation](#citation) · [License and acknowledgements](#license-and-acknowledgements)

## Repository layout

```
FlowMalTrans/
├── train.py                 # training entry point (translator; also plain MLM/CLM language models)
├── translate.py             # translation of BPE-encoded assembly (file / folder / stdin)
├── evaluate_bleu.py         # BLEU of translated vs. reference assembly (sacrebleu)
├── preprocess.py            # binarize a BPE text file into the .pth format the trainer reads
├── xlm/                     # the library: Transformer + flows, trainer, evaluator, data loading
│   ├── model/               #   transformer.py (encoder/decoder, latent-code injection), flows.py (scf, glow),
│   │                        #   memory/ + pretrain.py (XLM leftovers, not used by the paper)
│   ├── trainer.py           #   EncDecTrainer: CLM / MLM / denoising AE / back-translation / flow losses
│   ├── evaluation/          #   perplexity, accuracy, BLEU (multi-bleu.perl) during training
│   ├── data/                #   dictionary, datasets, loaders
│   └── utils.py, optim.py, slurm.py, logger.py   # experiment folder, optimizers, SLURM/multi-GPU init, logging
├── requirements.txt, setup.py  # dependencies; `pip install -e .`
├── LICENSE
├── scripts/
│   ├── train_arm32_x86.sh   #   the training recipe of the paper runs, as a command line
│   └── binarize.sh          #   run preprocess.py on every file of a data folder
├── data_prep/
│   ├── format_assembly.py   #   IDA .gdl -> normalized one-basic-block-per-line text (all six ISAs)
│   ├── bpe.sh               #   joint fastBPE learn/apply/vocab for an ISA pair
│   └── legacy_notebooks/    #   the original per-ISA notebooks and helper notebooks (kept for reference)
├── malware_detection/       # LSTM detector: train_lstm.py, test_lstm.py (+ lstm_common.py)
├── analysis/                # exploratory notebooks (dataset statistics, vocabulary overlap, opcode similarity)
├── tests/smoke_test.py      # end-to-end check on synthetic data (see "Smoke test")
└── assets/overview.png
```

Not included in this repository: datasets, trained checkpoints, malware samples (see [Responsible use](#responsible-use)).
`dumped/` (experiment output), `data/` and `output/` are git-ignored.

## Installation

```bash
git clone https://github.com/mhu16419/FlowMalTrans.git
cd FlowMalTrans
python -m venv .venv && source .venv/bin/activate        # Python >= 3.8 (setup.py); only 3.10 was tested
pip install -r requirements.txt                          # torch, numpy, sacrebleu (+ scikit-learn, matplotlib, jupyter)
pip install -e .                                         # optional: makes `import xlm` work from anywhere
```

* **A CUDA GPU is required for training** (the training code moves models and batches to the GPU). `translate.py` also
  runs on CPU (`--device cpu`, slow). Multi-GPU / SLURM code paths are inherited from XLM and were not tested; all
  experiments of the paper are single-GPU.
* The code was tested with Python 3.10, **torch 2.3.1 (CUDA 12.1)**, numpy 1.26, sacrebleu 2.6 (newer torch versions are
  untested; if you run into problems use `pip install torch==2.3.1`). NVIDIA apex is only needed for `--amp >= 0` /
  `--fp16 true` and is not required otherwise.
* [fastBPE](https://github.com/glample/fastBPE) is needed only for data preparation
  (`git clone https://github.com/glample/fastBPE && cd fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast`;
  the binary is `fastBPE/fast`). The disassembly step needs IDA Pro (commercial).
* `perl` is needed if you want BLEU during training (`--eval_bleu true`, uses `xlm/evaluation/multi-bleu.perl`).

## Pipeline at a glance

```
 binaries ──IDA Pro──▶ .gdl ──format_assembly.py──▶ normalized blocks ──bpe.sh──▶ BPE text ──binarize.sh──▶ *.pth
                                                                                                          │
                       train.py  (CLM/MLM pre-training, then DAE + back-translation [+ flow adapter]) ◀───┘
                           │  periodic-<epoch>.pth
                           ▼
 source-ISA BPE files ──translate.py──▶ translated (X86-64) files ──┬──▶ evaluate_bleu.py   (vs. parallel references)
                                                                    └──▶ malware_detection/test_lstm.py  (AUC, X86-64 LSTM)
                                                                                       ▲
                                  x86 benign/malware ──malware_detection/train_lstm.py ┘
```

## 1. Prepare the data

The translator is trained on **monolingual** assembly of the two ISAs (parallel data is only used for validation/test).
In the paper the training corpus consists of openssl-1.1.1p, binutils-2.34, findutils-4.8.0 and libgpg-error-1.45;
the test corpus (zlib-1.2.11, coreutils-9.0, diffutils-3.7) is disjoint. All programs are cross-compiled with GCC at
O0–O3 for each ISA and disassembled with IDA Pro.

**Step 1.1 – export basic-block graphs.** Disassemble every binary with IDA Pro and export the control-flow graph as
`.gdl`, one file per binary. *(There is no script for this step in the repo; IDA is commercial.)*

**Step 1.2 – normalize.** `data_prep/format_assembly.py` turns the `.gdl` files into text with **one basic block per
line**, instructions joined by ` <E> `, applying the normalization rules of the paper *as implemented in the original
notebooks*: numbers → `<VALUE>` / `<HEX>`; function names → `<FUNC>` (only with `--func-lists`, or `--call-to-func on`);
IDA dummy names (`off_`, `seg_`, `loc_`, `sub_`, ...) → `<OFF>`, `<SEG>`, `<LOC>`, `<SUB>`, ... **for `mips32` and
`ppc32` only**. The `x86`, `i386`, `arm32` and `arm64` profiles reproduce their notebooks, in which generic `<TAG>` rules fire
first, so most dummy names become `<TAG>` (e.g. `jz short loc_401040` → `jz short <TAG><VALUE>`) and rule R1 of the paper is
not reproduced for them:

```bash
python data_prep/format_assembly.py --isa arm32 --input gdl/arm32 --output formatted/arm32
python data_prep/format_assembly.py --selftest          # quick check that all six ISA profiles run
```

Supported `--isa`: `x86`, `i386`, `arm32`, `arm64`, `mips32`, `ppc32`. Useful options: `--func-lists DIR` (function-name
lists for the `<FUNC>` rule), `--cut` (split very long blocks), `--unformatted`, `--jobs N`, `--limit N`;
`--extract-vocab` / `--extract-func-names` build the opcode/register/function lists some ISAs need (`mips32` needs
`--opcodes` and `--registers`). The script
reproduces the behaviour of the original notebooks in `data_prep/legacy_notebooks/` (verified on synthetic
`.gdl` input for all six ISAs); it has not been compared against the paper's raw IDA exports, which are not in the repo.

**Step 1.3 – split.** Put the text into one folder per ISA pair with these names (`l1 < l2` alphabetically; the
parallel files must be line-aligned and non-empty):

```
data/raw/arm32-x86/
    train.arm32   valid.arm32   test.arm32          # monolingual, one basic block per line
    train.x86     valid.x86     test.x86
    valid.arm32-x86.arm32   valid.arm32-x86.x86     # parallel blocks of the same source (valid / test only)
    test.arm32-x86.arm32    test.arm32-x86.x86
```

The monolingual `valid`/`test` files are required for every language; the parallel `valid`/`test` pair is required
because back-translation is evaluated on it. Training itself needs **no** parallel data.

**Step 1.4 – BPE.** One *joint* BPE model per ISA pair, so both ISAs share a vocabulary. The number of merge operations
used in the paper (Table 1) is

| ISA pair | merges | | ISA pair | merges |
|---|---:|---|---|---:|
| i386 ⟷ X86-64 | 10,000 | | PPC32 ⟷ X86-64 | 17,000 |
| ARM32 ⟷ X86-64 | 22,000 | | M68K ⟷ X86-64 | 9,000 |
| ARM64 ⟷ X86-64 | 9,000 | | MIPS32 ⟷ X86-64 | 14,000 |

```bash
data_prep/bpe.sh /path/to/fastBPE/fast 22000 arm32 x86 data/raw/arm32-x86 data/processed/arm32-x86
```

This writes the BPE'd files, `codes` and `vocab` (`<token> <count>` per line) into `data/processed/arm32-x86`.

**Step 1.5 – binarize.**

```bash
scripts/binarize.sh data/processed/arm32-x86           # runs preprocess.py on every text file -> <file>.pth
# single file:  python preprocess.py data/processed/arm32-x86/vocab data/processed/arm32-x86/train.arm32
```

`data/processed/arm32-x86` is what you pass to `--data_path`. An existing `.pth` is reused; delete it to re-binarize.
The special tokens `<s>`, `</s>`, `<pad>`, `<unk>`, `<special0..9>` are added in front of the vocabulary.

## 2. Train the translator

`train.py` is a single entry point. [`scripts/train_arm32_x86.sh`](scripts/train_arm32_x86.sh) is the recipe of our recorded
arm32→x86 run. **It trains without the flow adapter (`--use_latent_code false`)**, i.e. it corresponds to the "None" row of
the paper's flow ablation (Table 10), not to the 3-`scf` flow-adapter model behind Tables 2 and 3. How to switch the flow
adapter on is described [below](#2-train-the-translator) ("Flow adapter and latent code"):

```bash
bash scripts/train_arm32_x86.sh          # DATA=data/processed/arm32-x86 DUMP=dumped (override via env vars)
```

which is equivalent to

```bash
python train.py --exp_name arm32-x86 --dump_path dumped \
    --data_path data/processed/arm32-x86 --lgs arm32-x86 \
    --encoder_only false --separate_decoders true --use_latent_code false \
    --emb_dim 64 --n_layers 4 --n_heads 4 --dropout 0.1 --attention_dropout 0.1 \
    --ae_steps arm32,x86 --clm_steps arm32,x86 --mlm_steps arm32,x86 \
    --bt_steps arm32-x86-arm32,x86-arm32-x86 \
    --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 \
    --two_stage true --bt_begin_epoch 2000 --eval_periodic 20 --eval_every 0 \
    --epoch_size 4000 --accumulate_gradients 8 --max_epoch 10000 \
    --batch_size 300 --tokens_per_batch 2000 --max_len 200 \
    --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0003" \
    --save_periodic 200 --debug_slurm true
```

**What it does.** One shared Transformer encoder (with an ISA embedding) and one decoder per ISA
(`--separate_decoders true`; the first language of `--lgs` owns `src_decoder`, the second `tgt_decoder`).

* *Stage 1, epochs `< --bt_begin_epoch`*: causal LM (CLM), masked LM (MLM) and denoising auto-encoding (AE, with word
  shuffle / dropout / blank noise) on both ISAs.
* *Stage 2, epochs `>= --bt_begin_epoch`*: back-translation (BT) only, in both directions
  (`arm32→x86→arm32`, `x86→arm32→x86`). This is `--two_stage true`; without it the stage-1 steps keep running and BT is
  added on top (the default behaviour).
* `--eval_periodic N`: in every iteration of every N-th epoch (epoch 0 included) one **extra pass over `--bt_steps`** is
  run, so these epochs contain BT steps even in stage 1. Despite its historical name this is *training*, not evaluation
  (it updates the weights); the number of steps per epoch does not change. `0` (default) disables it.
* `--eval_every N` runs the validation/test evaluation (perplexity, accuracy, and BLEU with `--eval_bleu true`) every N
  epochs (default 1); `0` (as in the recipe) skips it. A `best-<metric>.pth` is written only if you also pass
  `--validation_metrics`, e.g. `--validation_metrics _valid_arm32-x86_mt_ppl`; a leading `_` means *lower is better*
  (`valid_arm32-x86_mt_bleu` is maximised). `--stopping_criterion '<metric>,<patience>'` uses the same convention and
  needs `--eval_every 1`. The CLM/MLM evaluations stream the whole valid/test sets, so evaluating every epoch is slow
  for long epochs.
* One *epoch* is `epoch_size × accumulate_gradients` sentences counted in `--batch_size` units, not a pass over the data.
* The paper trains "until the loss drops below 0.3" (Sec. 5.2). The code has no such stopping rule: the recipe runs for
  `--max_epoch 10000` and writes `periodic-<epoch>.pth` every 200 epochs (the translation and detection examples below use
  `periodic-9400.pth` as a placeholder). Choose the checkpoint yourself, e.g. from the BT loss in `train.log`.

**Settings of the recorded runs.** The values above are those stored in the `params.pkl` of our arm32→x86 run. The main
emb-64 run of the other pairs used the same recipe with these differences: learning rate 3e-4 (arm32 and the two IR
pairs), 5e-4 (i386, mips32) or 1e-3 (arm64, ppc32, s390x); `--bt_begin_epoch` 1000 (i386, mips32), 1800 (arm64) or 2000;
`--epoch_size` 3000 (i386, mips32, i386ir–ppc32ir) or 4000; `--max_len` 200 (arm32, arm64, ppc32) or 300 (i386, mips32,
s390x, IR pairs); `--batch_size` 64–350. The parameter study used `--emb_dim 32 / 64 / 128`. There is no finished M68K
run among our recorded runs, and Table 5 of the paper lists values (batch size 256, maximum length 512, mean pooling)
that none of the recorded runs used.

**Output** goes to `dumped/<exp_name>/<exp_id>/` (`<exp_id>` is random unless `--exp_id` is given):
`params.pkl` and `train.log` (every parameter and loss), `periodic-<epoch>.pth` every `--save_periodic` epochs
(encoder, `src_decoder`, `tgt_decoder`, optimizer state, vocabulary, parameters), and `hypotheses/` with the reference
files used for BLEU. Add `--save_last_checkpoint true` to also keep `checkpoint.pth`, which is resumed automatically
when you restart with the same `--exp_id`.

**Resume / reload.** `--reload_checkpoint dumped/arm32-x86/<exp_id>/periodic-<epoch>.pth` restores weights, epoch,
iteration counter and learning-rate step (Adam moments are *not* restored; `--is_checkpoint false` loads weights only).
If the file holds no optimizer state (some older checkpoints) the learning-rate schedule restarts from its warm-up.
`--max_epoch` counts the epochs of the *current* run, not an absolute epoch: resuming at epoch 5600 with
`--max_epoch 10000` would train until epoch 15600, so pass the number of remaining epochs.

**Flow adapter and latent code.** Enable them with

```
--use_latent_code true --flow_type scf        # or glow;  --z_size 256 --hiddenflow_flow_nums 3 --hiddenflow_layers 3
```

The sentence code `z` is computed from the encoder output by mean-pooling, max-pooling and the first token
(`--use_mean_pooling`, `--use_max_pooling`, `--use_CSL`; concatenated or summed, `--concat_h`) and, if needed, projected to
`--z_size`. One flow per ISA (`src_flow`, `tgt_flow`) is trained with a density loss (`--nll_weight`); in a
back-translation step the code is mapped `src_flow → Gaussian → tgt_flow` and fed to the other decoder, where it is
injected before the output projection (`--latent_concat`, `--use_weight_network`; `--detach_type`,
`--consider_nll_in_bt` control gradients). The flow code path was exercised end to end on small synthetic runs (train, evaluate, resume; translation with
latent codes was tested on randomly initialised models), but the values of `z_size`, pooling and loss weights used for the paper's flow experiments are not recorded in this
repository; the defaults above are the code defaults, not a verified reproduction of the paper's best configuration.
The sentence code of a latent-code model is pooled over the zero-padded batch, so its translations depend slightly on the
batch composition (`--batch_size`, sorting); checkpoints were trained that way and this is kept for compatibility.

**Other notes**

* `--encoder_only` defaults to `true` (a plain language model, as in XLM). Translation needs `--encoder_only false`.
* The *order* of `--lgs` matters (`arm32-x86` ⇒ `src_decoder` generates arm32, `tgt_decoder` generates x86); `translate.py`
  reads it from the checkpoint.
* `--seed` seeds python, numpy (which also drives the batch order) and torch; bitwise-identical runs on a GPU are not guaranteed.
* `python train.py --help` lists every option, including the XLM options (memory, adaptive softmax, multi-GPU) that the
  paper does not use.
* Plain language-model pre-training (as used for the IR baselines) is the same script with
  `--encoder_only true --clm_steps X --mlm_steps X --lgs X`.

## 3. Translate

```bash
python translate.py --model_path dumped/arm32-x86/<exp_id>/periodic-9400.pth --info        # what is in the checkpoint?

# a folder of BPE-encoded files (one file per binary) -> same file names in the output folder
python translate.py --model_path dumped/arm32-x86/<exp_id>/periodic-9400.pth \
    --src_lang arm32 --tgt_lang x86 \
    --src_folder data/detection/arm32/benign --output_folder output/arm32_translation/arm32_benign \
    --batch_size 200 --max_len 200

# a single file, or stdin -> stdout
python translate.py --model_path ... --src_lang arm32 --tgt_lang x86 --src_path in.bpe --output_path out.bpe
cat in.bpe | python translate.py --model_path ... --src_lang arm32 --tgt_lang x86 > out.bpe
```

* Input must be BPE-segmented with the **same codes and vocabulary as training**, one basic block per line. For the
  detection corpora (one file per binary, produced by `format_assembly.py`) apply the `codes` of the pair's training data to
  every file, e.g. `for f in formatted/arm32_benign/*; do /path/to/fastBPE/fast applybpe data/detection/arm32/benign/$(basename "$f") "$f" data/processed/arm32-x86/codes; done`.
  The X86-64 detector files must be segmented with the same `codes`, because they are embedded with the same checkpoint.
* The right decoder is chosen from `--tgt_lang` and the `--lgs` order stored in the checkpoint; both directions work.
* Lines longer than `--max_len` are dropped by default (`--long_lines drop`); use `--long_lines empty` to keep the output
  line-aligned with the input (needed for BLEU) or `truncate`. Source files are never deleted; existing outputs are skipped
  (`--overwrite` to replace them). Decoding is greedy (`--beam_size 1`) by default, as in the training-time evaluation and
  the original translation scripts (the paper does not state the decoding strategy); batches are length-sorted for speed.
* Output: one line per translated basic block, X86-64 tokens in the model vocabulary with `<E>` between instructions.
* Checkpoints are Python pickles: only load checkpoints you trust.

## 4. Evaluate the translation (BLEU)

The paper compiles the same program for the source ISA and for X86-64, translates the source binary and reports the
BLEU between the translated binary and the X86-64 binary, averaged over binaries (SacreBLEU, tokenization `none`,
add-one smoothing). `evaluate_bleu.py` implements this protocol for parallel test files that have the same names and are
line-aligned (translate with `--long_lines empty`): one corpus-level SacreBLEU score per file (the lines of a file are its
basic blocks), averaged over files, on the BPE-segmented text and on the paper's 0–1 scale. The paper does not say how the
blocks of the two binaries are aligned or whether BLEU is corpus- or sentence-level, and the original BLEU notebook is not
comparable (see below), so scores from this script are not guaranteed to reproduce Table 2:

```bash
python evaluate_bleu.py --hyp_folder output/arm32_translation/arm32_benign --ref_folder data/parallel_test/x86
python evaluate_bleu.py --hyp_path out.bpe --ref_path ref.bpe
```

Files whose line counts differ are reported and skipped (`--mismatch truncate|error` changes this; `--per_file` prints
each file). The script replaces `BLEU_Evaluation.ipynb`, which fed strings instead of token lists to its n-gram code and
therefore computed a character-overlap score (see the git history).

## 5. Malware detection

The detector is a small LSTM (embedding lookup → LSTM → max-pool over time → linear, `BCEWithLogitsLoss`, Adam). It has
no vocabulary of its own: tokens are mapped to the **frozen token embeddings of the translation checkpoint** (`dico_word2id`
and `encoder.embeddings.weight`). Consequently a detector is tied to the translation checkpoint it was trained with — train
and test with the same `--translation-ckpt`.

```bash
# 1. train on X86-64 benign / malware token files (BPE-segmented, one file per binary)
python malware_detection/train_lstm.py \
    --benign-dir data/detection/x86/benign --malware-dir data/detection/x86/malware \
    --translation-ckpt dumped/arm32-x86/<exp_id>/periodic-9400.pth \
    --out-dir detector/arm32 --hidden 16 --layers 2 --batch-size 24 --epochs 4 --test-size 0.4

# 2. score the TRANSLATED source-ISA files with the X86-64 detector -> AUC, ROC curve, per-file predictions
python malware_detection/test_lstm.py \
    --benign-dir output/arm32_translation/arm32_benign --malware-dir output/arm32_translation/arm32_malware \
    --translation-ckpt dumped/arm32-x86/<exp_id>/periodic-9400.pth \
    --lstm-weights detector/arm32/lstm_it3.pt --out-prefix results/arm32
```

`train_lstm.py` writes `lstm_it<epoch>.pt` after every epoch, `train_log.json` and `split.json` (the train/held-out file
lists); `test_lstm.py` writes `<prefix>_predictions.txt`, `<prefix>_roc.png` and `<prefix>_metrics.json` (AUC, number of
files, OOV ratio). Hidden size and layer count are read from the weight file. Details such as the length threshold (70th
percentile of the token counts, truncate / zero-pad, padding is not masked) follow the original notebooks; see `--help`
of both scripts.

The "Same-ISA detector" column of Table 3 is obtained by training an LSTM on the *native* files of the ISA itself
(`train_lstm.py --benign-dir <native-ISA benign> --malware-dir <native-ISA malware> --translation-ckpt <that pair's
checkpoint> ...`) and scoring its held-out split with `test_lstm.py --split-json <out-dir>/split.json` (`--split train`
scores the training part).

The hyper-parameters in the example are those of the original ISA notebooks (2 LSTM layers, 16 hidden units, 40 %
hold-out). The paper is not consistent about them: Sec. 5.4 says two layers, Table 6 lists 3 layers and batch size 36,
and Sec. 5.4 describes an 80 % / 20 % train/test split of the malware samples (`--test-size 0.2`). `--layers`,
`--batch-size` and `--test-size` let you choose. No CNN detector (reported in the appendix of the paper) is included.

## Smoke test

`tests/smoke_test.py` runs the whole pipeline (binarize → train for a few epochs with separate decoders, with the flow
adapter (`scf`, `glow`), evaluation and early stopping → resume → translate in both directions → BLEU → LSTM train/test →
`format_assembly.py --selftest`) on a tiny synthetic corpus, checks that every stage produces its outputs and that invalid
command lines are rejected. It needs a CUDA GPU and
takes about 6–7 minutes on a shared GPU (about 1 GB of GPU memory):

```bash
python tests/smoke_test.py --workdir /tmp/flowmaltrans_smoke
```

## Results reported in the paper

Numbers below are copied from the paper (X86-64 is always the target ISA; the paper's tables are authoritative). They were
obtained with the flow adapter; the checkpoints produced by `scripts/train_arm32_x86.sh` do not use it.

**Translation quality (BLEU, Table 2)**

| Source ISA | FlowMalTrans | UnsuperBinTrans |
|---|---:|---:|
| i386 | 0.41 | 0.35 |
| ARM32 | 0.40 | 0.28 |
| ARM64 | 0.37 | 0.32 |
| MIPS32 | 0.36 | 0.27 |
| PPC32 | 0.31 | 0.25 |
| M68K | 0.39 | 0.31 |

**Malware detection (LSTM, AUC, Table 3)** — X86-64 detector applied to translated code

| Source ISA | FlowMalTrans | UnsuperBinTrans | UniMap | CrossIns2Vec | IR-based | Same-ISA detector |
|---|---:|---:|---:|---:|---:|---:|
| i386 | **0.996** | 0.723 | 0.953 | – | 0.819 | 0.998 |
| ARM32 | **0.981** | 0.818 | 0.953 | 0.976 | 0.815 | 0.986 |
| ARM64 | **0.965** | 0.638 | 0.910 | – | 0.825 | 0.705 |
| MIPS32 | **0.971** | 0.725 | 0.933 | 0.964 | 0.791 | 0.974 |
| PPC32 | **0.973** | 0.722 | 0.919 | 0.967 | 0.639 | 0.849 |
| M68K | **0.931** | 0.689 | 0.893 | – | 0.476 | 0.790 |

Bold = best of the cross-ISA methods; the Same-ISA detector (trained on the ISA's own malware) is higher for i386, ARM32 and
MIPS32. The text of Sec. 5.5 quotes Same-ISA AUCs of 0.952 (MIPS32) and 0.949 (PPC32) that disagree with Table 3(b) (0.974,
0.849); the table values are used here.

Ablations of the paper (LSTM AUC, Tables 10–12): three sequential `scf` flows (`--hiddenflow_flow_nums 3`) are best for every
ISA except ARM32, where three `glow` flows are marginally higher (0.983 vs. 0.981); e.g. i386 0.996 vs. 0.854 without the
flow adapter; five sequential flows are clearly worse. Removing any normalization rule hurts strongly (for example ARM32:
0.981 with all rules, 0.521 without `<OFF>`/`<SEG>`, 0.575 without `<FUNC>`, 0.533 without `<VALUE>`/`<HEX>`); embedding size
64 is better than 32 and 128. Training one translator takes about 22–25 hours on one RTX 4090.

Basic blocks used for translator training (paper, appendix A.3): X86-64 2,789,119 · i386 2,803,557 · ARM64 7,413,083 ·
ARM32 5,812,795 · MIPS32 4,813,685 · PPC32 3,964,237 · M68K 5,463,395. Malware (VirusShare) used for the detectors:
X86-64 2,140 · i386 1,740 · ARM32 1,581 · MIPS32 1,430 · PPC32 962 · M68K 545 · ARM64 128.

## Scope, limitations and known gaps

Scope of the method (from the paper): evaluated on Linux-based malware samples only (Windows, iOS and Android are
untested); the samples are uncompressed, packing is out of scope, and robustness to code obfuscation is not evaluated
(the paper notes that VirusShare samples probably contain obfuscated code without knowing which obfuscations). The
model is trained on the pairs {i386, ARM32, ARM64, MIPS32, PPC32, M68K} ⟷ X86-64 but only the direction {ISA} → X86-64 is
evaluated; it is not any-to-any, and using ARM32 as a *target* is future work.

Implementation notes (how the released code realizes the method):

* One **shared Transformer encoder** (with an ISA embedding) and **one decoder per ISA** (`--separate_decoders true`). The
  sentence code of Eq. (4) of the paper (max-pool + mean-pool + first-token state, then a linear map) is
  `--concat_h false` with all three poolings on; the code default *concatenates* them (`--concat_h true`). The gate of
  Eq. (5) is `--latent_concat false --use_weight_network true`; the default is `--latent_concat true`. Which of these
  variants produced the paper's flow results is not recorded in this repository.

Things this repository does **not** provide:

* The binary → `.gdl` export (needs IDA Pro) and the compilation of the benign corpora.
* A tool that aligns basic blocks between two ISAs to build the parallel valid/test sets, and the train/valid/test
  split used in the paper. `data_prep/legacy_notebooks/find_parallel_blocks.ipynb` only pairs files of the same program.
* Block-level deduplication of the training corpora (the paper, Sec. 5.2, normalizes and deduplicates the basic blocks);
  only the file-level `data_prep/legacy_notebooks/duplicate_file_detection.ipynb` is included.
* Formatting profiles for M68K and S390x (the paper uses M68K; the six profiles in `format_assembly.py` are the ones that
  have notebooks). The `x86`, `i386`, `arm32` and `arm64` profiles are those of the original notebooks: they do not emit
  every special token of the paper's final normalization (for example `<ADDR>`), and for them the dummy-name rule R1
  yields `<TAG>` instead of `<OFF>` / `<SEG>` / `<LOC>`.
* Datasets, trained checkpoints, a CNN detector and the figure of the model architecture.
* Tested multi-GPU / SLURM training. Use a single GPU.

Verified in this repository (torch 2.3.1, one GPU): all stages of the [smoke test](#smoke-test); for the recorded
configuration (separate decoders, no flow adapter) losses, optimizer updates and learning-rate schedules of the cleaned
training code are numerically identical to the code it was derived from in 17 step scenarios, and `train.py` with the
recipe flags performs the same sequence of training steps as the original notebook script; loading, resuming and
translating with checkpoints of earlier runs; the data-formatting and LSTM scripts against the notebooks they replace on
synthetic data. Not verified: bit-exact reproduction of the paper's numbers, DDP / apex / memory code paths.

## Responsible use

Following the paper's ethics statement, **no malware binaries are distributed with this work**. Only cryptographic hash
identifiers of VirusShare samples may be shared so that qualified researchers can obtain the samples themselves from
[VirusShare](https://virusshare.com). The folders `Malware_detection_code/` and `Malware_detection_files/` that some local
experiments use are git-ignored on purpose; never commit tensors, translated files or models derived from malware samples.
The paper also announces the release of the compiled benign corpora and the VirusShare hash lists; they are not part of
this repository yet. The method targets *defensive* malware detection; do not use it to build or evade-test malware.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{hu-etal-2025-flowmaltrans,
    title = "{F}low{M}al{T}rans: Unsupervised Binary Code Translation for Malware Detection Using Flow-Adapter Architecture",
    author = "Hu, Minghao and Wang, Junzhe and Zhao, Weisen and Zeng, Qiang and Luo, Lannan",
    editor = "Christodoulopoulos, Christos and Chakraborty, Tanmoy and Rose, Carolyn and Peng, Violet",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.173/",
    doi = "10.18653/v1/2025.findings-emnlp.173",
    pages = "3251--3272"
}
```

## License and acknowledgements

The original code of this repository is released under the MIT license in [LICENSE](LICENSE). The training, model and
data-loading code (`xlm/`, `train.py`, `translate.py`, `preprocess.py`) is derived from Facebook AI Research's
[XLM](https://github.com/facebookresearch/XLM) (Lample & Conneau, *Cross-lingual Language Model Pretraining*, 2019); those
files keep their original copyright headers and remain subject to XLM's license (CC BY-NC 4.0, non-commercial). The flow
building blocks in `xlm/model/flows.py` follow Andrej Karpathy's
[pytorch-normalizing-flows](https://github.com/karpathy/pytorch-normalizing-flows) (`scf` = scaling and coupling flow,
RealNVP-style), and `xlm/evaluation/multi-bleu.perl` is the Moses BLEU script. This work was supported by NSF grant
CNS-2304720 and the Commonwealth Cyber Initiative.
