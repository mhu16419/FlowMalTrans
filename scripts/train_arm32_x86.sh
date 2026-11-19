#!/usr/bin/env bash
# Recipe of the recorded arm32 -> x86 run: emb_dim=64, one shared encoder + two separate decoders, WITHOUT the flow
# adapter (--use_latent_code false). In the paper's terms this is the "None" row of the flow ablation (Table 10), not
# the 3-scf flow-adapter model behind Tables 2 and 3; add the flow flags listed at the end of this file for that.
#
# The values below are those stored in the params.pkl of that run (a 4-layer / 4-head encoder and one 4-layer
# decoder per ISA, trained on BPE'd assembly).
#
# Prerequisites:
#   data/processed/arm32-x86/{train,valid,test}.{arm32,x86}.pth        monolingual data (preprocess.py)
#   data/processed/arm32-x86/{valid,test}.arm32-x86.{arm32,x86}.pth    small parallel valid/test sets
#   a single CUDA GPU (NVIDIA apex is only needed for --amp >= 0)
#
# Two-stage schedule
#   epochs 0 .. 1999     : CLM + MLM + denoising auto-encoding (AE) on both ISAs (CLM/MLM train the target decoder)
#   epochs 2000 .. 9999  : back-translation arm32->x86->arm32 and x86->arm32->x86 only
#   every 20th epoch     : one extra BT step per iteration (--eval_periodic 20), also 0 and 2000
#
# 1 epoch = 4000 * 8 (accumulate_gradients) "sentences" counted in --batch_size units, i.e. 32000 / 300 ~ 107 steps.
set -euo pipefail
cd "$(dirname "$0")/.."      # run from the repository root, wherever the script is called from

DATA=${DATA:-data/processed/arm32-x86}
DUMP=${DUMP:-dumped}
SRC=arm32
TGT=x86

python train.py \
    --exp_name "${SRC}-${TGT}" \
    --dump_path "${DUMP}" \
    --data_path "${DATA}" \
    --lgs "${SRC}-${TGT}" \
    --encoder_only false \
    --separate_decoders true \
    --use_latent_code false \
    --emb_dim 64 --n_layers 4 --n_heads 4 \
    --dropout 0.1 --attention_dropout 0.1 \
    --ae_steps "${SRC},${TGT}" \
    --clm_steps "${SRC},${TGT}" \
    --mlm_steps "${SRC},${TGT}" \
    --bt_steps "${SRC}-${TGT}-${SRC},${TGT}-${SRC}-${TGT}" \
    --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 \
    --two_stage true --bt_begin_epoch 2000 \
    --eval_periodic 20 --eval_every 0 \
    --epoch_size 4000 --accumulate_gradients 8 --max_epoch 10000 \
    --batch_size 300 --tokens_per_batch 2000 --max_len 200 \
    --optimizer "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0003" \
    --save_periodic 200 \
    --debug_slurm true

# Resume (weights, epoch, iteration and LR step are restored, Adam moments are not):
#   add  --reload_checkpoint dumped/arm32-x86/<exp_id>/periodic-<epoch>.pth  (and keep the same --exp_id if you want
#   the new files in the same folder).
#
# With flows (not used by any finished run in dumped/): add
#   --use_latent_code true --flow_type scf --z_size 256 --use_weight_network false
# and a latent-code checkpoint can only be resumed with the same flags.
