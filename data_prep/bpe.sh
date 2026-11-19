#!/usr/bin/env bash
# Learn a JOINT fastBPE model for two ISAs and apply it to every text file of a data folder.
#
#   data_prep/bpe.sh <fastBPE_binary> <n_merges> <lang1> <lang2> <raw_dir> <out_dir>
#
# <raw_dir> holds the normalized text files (one basic block per line), named
#     train.<lang>  valid.<lang>  test.<lang>                       for <lang> in {lang1, lang2}
#     valid.<l1>-<l2>.<lang>  test.<l1>-<l2>.<lang>                 parallel evaluation data (l1 < l2 alphabetically)
# The script writes into <out_dir> the BPE-segmented files with the same names, plus
#     codes   the learned merge operations   (fast learnbpe, trained on train.<lang1> + train.<lang2>)
#     vocab   "<token> <count>" per line      (fast getvocab on the BPE-segmented training files)
# The merge count per ISA pair used in the paper is listed in README.md (Table 1 of the paper).
#
# fastBPE: git clone https://github.com/glample/fastBPE && cd fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
#          (the binary is fastBPE/fast; pass its path, it is not searched in PATH)
set -euo pipefail

if [ "$#" -ne 6 ]; then
    awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"
    exit 1
fi

FAST=$1; N_MERGES=$2; L1=$3; L2=$4; RAW=$5; OUT=$6

[ -x "$FAST" ] || { echo "fastBPE binary not found or not executable: $FAST" >&2; exit 1; }
[[ "$N_MERGES" =~ ^[0-9]+$ ]] || { echo "n_merges must be an integer, got: $N_MERGES" >&2; exit 1; }
[ "$L1" != "$L2" ] || { echo "lang1 and lang2 must differ" >&2; exit 1; }
[ -d "$RAW" ] || { echo "raw_dir not found: $RAW" >&2; exit 1; }
if [ -d "$OUT" ] && [ "$RAW" -ef "$OUT" ]; then
    echo "out_dir must differ from raw_dir (applybpe would overwrite the input files)" >&2; exit 1
fi
for f in "$RAW/train.$L1" "$RAW/train.$L2"; do
    [ -f "$f" ] || { echo "missing training file: $f" >&2; exit 1; }
done
mkdir -p "$OUT"

echo "Learning $N_MERGES joint BPE merges on train.$L1 + train.$L2 ..."
"$FAST" learnbpe "$N_MERGES" "$RAW/train.$L1" "$RAW/train.$L2" > "$OUT/codes"
[ -s "$OUT/codes" ] || { echo "learnbpe produced no codes (is $FAST really fastBPE?)" >&2; exit 1; }

n_files=0
for f in "$RAW"/*; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    lang=${name##*.}
    if [ "$lang" = "$L1" ] || [ "$lang" = "$L2" ]; then
        "$FAST" applybpe "$OUT/$name" "$f" "$OUT/codes"
        n_files=$((n_files + 1))
    fi
done
echo "Applied BPE to $n_files files."

echo "Building the vocabulary ..."
"$FAST" getvocab "$OUT/train.$L1" "$OUT/train.$L2" > "$OUT/vocab"
echo "Done: $OUT/codes  $OUT/vocab  ($(wc -l < "$OUT/vocab") vocabulary entries)"
