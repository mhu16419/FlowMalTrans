#!/usr/bin/env bash
# Binarize every BPE-segmented text file of a data folder with preprocess.py.
#
#   scripts/binarize.sh <data_dir> [vocab_file]
#
# <data_dir> is the folder produced by data_prep/bpe.sh (BPE'd text files + a `vocab` file). Every file that is not
# vocab / codes / *.pth gets a <file>.pth next to it. Point training at it with  --data_path <data_dir>.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"
    exit 1
fi

DIR=${1%/}
VOCAB=${2:-$DIR/vocab}
HERE=$(cd "$(dirname "$0")/.." && pwd)

[ -f "$VOCAB" ] || { echo "vocabulary not found: $VOCAB" >&2; exit 1; }

n=0
for f in "$DIR"/*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        vocab|codes|*.pth) continue ;;
    esac
    python "$HERE/preprocess.py" "$VOCAB" "$f"
    n=$((n + 1))
done
echo "Binarized $n files in $DIR"
