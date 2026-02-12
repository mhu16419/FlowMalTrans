#!/bin/bash

ISA="x86"
FASTBPE_PATH="../tools/fastBPE/fast"
#BASE_PATH="../original_dataset/ir_formatted"
BASE_PATH="../original_dataset/gdl_pb_unformatted"

#CORPUS_PATH="$BASE_PATH/${ISA}"
CORPUS_PATH="$BASE_PATH/${ISA}_1"

VOCAB="$BASE_PATH/${ISA}_vocab.txt"

cat "$CORPUS_PATH"/* > "$BASE_PATH/all_processed.txt"
# Get vocabulary from the concatenated file
$FASTBPE_PATH getvocab "$BASE_PATH/all_processed.txt" > "$VOCAB"
echo "Vocab list generated in $VOCAB"
rm "$BASE_PATH/all_processed.txt"

