#!/bin/bash



#Usage: 
#./BPE_get_vocab.sh mips32ir 0  # For processing malware files with mips32ir ISA
#./BPE_get_vocab.sh x86 1       # For processing benign files with x86 ISA


# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ISA> <TYPE>"
    echo "ISA: select the right ISA"
    echo "TYPE: 0 for benign, 1 for malware"
    exit 1
fi

ISA="$1"  # Change this to "x86" to switch the ISA
TYPE="$2"

FASTBPE_PATH="../tools/fastBPE/fast"
#BASE_PATH="../original_dataset/ir_formatted"
CODES="codes_300_20240815"

if [ "$TYPE" -eq 1 ]; then
    echo "Processing malware files..."
    BASE_PATH="../original_dataset/ir_formatted"
    CORPUS_PATH="$BASE_PATH/${ISA}_malware"
    OUTPUT_PATH="../Malware_detection_files/tensors/BPE_300_20240815/$ISA/${ISA}_malware"
    #OUTPUT_PATH="$BASE_PATH/${ISA}_malware_cutted_bpe"
    VOCAB="$BASE_PATH/${ISA}_vocab.txt"
    #VOCAB="$BASE_PATH/vocab/${ISA}_vocab.txt"

elif [ "$TYPE" -eq 0 ]; then
    echo "Processing benign files..."
    BASE_PATH="../original_dataset/ir_formatted"
    CORPUS_PATH="$BASE_PATH/${ISA}"
    OUTPUT_PATH="../Malware_detection_files/tensors/BPE_300_20240815/$ISA/${ISA}_benign"
    #OUTPUT_PATH="$BASE_PATH/${ISA}_cutted_bpe"
    VOCAB="$BASE_PATH/${ISA}_malware_vocab.txt"
    #VOCAB="$BASE_PATH/vocab/${ISA}_malware_vocab.txt"
else
    echo "Invalid TYPE. Please use 0 for malware or 1 for benign."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"


# Process each file in CORPUS_PATH
for file in "$CORPUS_PATH"/*; do
    filename=$(basename "$file")
    $FASTBPE_PATH applybpe "$OUTPUT_PATH/$filename" "$file" "$CODES"
done
echo "Finish processing all the files, stored to $OUTPUT_PATH"

# Concatenate all processed files into a single file
cat "$OUTPUT_PATH"/* > "$OUTPUT_PATH/all_processed.txt"


# Get vocabulary from the concatenated file
$FASTBPE_PATH getvocab "$OUTPUT_PATH/all_processed.txt" > "$VOCAB"
echo "Vocab list generated in $VOCAB"

# Remove the temporary concatenated file
rm "$OUTPUT_PATH/all_processed.txt"

echo "BPE processing completed"