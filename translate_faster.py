import os
import io
import sys
import argparse
import torch

from sacrebleu import corpus_bleu
from xlm.utils import AttrDict
from xlm.utils import bool_flag, initialize_exp
from xlm.data.dictionary import Dictionary
from xlm.model.transformer import TransformerModel
import concurrent.futures

import torch
torch.autograd.set_detect_anomaly(True)

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/test_translate/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of sentences per batch")
    parser.add_argument("--max_len",type=int, default=200,help="the maximum sentence input for translation.")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    #this argument is used only when translating a single file
    #parser.add_argument("--output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    #input source language path
    #this argument is used when translating single file.
    #parser.add_argument("--src_path",type=str,default="",help="The path to the src language path")
    #this argument is used when translating multiple files
    parser.add_argument("--src_folder",type=str,default="",help="The path to the source language folder")
    #this argument is used when translating multiple files
    parser.add_argument("--output_folder",type=str,default="",help="The path to the output folder")
    #this reference folder is used for evaluation
    parser.add_argument("--reference_folder",type=str,default="",help="The path to the reference file folder")
    parser.add_argument("--num_files",type=int,default=1,help="The number of reference files used for evaluation")
    # dummy parser
    #parser.add_argument('-f')
    #parser.add_argument('-f', '--fff', default='1', help='dummy argument to fool ipython')

    return parser

def read_file_content(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        return file.read()
    
def translate_file(params, encoder, tgt_decoder, dico, filename):
    src_path = os.path.join(params.src_folder, filename)
    output_path = os.path.join(params.output_folder, filename)

    if os.path.isfile(src_path):
        if os.path.exists(output_path):
            print(f"Skipping file: {output_path} (output file already exists)")
            return

        # read sentences from a given source file
        src_sent = []
        with open(src_path, "r") as file:
            for line_number, line in enumerate(file, start=1):
                tokens = line.strip().split()
                if len(tokens) > params.max_len:
                    print(f"Ignored line {line_number} in file {filename} since its length is {len(tokens)}")
                    continue
                src_sent.append(line)
        if not src_sent:
            os.remove(src_path)
            return

        print(f"Translating file {filename} ...")
        
        with io.open(output_path, 'w', encoding='utf-8') as f:
            for i in range(0, len(src_sent), params.batch_size):
                # prepare batch
                word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                            for s in src_sent[i:i + params.batch_size]]
                lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
                batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
            
                batch[0] = params.eos_index
                for j, s in enumerate(word_ids):
                    if lengths[j] > 2:  # if sentence not empty
                        batch[1:lengths[j] - 1, j].copy_(s)
                    batch[lengths[j] - 1, j] = params.eos_index
                langs = batch.clone().fill_(params.src_id)

                # encode source batch and translate it
                encoded = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
                encoded = encoded.transpose(0, 1)
                decoded, dec_lengths = tgt_decoder.generate(encoded, lengths.cuda(), params.tgt_id, max_len=int(1.3 * lengths.max().item() + 10))

                # convert sentences to words
                for j in range(decoded.size(1)):
                    sent = decoded[:, j]
                    delimiters = (sent == params.eos_index).nonzero().view(-1)
                    assert len(delimiters) >= 1 and delimiters[0].item() == 0
                    sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

                    target = " ".join([dico[sent[k].item()] for k in range(len(sent))])
                    f.write(target + "\n")

    print(f"Finished translating {filename}")

def main(params):
    # initialize the experiment
    logger = initialize_exp(params)

    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    tgt_decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    
    encoder.load_state_dict(reloaded['encoder'])
    tgt_decoder.load_state_dict(reloaded['tgt_decoder'])

    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

    # create the output folder if it doesn't exist
    os.makedirs(params.output_folder, exist_ok=True)

    # the translation process using multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(translate_file, params, encoder, tgt_decoder, dico, filename): filename 
                   for filename in os.listdir(params.src_folder)}
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                future.result()
                print(f"Processed {filename} successfully.")
            except Exception as exc:
                print(f"File {filename} generated an exception: {exc}")     

if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # generate parser / parse parameters
    parser = get_parser()
    #params = parser.parse_args()
    params = parser.parse_args(args=[]) #the content args=[] is necessary to use in ipython notebook
    #params.model_path="dumped/ppc32ir-x86ir/20240515_32/periodic-3000.pth"
    params.model_path="dumped/ppc32ir-x86ir/20240620_64/periodic-3000.pth"

    params.dumped_path="dumped/test_translate/"
    #params.output_path="./dumped/test_translfate/ppc32ir-x86ir/translated_ppc32ir.txt"
    params.exp_name="ppc32ir-x86ir"
    params.src_lang="ppc32ir"
    params.tgt_lang="x86ir"
    
    #params.src_path="translate.x86ir"

    #params.src_folder="data/parallel_dataset_evaluation/ppc32ir-x86ir/ppc32ir_original/ppc32ir_zlib_O2/"
    #params.output_folder="data/parallel_dataset_evaluation/ppc32ir-x86ir/ppc32ir_128/"

    #params.src_folder="bpe_process/ppc32ir/ppc32ir_pb_bpe_cutted/"
    #params.output_folder="Malware_detection_files/param_study_ppc32ir/ppc32ir-x86ir-32/ppc32ir_benign"

    #params.src_folder=f"original_dataset/gdl_pb_formatted/arm32_malware_cutted"
    #params.output_folder="Malware_detection_files/tensors/ppc32ir_translation/ppc32ir_benign_20240608"
    #params.output_folder=f"Malware_detection_files/tensors/ppc32ir_translation/ppc32ir_malware_20240613"

    params.src_folder=f"original_dataset/ir_formatted/ppc32ir_malware_cutted"
    params.output_folder=f"Malware_detection_files/tensors/ppc32ir_translation/ppc32ir_20240620_64/ppc32ir_malware"



    #params.num_files=300
    params.batch_size=200
    params.max_len=200

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang

    #assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
       main(params)

    
    #This part after the main() is used for evaluation. 
   #  bleu_score=compute_bleu(params.output_folder,params.reference_folder,params.num_files)
   #  print(f"Evaluation for the bleu score is:{bleu_score}")    
