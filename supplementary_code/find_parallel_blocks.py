import json
import random
import argparse
from logging import getLogger
import logging
import os
from datetime import datetime
import glob

src_ISA="arm32"
tgt_ISA="x86"
src_path=f"parallel_blocks/{src_ISA}/"
tgt_path=f"parallel_blocks/{tgt_ISA}/"
output_path=f"parallel_blocks/{src_ISA}-{tgt_ISA}"

src_files = glob.glob(src_path+"**/*.gdl", recursive=True)
tgt_files = glob.glob(tgt_path+"**/*.gdl", recursive=True)

if not os.path.exists(output_path):
	os.makedirs(output_path)

notebook_path = os.path.abspath("find_parallel_blocks.ipynb")
notebook_dir = os.path.dirname(notebook_path)

# Set up the log file relative to the notebook directory
current_date = datetime.now().strftime("%Y%m%d")
random_num = random.randint(1000, 9999)
log_path=f"parallel_blocks/log"
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, f"find_parallel_blocks.log")

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Begin to find the parallel blocks.")

def remove_unuseful_lines(content):
     lines=content.split("\n")
     new_lines=[line.split('//')[0].rstrip() for line in lines]

     return "".join(new_lines)



def extract_blocks(content):
     blocks=[]
     current_block=[]
     start_linnum=None
     end_linnum=None
     for line in content.split("\n"):
          line=line.split(";")[0].strip()
          if line.startwith("node"):
               if current_block:
                    blocks.append((current_block,start_linnum,end_linnum))
               current_block=[]
               start_linnum=None
               end_linnum=None
          elif           


def main():
    src_list = []
    tgt_list = []
    for src_file in src_files:
        src_dir, src_name = os.path.split(src_file)
        src_suffix = src_dir.split("\\")[-1].split("-")[-1]
        #logging.info(f"The suffix is {src_suffix}")
        
        for tgt_file in tgt_files:
            tgt_dir, tgt_name = os.path.split(tgt_file)
            tgt_suffix = tgt_dir.split("\\")[-1].split("-")[-1]
            #logging.info(f"The suffix is {tgt_suffix}")
            
            if src_name == tgt_name and src_suffix == tgt_suffix:
                src_list.append(src_file)
                tgt_list.append(tgt_file)
                break
    logging.info(f"Found {len(src_list)} pairs of parallel files.")
    for item in src_list:
        logging.info(f"Source file paths: {item}")        
    logging.info(f"Target file paths: {tgt_list}")        
    for i in range(len(src_list)):
         src_file=src_list[i]
         tgt_file=tgt_list[i]

         with open(src_file,"r") as f1, open(tgt_file,"r") as f2:
              src_content=f1.read()
              tgt_content=f2.read()

         src_blocks = src_content.split("node: {")
         tgt_blocks = tgt_content.split("node: {")   
         src_out=[]
         tgt_out=[]






if __name__ == '__main__':
    main()    





