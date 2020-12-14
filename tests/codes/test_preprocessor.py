# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-18
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../"))

from tqdm.auto import tqdm
from pathlib import Path
from resolution.common.data.preprocessor import BertMaskRewritePreprocessor

basename = "/home/zs261988/data/rewrite/business/"
from_file = "alipay_val.txt"
to_file = "mask_alipay_val.txt"

def main():
    preprocessor = BertMaskRewritePreprocessor(max_len=256)
    is_split_query_word = False
    is_expand = False

    preprocessor.read_and_save(Path(basename) / from_file,
                               Path(basename) / to_file,
                               is_split_query_word=is_split_query_word,
                               is_expand=is_expand)

def preserve_valid():
    with open(Path(basename) / from_file, "r", encoding="utf-8") as f, open(Path(basename) / to_file, "w", encoding="utf-8") as to_f:
        for line in tqdm(f):
            line = line.strip()
            line_list =  line.split("\t\t")
            context, query, rewrite, *_ = line_list

            to_f.write(context+"\t\t"+query+"\t\t"+rewrite)
            to_f.write("\n")


if __name__ == '__main__':
    main()
