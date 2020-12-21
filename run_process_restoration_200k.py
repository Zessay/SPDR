# coding=utf-8
# @Author: 莫冉
# @Date: 2020-12-14
"""
我们的preprocessor函数期望的输入形式为：
context \t\t query \t\t rewrite \t\t restore_tokens (if exists, split with whitespace)

而restoration-200k数据集中，将source和target分开存在`.sr`和`.tr`文件中，
`.sr`文件中数据的存储形式为
utt_1 <\split> utt_2 <\split> utt_3 <\split> utt_4 || utt_5 | restore_tokens (split with whitespace)
`.tr`文件中则存储每一个样本对应的改写之后的结果
注意，`.sr`和`.tr`文件中所有的语句，都将字用空格分隔
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from resolution.common.data.preprocessor import BertMaskRewritePreprocessor


def preprocess_restoration_200k(basename: str, tgt_basename: str , file_type: str = "train"):
    with open(Path(basename) / f"{file_type}.sr", "r", encoding="utf-8") as sf, \
        open(Path(basename) / f"{file_type}.tr", "r", encoding="utf-8") as tf, \
        open(Path(tgt_basename) / f"{file_type}.txt", "w", encoding="utf-8") as f:
        for source, target in zip(sf, tf):
            source, target = source.strip(), target.strip()
            context, query_and_restore_string = source.split("||")
            # 将空格和<split>替换掉
            context = context.replace(" ", "")
            context = context.replace("<split>", "<EOS>")
            try:
                query, restore_string = query_and_restore_string.split("|")
            except:
                continue
            query = query.replace(" ", "")
            # 将target中的空格替换掉
            target = target.replace(" ", "")
            # 写入新的文件中
            if context and query and target:
                f.write(context + "\t\t")
                f.write(query + "\t\t")
                f.write(target + "\t\t")
                f.write(restore_string + "\n")

def fix_test_file(basename: str, file_type: str = "test"):
    with open(Path(basename) / f"{file_type}.sr", "r", encoding="utf-8") as sf, \
        open(Path(basename) / f"{file_type}_tmp.sr", "w", encoding="utf-8") as f:
        for i, line in enumerate(sf):
            line = line.strip()
            if i == 3222 and "S-N" in line:
                line = "有 没 有 清 新 一 点 的 局 <split> 可 以 丢 手 绢 嘛 <split> " \
                       "这 个 也 不 错 <split> 石 头 剪 刀 布 输 了 删 耳 光 也 不 错 " \
                       "|| 已 经 玩 过 了 | 石 头 剪 刀 布 输 了 删 耳 光"
            if i == 3223 and "|| |" in line:
                continue
            f.write(line + "\n")

    # 删除原来的文件
    os.remove(Path(basename) / f"{file_type}.sr")
    # 将临时文件名修改为新的文件名
    os.rename(src=Path(basename) / f"{file_type}_tmp.sr", dst=Path(basename) / f"{file_type}.sr")


def main():
    parser = argparse.ArgumentParser()

    # 需要的参数
    parser.add_argument(
        "--basename",
        default=None,
        type=str,
        required=True,
        help="The basename of the processed restoration-200k dataset."
    )
    parser.add_argument(
        "--tgt_basename",
        default=None,
        type=str,
        required=True,
        help="The path to save the processed files for spdr."
    )

    args = parser.parse_args()

    files = ["train", "valid", "test"]
    basename = args.basename
    tgt_basename = args.tgt_basename

    # 先对原始的test文件进行修正
    print("修正原始的test文件中的内容")
    fix_test_file(basename, file_type="test")

    preprocessor = BertMaskRewritePreprocessor()

    # 得到了需要预处理之前的数据
    for file_type in files:
        print(f"当前正在处理文件{file_type}")
        preprocess_restoration_200k(basename, tgt_basename, file_type=file_type)
        if file_type == "train":
            is_expand = False
        else:
            is_expand = False
        preprocessor.read_and_save(from_file=Path(tgt_basename) / f"{file_type}.txt",
                                   to_file=Path(tgt_basename) / f"mask_{file_type}.txt",
                                   context_split_token="<EOS>",
                                   is_split_query_word=False,
                                   is_expand=is_expand)


if __name__ == '__main__':
    main()




