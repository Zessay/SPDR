# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-24
"""
统计数据集的一些信息
"""
from pathlib import Path

basename = "/home/zs261988/data/rewrite/mask"
train_file = "mask_train.txt"
valid_file = "mask_val.txt"


def print_statistic(file: str, prefix: str = "train_"):
    count, rewrite_count = 0, 0
    context_len, query_len, rewrite_len = 0.0, 0.0, 0.0
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            context, query, rewrite, *_ = line.split("\t\t")
            if query != rewrite:
                rewrite_count += 1
            context_len += len(context)
            query_len += len(query)
            rewrite_len += len(rewrite)
            count += 1
    # 计算要改写的比例
    # context平均长度
    # query平均长度
    # rewrite平均长度
    ratio = rewrite_count / count
    avg_context = context_len / count
    avg_query = query_len / count
    avg_rewrite = rewrite_len / count

    print(prefix+"count: ", count)
    print(prefix+"rewrite_ratio: ", ratio)
    print(prefix+"avg_context: ", avg_context)
    print(prefix+"avg_query: ", avg_query)
    print(prefix+"avg_rewrite: ", avg_rewrite)


def main():
    # 打印训练集的统计信息
    print_statistic(Path(basename) / train_file, prefix="train_")
    # 打印验证集的统计信息
    print_statistic(Path(basename) / valid_file, prefix="valid_")

if __name__ == '__main__':
    main()
