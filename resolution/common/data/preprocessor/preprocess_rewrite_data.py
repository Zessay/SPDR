# coding=utf-8
# @Author: 莫冉
# @Date: 2020-06-29
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../../../../../"))
import logging
from tqdm import tqdm
import numpy as np
from typing import List, TextIO
import jieba

from resolution.common.data.tokenizer import ChineseCharacterTokenizer


logger = logging.getLogger(__name__)


class BertMaskRewritePreprocessor(object):
    def __init__(self, start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 tolerance: int = 5,
                 max_len: int = 512):
        super().__init__()
        self._tokenizer = ChineseCharacterTokenizer(do_lowercase=True,
                                                    never_split=[start_token, end_token])
        self._start_token = start_token
        self._end_token = end_token
        # 初始化jieba，将start和end单词添加到词表中
        jieba.initialize()
        jieba.suggest_freq(start_token, tune=True)
        jieba.suggest_freq(end_token, tune=True)
        self.tolerance = tolerance  # 表示容忍rewrite span中多少个词没有在当前context span中出现
        self.max_len = max_len  # 表示context + query（包括special tokens）的最大长度

    def _find_lcs(self, query: List[str], rewrite: List[str]):
        """找到query中每个token在rewrite中出现的位置"""
        query_len, rewrite_len = len(query), len(rewrite)
        # 用来保存对应位置的匹配结果
        match = [[0 for _ in range(rewrite_len + 1)] for _ in range(query_len + 1)]
        # 用来保存回溯的位置
        track = [[None for _ in range(rewrite_len + 1)] for _ in range(query_len + 1)]
        # 保存每个query中的token在rewrite中出现的位置
        occur_pos = [-1 for _ in range(query_len)]
        for row in range(query_len):
            for col in range(rewrite_len):
                if query[row] == rewrite[col]:
                    # 字符匹配成功，该位置等于左上方的值+1
                    match[row + 1][col + 1] = match[row][col] + 1
                    track[row + 1][col + 1] = "ok"
                elif match[row + 1][col] >= match[row][col + 1]:
                    # 左值大于上值，则该位置的值为左值转移而来，标记回溯方向为左
                    match[row + 1][col + 1] = match[row + 1][col]
                    track[row + 1][col + 1] = "left"
                else:
                    # 上值大于左值，则该位置的值为上值转移而来，标记回溯方向为上
                    match[row + 1][col + 1] = match[row][col + 1]
                    track[row + 1][col + 1] = "up"
        # print("match: ", match)
        # print("track: ", track)
        # 从后向前回溯
        row, col = query_len, rewrite_len
        while match[row][col]:
            # 获取回溯位置的标记
            tag = track[row][col]
            # 如果匹配成功，记录该字符在query中对应的rewrite中的位置
            if tag == "ok":
                # 向前找匹配并且是ok的位置，而且match值不能小于当前位置
                origin_col = col + 1
                cur_col = col - 1
                for k in range(col - 1, 0, -1):
                    if match[row][k] < match[row][col]:
                        cur_col = k
                        break
                cur_col += 1
                # 向后找第一个ok的位置，保证词尽可能连续
                first_col = origin_col - 1
                for k in range(cur_col, origin_col):
                    if track[row][k] == "ok":
                        first_col = k
                        break
                last_col = origin_col - 1
                cur_len = match[row][first_col]
                if first_col != last_col:
                    # 如果当前token前面没有token了，就不用取前面的了
                    if cur_len <= 1:
                        col = last_col
                    else:
                        col = first_col
                else:
                    col = first_col
                row -= 1
                col -= 1
                occur_pos[row] = col
            # 向左边找上一个匹配的位置
            elif tag == "left":
                col -= 1
            # 向上面找上一个匹配的位置
            elif tag == "up":
                row -= 1

        return occur_pos

    def _find_last_rm(self, cur_pos: int, occur_pos: List[int]):
        """找到连续的失配字符（即-1对应的位置）中最后一个"""
        result = cur_pos
        try:
            while occur_pos[cur_pos] == -1:
                result = cur_pos
                cur_pos += 1
        except BaseException:
            pass  # 防止数组越界，即直到最后一个都是-1
        return result

    def _find_pre_rm(self, cur_pos: int, occur_pos: List[int]):
        """找到连续失配字符中最前面一个"""
        result = cur_pos
        try:
            while cur_pos >= 0 and occur_pos[cur_pos] == -1:
                result = cur_pos
                cur_pos -= 1
        except BaseException:
            pass  # 防止数组越界，一直到第一个都是-1
        return result

    def _find_seg_in_context_reverse(self, segment: List[str],
                                     context_lens: List[int],
                                     context_tokens: List[List[str]]):
        start = -1
        end = -1
        before_neg_num = np.infty  # 无穷大
        length = len(segment)
        for prev_len, cur_turn_tokens in zip(context_lens, context_tokens):
            # 查找segment在当前turn的tokens出现的位置
            cur_occur_pos = self._find_lcs(segment, cur_turn_tokens)
            # 如果有的词没有出现过
            neg_flag = 0
            for pos in cur_occur_pos:
                if pos == -1:
                    neg_flag += 1
            # 如果存在没有出现过的词
            if neg_flag:
                # 在span中有超过tolerance的词没有在context中出现过
                if neg_flag == length or neg_flag > self.tolerance:
                    continue
                # 当前span中-1的数量少于之前的span中-1的数量
                elif neg_flag < before_neg_num:
                    first = 0
                    cur_start = cur_occur_pos[first]
                    while cur_start == -1 and first < length:
                        first += 1
                        cur_start = cur_occur_pos[first]
                    last = length - 1
                    cur_end = cur_occur_pos[last]
                    while cur_end == -1 and last > first:
                        last -= 1
                        cur_end = cur_occur_pos[last]
                    if cur_start != -1 and cur_end != -1:
                        start = cur_start + prev_len
                        end = cur_end + prev_len
                    before_neg_num = neg_flag
            else:
                start = cur_occur_pos[0] + prev_len
                end = cur_occur_pos[-1] + prev_len
                return start, end

        return start, end

    def _limit_max_length(self, context: List[List[str]], query: List[str]):
        context_len = sum([len(turn) for turn in context])
        query_len = len(query) + 2  # 算上之后要加上的[CLS]和[SEP]
        total_len = context_len + query_len

        # 如果大于最大长度限制
        if total_len > self.max_len:
            sub_len = total_len - self.max_len
            while sub_len > 0:
                first_turn = context.pop(0)
                first_turn_len = len(first_turn)
                if first_turn_len > sub_len:
                    remained_turn = [self._start_token] + first_turn[-sub_len + 1:]
                    context = [remained_turn] + context
                    sub_len = 0
                else:
                    sub_len = sub_len - first_turn_len + 1
                    context[0] = [self._start_token] + context[0]
        return context

    def get_mask_label(self,
                       query: List[str],
                       rewrite: List[str],
                       occur_pos: List[int]):
        """获取query所有位置对应的mask标签值"""
        query_len, rewrite_len = len(query), len(rewrite)
        mask_label = [0 for _ in range(query_len)]
        i = 0

        while i < query_len:
            # 如果是最后一个位置，分两种情况
            if i == query_len - 1:
                if occur_pos[i] == -1 and (rewrite_len - 1 not in occur_pos):
                    mask_label[i] = 1
                elif occur_pos[i] != rewrite_len - 1:
                    mask_label[i] = 1
            else:
                # 如果当前query中的token没有在rewrite中出现过
                if occur_pos[i] == -1:
                    # 找到最后一个—1的位置
                    origin_i = i
                    i = self._find_last_rm(i, occur_pos)
                    # 如果最后一个位置是-1，则走一遍特殊情况
                    if i == query_len - 1:
                        i = i - 1
                    else:
                        latter_pos = occur_pos[i + 1]
                        # 如果最前面的-1是初始位置
                        if origin_i == 0:
                            if latter_pos != 0:
                                mask_label[i] = 1
                        else:
                            # 前一个出现的位置和后一个出现位置的差如果不等于1
                            prev_pos = occur_pos[origin_i - 1]
                            if latter_pos - prev_pos != 1:
                                mask_label[i] = 1
                else:
                    next_pos = occur_pos[i + 1]
                    # 直接走初始位置为-1的下一轮
                    if next_pos == -1:
                        pass
                    else:
                        if next_pos - occur_pos[i] != 1:
                            mask_label[i] = 1

            i += 1

        # 添加 [CLS] 对应的标签
        if occur_pos[0] != 0 and (0 not in occur_pos):
            mask_label = [1] + mask_label
        else:
            mask_label = [0] + mask_label

        # 添加[SEP]对应的标签
        mask_label += [0]

        return mask_label

    def get_start_end_in_rewrite(self,
                                 mask_label: List[int],
                                 rewrite: List[str],
                                 occur_pos: List[int]):
        total_len = len(mask_label)
        rewrite_len = len(rewrite)
        occur_pos = [-1] + occur_pos + [-1]
        # 用于存储起始和结束位置
        start = [-1 for _ in range(total_len)]
        end = [-1 for _ in range(total_len)]
        for i, label in enumerate(mask_label):
            # 如果标签为1，说明这个位置后面需要填充内容
            if label:
                # 如果i=0，说明[CLS]后面需要填充内容
                if i == 0:
                    start[i] = 0
                    end_pos = occur_pos[i + 1]
                    if end_pos == -1:
                        end[i] = occur_pos[self._find_last_rm(
                            i + 1, occur_pos) + 1]
                    else:
                        end[i] = end_pos
                # 如果是最后一个位置，即[SEP]的前一个位置
                elif i == total_len - 2:
                    if occur_pos[i] != -1:
                        start[i] = occur_pos[i] + 1
                    else:
                        prev_i = self._find_pre_rm(i, occur_pos)
                        if prev_i == 0:
                            start[i] = 0
                        else:
                            start[i] = occur_pos[prev_i - 1] + 1
                    end[i] = rewrite_len
                else:
                    # 如果occur_pos对应的位置为-1
                    if occur_pos[i] == -1:
                        prev_i = self._find_pre_rm(i, occur_pos)
                        if prev_i == 0:
                            start[i] = 0
                        else:
                            start[i] = occur_pos[prev_i - 1] + 1

                        # 如果起始位置为0，并且[CLS]的位置对应的mask为1
                        # 为了避免和[CLS]的内容重复，设为0
                        if prev_i == 0 and mask_label[0] == 1:
                            end[i] = 0
                        else:
                            end[i] = occur_pos[i + 1]
                    else:
                        start[i] = occur_pos[i] + 1
                        # 由于不可能存在occur_pos[i+1]等于-1的情况
                        # 所以可以直接赋值
                        end[i] = occur_pos[i + 1]
        return start, end

    def get_start_end_in_context(self, mask_label: List[int],
                                 start: List[int], end: List[int],
                                 rewrite: List[str], context: List[List[str]]):
        """找到在context中的start和end的位置"""
        new_start = [-1 for _ in range(len(start))]
        new_end = [-1 for _ in range(len(end))]
        # 先计算context中每个turn的长度
        # 并且reverse turn的顺序，从最近轮开始查找
        context_lens = np.cumsum([0] + [len(t) for t in context])[:-1][::-1]
        context_tokens = context[::-1]
        for i, label in enumerate(mask_label):
            if label:
                # 得到当前需要补全的序列
                r_start, r_end = start[i], end[i]
                if r_start > r_end:
                    print(f"Error occur in position {i}")
                    continue
                elif r_start == r_end:
                    if r_start != -1:
                        new_start[i] = 0
                        new_end[i] = 0
                    continue
                else:
                    segment = rewrite[r_start:r_end]
                    start_index, end_index = self._find_seg_in_context_reverse(
                        segment, context_lens, context_tokens)
                    if start_index != -1 and end_index != -1:
                        new_start[i] = start_index
                        new_end[i] = end_index

        return new_start, new_end

    def get_start_end(self, mask_label: List[int], rewrite: List[str],
                      context: List[List[str]], occur_pos: List[int]):
        rewrite_start, rewrite_end = self.get_start_end_in_rewrite(mask_label,
                                                                   rewrite,
                                                                   occur_pos)
        start, end = self.get_start_end_in_context(mask_label,
                                                   rewrite_start,
                                                   rewrite_end,
                                                   rewrite, context)
        return start, end

    def _split_query_to_words(self, query_tokens: List[str], start: List[int], end: List[int]):
        # 首先定义一个列表保存当前query的spans
        query_spans = []
        cur_query_span = ""
        new_start_label = []
        new_end_label = []
        new_mask_label = []
        # 遍历start和end的标签
        # 如果同时为-1，说明此处不需要填充
        for token, start_label, end_label in zip(query_tokens, start, end):
            # 这两种情况下，
            if (start_label == -1 and end_label == -1) or (start_label == 0 and end_label == 0):
                # 表明当前的token不需要消解，添加到已有的span的后面
                cur_query_span += token
            else:
                # 如果是第一次遇到消解的位置
                # 并且前面已经有值了，则添加对应的label
                if len(new_start_label) == 0 and cur_query_span:
                    new_start_label.append(-1)
                    new_end_label.append(-1)
                    new_mask_label.append(0)
                # 要保证非空（首字符就需要消解的时候容易出现）
                if cur_query_span:
                    query_spans.append(cur_query_span)
                # 清空当前span
                cur_query_span = token
                # 添加当前的标签
                new_start_label.append(start_label)
                new_end_label.append(end_label)
                new_mask_label.append(1)
        # 如果到最后还有没有添加的span则添加
        if cur_query_span:
            query_spans.append(cur_query_span)
        # 如果到最后标签都还是空的，则添加0标签
        if len(new_start_label) == 0:
            new_start_label.append(-1)
            new_end_label.append(-1)
            new_mask_label.append(0)
        # 保留sep这个token对应的标签，添加到最后面
        sep_start_label = start[-2]
        sep_end_label = end[-2]
        if (sep_start_label == -1 and sep_end_label == -1) or (sep_start_label == 0 and sep_end_label == 0):
            sep_mask_label = 0
        else:
            sep_mask_label = 1
        # 得到了当前query的所有span之后，需要使用jieba对每个span分词，同时考虑最终的
        query_words, word_start_label, word_end_label, word_mask_label = [], [], [], []
        for span, start_label, end_label, mask_label in zip(
                query_spans, new_start_label, new_end_label, new_mask_label):
            # 对当前的span分词
            cur_words = jieba.lcut(span)
            # 总词表中添加当前的词
            query_words += cur_words
            # start, end和mask label添加对应的标签值
            word_start_label.append(start_label)
            word_end_label.append(end_label)
            word_mask_label.append(mask_label)
            length = len(cur_words) - 1
            word_start_label += [-1] * length
            word_end_label += [-1] * length
            word_mask_label += [0] * length

        # 添加对应的sep的标签
        word_start_label.append(sep_start_label)
        word_end_label.append(sep_end_label)
        word_mask_label.append(sep_mask_label)

        assert len(word_start_label) == len(query_words) + 1
        assert len(word_end_label) == len(query_words) + 1
        assert len(word_mask_label) == len(query_words) + 1
        return query_words, word_start_label, word_end_label, word_mask_label

    def _expand_oneturn_sample(self, context_tokens: List[List[str]], start: List[int], end: List[int]):
        # 定义flag表示是否可以添加
        # True表示可以添加，False表示不可以添加
        flag = False
        # 计算start中除了-1之外的最小值
        min_start = np.infty   # 极大值
        for pos in start:
            if pos != -1 and pos != 0 and pos < min_start:
                min_start = pos   # 得到start的最小值
        # 如果只有一个turn
        if len(context_tokens) < 2:
            return flag, None, None, None
        # 计算第一个turn的长度
        first_turn = context_tokens[0]
        first_turn_len = len(first_turn) - 1
        # 如果最小的起始位置都比第一个turn要大，则直接去除第一个turn
        # 如果min_start是inf，说明并没有改写，也还是可以去除第一个turn
        if min_start > first_turn_len:
            # 去除第一个turn
            context_tokens = context_tokens[1:]
            # 将start和end中所有不是-1和0的减去first_turn_len
            for i, (start_pos, end_pos) in enumerate(zip(start, end)):
                if start_pos != -1 and start_pos != 0:
                    start[i] = start_pos - first_turn_len
                if end_pos != -1 and end_pos != 0:
                    end[i] = end_pos - first_turn_len
            # 得到去除第一个turn之后的context
            # 注意每一个turn后面有一个[SEP]
            all_context_tokens = []
            turns = len(context_tokens) - 1
            for k, turn in enumerate(context_tokens):
                turn_len = len(turn) - 1
                for i, token in enumerate(turn):
                    if i < turn_len:
                        all_context_tokens.append(token)
                    else:
                        # 如果是当前轮的最后一个token
                        # 但是不是最后一轮
                        if k < turns:
                            all_context_tokens.append("<EOS>")
            context = "".join(all_context_tokens)
            flag = True   # 表明可以添加
            return flag, context, start, end

        return flag, None, None, None

    def _write_to_file(self, to_f: TextIO, context: str, query: str, rewrite: str,
                       mask_label: List[int], start: List[int], end: List[int],
                       restore_string: str = ""):
        # 正常写入
        to_f.write(context + "\t\t" + query + "\t\t" + rewrite + "\t\t")
        # 写入mask的标签
        mask_label = [str(x) for x in mask_label]
        mask_string = ",".join(mask_label)
        to_f.write(mask_string + "\t\t")
        # 写入start标签
        start = [str(x) for x in start]
        start_string = ",".join(start)
        to_f.write(start_string + "\t\t")
        # 写入end标签
        end = [str(x) for x in end]
        end_string = ",".join(end)
        to_f.write(end_string)
        if restore_string:
            to_f.write("\t\t"+restore_string)
        to_f.write("\n")

    def read_and_save(self, from_file: str, to_file: str, context_split_token: str = "<EOS>",
                      is_split_query_word: bool = False, is_expand: bool = True):
        """
        is_split_query_word: bool型，表示对于query是按照分词的形式，还是分字的形式
        is_expand: bool型，表示是否对数据集进行扩充，所谓的扩充就是指
                   如果当前query需要扩充的信息没有出现在更前面的轮次中，则删除之前的轮次，
                   这样在训练时可以更加关注到有效的轮次
        """
        with open(from_file, 'r', encoding="utf-8") as f, open(to_file, 'w', encoding="utf-8") as to_f:
            for line in tqdm(f):
                line = line.strip()
                line_list = line.split("\t\t")  # 分割成context, query, rewrite
                # 去除空白字符
                for i in range(3):
                    line_list[i] = "".join([c for c in line_list[i] if c.strip()])
                context_list = line_list[0].split(context_split_token)
                # 获取context中的每个token
                context_tokens = [self._tokenizer.tokenize(
                    string) for string in context_list]
                context_tokens = [[token.text for token in cur_turn]
                                  for cur_turn in context_tokens]
                # 在context的第一句前后加上 [CLS]和[SEP]
                for i in range(len(context_tokens)):
                    if i == 0:
                        context_tokens[i] = [self._start_token] + context_tokens[i] + [self._end_token]
                    else:
                        context_tokens[i] = context_tokens[i] + [self._end_token]

                query, rewrite = line_list[1], line_list[2]

                # 判断是否存在restore tokens
                if len(line_list) > 3:
                    restore_string = line_list[3]
                else:
                    restore_string = ""

                # 获取query中每个token在rewrite中出现的位置
                query_tokens = [
                    token.text for token in self._tokenizer.tokenize(query)]
                rewrite_tokens = [
                    token.text for token in self._tokenizer.tokenize(rewrite)]
                occur_pos = self._find_lcs(query_tokens, rewrite_tokens)
                # 限制context+query的最大长度
                context_tokens = self._limit_max_length(
                    context_tokens, query_tokens)
                # 将限制长度之后的context转化为句子
                context_strings = []
                for i, turn in enumerate(context_tokens):
                    if i == 0:
                        # 如果是第一句话，去除前面[CLS]和后面[SEP]的标识符
                        context_strings.append("".join(turn[1:-1]))
                    else:
                        # 如果不是第一句话，只需要去除后面的[SEP]标识符
                        context_strings.append("".join(turn[:-1]))
                context = "<EOS>".join(context_strings)

                # 计算mask_label、start以及end的向量
                mask_label, start, end = None, None, None
                try:
                    mask_label = self.get_mask_label(
                        query_tokens, rewrite_tokens, occur_pos)
                    start, end = self.get_start_end(mask_label, rewrite_tokens,
                                                    context_tokens, occur_pos)
                except Exception as e:
                    print(f"发生错误 {e}")

                if mask_label and start and end:
                    if is_split_query_word:
                        query_words, word_start_label, word_end_label, word_mask_label = self._split_query_to_words(
                            query_tokens, start=start, end=end)
                        query_string = "<\s>".join(query_words)
                        # 保存新的query以及对应的标签
                        query = query_string
                        mask_label = word_mask_label
                        start = word_start_label
                        end = word_end_label
                    # 写入正常的数据
                    self._write_to_file(to_f, context, query, rewrite, mask_label, start, end,
                                        restore_string=restore_string)
                    if is_expand:
                        # 判断是否可以扩充数据
                        # 当context轮次大于一轮的时候
                        while len(context_tokens) > 1:
                            flag, context, start, end = self._expand_oneturn_sample(context_tokens, start, end)
                            # 可以扩充则添加对应的数据
                            # 并且context不为空
                            if flag and context:
                                self._write_to_file(to_f, context, query, rewrite, mask_label, start, end,
                                                    restore_string=restore_string)
                                context_tokens = context_tokens[1:]
                            else:
                                # 遇到不能扩充的情况直接跳出
                                break
                else:
                    print(f"没有结果的错误：{line}")


if __name__ == '__main__':
    # from_file = "/home/zs261988/data/rewrite/open_train.txt"
    # to_file = "/home/zs261988/data/rewrite/mask_word/origin_mask_train.txt"
    # is_split_query_word = True   # False表示按字分割，True表示按词分割
    #
    # preprocessor = BertMaskRewritePreprocessor()
    # preprocessor.read_and_save(from_file, to_file, is_split_query_word=is_split_query_word, is_expand=False)
    context = "今天天气怎么没有<EOS>天气预报说是多云"
    query = "明天呢"
    rewrite = "明天天气呢"

    preprocessor = BertMaskRewritePreprocessor()
    query_tokens = [token.text for token in preprocessor._tokenizer.tokenize(query)]
    rewrite_tokens = [token.text for token in preprocessor._tokenizer.tokenize(rewrite)]
    preprocessor._find_lcs(query_tokens, rewrite_tokens)

