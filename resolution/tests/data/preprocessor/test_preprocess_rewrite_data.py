# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-28
from unittest import TestCase, main

from resolution.common.data.preprocessor import BertMaskRewritePreprocessor


class TestBertMaskRewritePreprocessor(TestCase):
    def setUp(self):
        super().setUp()
        self.processor = BertMaskRewritePreprocessor()

    def get_tokenize_and_label(self, context, query, rewrite):
        # remove blank
        context = "".join([c for c in context if c.strip()])
        query = "".join([c for c in query if c.strip()])
        rewrite = "".join([c for c in rewrite if c.strip()])

        # tokenize and process context
        context_list = context.split("<EOS>")
        context_tokens = [self.processor._tokenizer.tokenize(string) for string in context_list]
        context_tokens = [[token.text for token in cur_turn] for cur_turn in context_tokens]
        # 在context的第一句前后加上[CLS]和[SEP]，其他句的后面加上[SEP]
        for i in range(len(context_tokens)):
            if i == 0:
                context_tokens[i] = [self.processor._start_token] + context_tokens[i] + [self.processor._end_token]
            else:
                context_tokens[i] = context_tokens[i] + [self.processor._end_token]

        # tokenize query and rewrite
        query_tokens = [token.text for token in self.processor._tokenizer.tokenize(query)]
        rewrite_tokens = [token.text for token in self.processor._tokenizer.tokenize(rewrite)]
        # get the query tokens' occur pos in the rewrite token list
        occur_pos = self.processor._find_lcs(query_tokens, rewrite_tokens)

        # To get the right label, we assume the start and end token will be
        # added in the start and end of query.
        # So the length of label is the length of query tokens +2.
        # 获取query中每个token对应的mask标签，mask=1说明后面需要填充内容
        mask_label = self.processor.get_mask_label(query_tokens, rewrite_tokens, occur_pos)
        # 获取query中span在rewrite中出现的前后位置
        start, end = self.processor.get_start_end_in_rewrite(mask_label, rewrite_tokens, occur_pos)
        # 获取每个span在context中出现的位置
        # 这个就是我们需要使用的span标签
        start_label, end_label = self.processor.get_start_end_in_context(
            mask_label, start, end, rewrite_tokens, context_tokens)

        return context_tokens, query_tokens, rewrite_tokens, mask_label, start_label, end_label

    def test_omission_in_beginning_of_query(self):
        context = "你知道华晨宇吗<EOS>唱歌的"
        query = "超棒"
        rewrite = "华晨宇超棒"

        context_tokens, query_tokens, rewrite_tokens, mask_label, start_label, end_label = self.get_tokenize_and_label(
            context, query, rewrite)

        assert len(query_tokens) == len(mask_label) - 2
        assert mask_label == [1, 0, 0, 0]
        assert start_label == [4, -1, -1, -1]
        assert end_label == [6, -1, -1, -1]

    def test_pronoun_in_query(self):
        context = "令狐冲会什么剑法<EOS>我喜欢令狐冲"
        query = "他会什么武功呀"
        rewrite = "令狐冲会什么武功呀"

        context_tokens, query_tokens, rewrite_tokens, mask_label, start_label, end_label = self.get_tokenize_and_label(
            context, query, rewrite)

        # find resolution position [CLS] and pronoun position `他`
        assert mask_label == [1, 1, 0, 0, 0, 0, 0, 0, 0]
        assert start_label == [13, 0, -1, -1, -1, -1, -1, -1, -1]
        assert end_label == [15, 0, -1, -1, -1, -1, -1, -1, -1]

    def test_inconsecutive_english_words_in_rewrite(self):
        context = "你认识river吗<EOS>我认识maryandmax"
        query = "那是谁"
        rewrite = "mary and max是谁"

        context_tokens, query_tokens, rewrite_tokens, mask_label, start_label, end_label = self.get_tokenize_and_label(
            context, query, rewrite)

        assert mask_label == [1, 1, 0, 0, 0]
        assert start_label == [10, 0, -1, -1, -1]
        assert end_label == [10, 0, -1, -1, -1]

    def test_tolerance_of_span_in_context(self):
        # we allow several words in the span not occur in the context
        context = "可以很流批<EOS>流批是神马意思"
        query = "意思就是夸你很厉害"
        rewrite = "流批的意思就是夸你很厉害"

        context_tokens, query_tokens, rewrite_tokens, mask_label, start_label, end_label = self.get_tokenize_and_label(
            context, query, rewrite)

        assert mask_label == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert start_label == [7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        assert end_label == [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


if __name__ == '__main__':
    main()