# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-11
import tempfile
from pathlib import Path
from unittest import TestCase, main

from resolution.common.data.reader import BertPointerRewriteReader

PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"


class TestBertPointerRewriteReader(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # The vocab file saved with allennlp Vocabulary
        vocab_path = "adabrain/tests/data/vocabulary"
        # 使用vocabulary中单词创建的bert词表
        bert_temp_dir = tempfile.mkdtemp(suffix="bert")
        with open(Path(vocab_path) / "tokens.txt", 'r', encoding="utf-8") as f, \
            open(Path(bert_temp_dir) / "vocab.txt", 'w', encoding="utf-8") as fp:
            fp.write("[PAD]"+"\n")
            for line in f:
                line = line.strip()
                fp.write(line)
                fp.write("\n")
        self.data_reader = BertPointerRewriteReader(model_name=bert_temp_dir,
                                                    vocab_path=vocab_path,
                                                    oov_token="@@UNKNOWN@@")

    def test_bert_pointer_rewrite_reader(self):
        train_dataset_path = FIXTURES_ROOT / "test_pointer_rewrite.txt"
        instance = list(self.data_reader.read(train_dataset_path))

        self.assertEqual(len(instance), 2, "data length error")

        # 取出一个实例
        one_instance = instance[0]
        context_field_ids = one_instance["context_ids"]
        context_tokens = context_field_ids.tokens
        # 取出对应的一个token，验证非None
        one_token = context_tokens[0]
        self.assertTrue(one_token.text_id is not None)

if __name__ == '__main__':
    main()