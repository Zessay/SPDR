# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-11
import json
import tempfile
from unittest import TestCase, main
from pathlib import Path
from numpy.testing import assert_allclose
from allennlp.common import Params
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.batch import Batch
from allennlp.models import Model, load_archive
from allennlp.commands.train import train_model_from_file

PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"


class TestBertPointerForRewrite(TestCase):
    def setUp(self) -> None:
        super().setUp()
        param_file = FIXTURES_ROOT / "pointer_rewrite" / "bert_transformer_pointer_rewrite.jsonnet"
        dataset_file = FIXTURES_ROOT / "test_pointer_rewrite.txt"
        self.param_file = param_file
        params = Params.from_file(self.param_file)

        # 构建适用于bert model的词表，和vocabulary词表保持一致
        vocab_path = params["dataset_reader"]["vocab_path"]
        # 新生成的bert词表的路径
        bert_temp_dir = tempfile.mkdtemp(suffix="bert")
        with open(Path(vocab_path) / "tokens.txt", 'r', encoding="utf-8") as f, \
            open(Path(bert_temp_dir) / "vocab.txt", 'w', encoding="utf-8") as fp:
            fp.write("[PAD]"+"\n")
            for line in f:
                line = line.strip()
                fp.write(line)
                fp.write("\n")

        # 改写config中的部分参数
        overrides_config = {
            "dataset_reader.model_name": bert_temp_dir,
            "model.model_name": params["model"]["model_name"] + "/config.json"
        }
        self.overrides_config = json.dumps(overrides_config)
        params = Params.from_file(self.param_file, params_overrides=self.overrides_config)
        # 获取reader
        reader = DatasetReader.from_params(params["dataset_reader"])
        instances = reader.read(str(dataset_file))
        # 如果存在词表的参数，则加载词表
        if "vocabulary" in params:
            vocab_params = params["vocabulary"]
            vocab = Vocabulary.from_params(
                params=vocab_params, instances=instances)
        else:
            vocab = Vocabulary.from_instances(instances)

        self.vocab = vocab
        self.instances = instances
        self.instances.index_with(vocab)
        # 加载模型
        # 将模型对应的model_name改成对应的config文件
        self.model = Model.from_params(params=params["model"], vocab=self.vocab)

        self.dataset = Batch(list(self.instances))
        self.dataset.index_instances(self.vocab)
        self.TEST_DIR = Path(tempfile.mkdtemp(prefix="allennlp_tests"))

    def test_model_can_train_save_and_load(self):
        save_dir = self.TEST_DIR / "save_and_load_test"
        archive_file = save_dir / "model.tar.gz"
        # test train and save
        model = train_model_from_file(self.param_file, save_dir, overrides=self.overrides_config)
        # test load
        loaded_model = load_archive(archive_file, cuda_device=-1).model
        state_keys = model.state_dict().keys()
        loaded_state_keys = loaded_model.state_dict().keys()
        assert state_keys == loaded_state_keys
        # make sure that the state dict (the parameters) are the same
        # for both models.
        for key in state_keys:
            assert_allclose(model.state_dict()[key].cpu().numpy(),
                            loaded_model.state_dict()[key].cpu().numpy(),
                            err_msg=key)

if __name__ == '__main__':
    main()