# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-11
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


class TestLSTMPointerForRewrite(TestCase):
    def setUp(self) -> None:
        super().setUp()
        param_file = FIXTURES_ROOT / "pointer_rewrite" / "lstm_lstm_pointer_rewrite.jsonnet"
        dataset_file = FIXTURES_ROOT / "test_pointer_rewrite.txt"
        self.param_file = param_file
        params = Params.from_file(self.param_file)

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
        self.model = Model.from_params(params=params["model"], vocab=self.vocab)

        self.dataset = Batch(list(self.instances))
        self.dataset.index_instances(self.vocab)
        self.TEST_DIR = Path(tempfile.mkdtemp(prefix="allennlp_tests"))

    def test_model_can_train_save_and_load(self):
        save_dir = self.TEST_DIR / "save_and_load_test"
        archive_file = save_dir / "model.tar.gz"
        # test train and save
        model = train_model_from_file(self.param_file, save_dir)
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
