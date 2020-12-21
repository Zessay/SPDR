# coding=utf-8
# @Author: 莫冉
# @Date: 2020-12-21
import os    # NOQA
import sys   # NOQA
sys.path.append(os.path.dirname(os.path.abspath(__file__)))                     # NOQA

import json
from allennlp.common.file_utils import cached_path
from resolution.common.data.reader.bert_word_span_resolution_reader import BertWordSpanResolutionReader  # NOQA
from resolution.common.models import BertSpanPointerResolution                                           # NOQA
from resolution.common.predictors.bert_span_resolution_predictor import load_model


def test_restoration_200k(model_path: str, predictor_name: str,
                          file_path: str, target_file_path: str,
                          batch_size: int = 16):
    # 加载模型
    predictor = load_model(model_path, predictor_name)

    # 读取数据
    instances = []
    predictions = []
    with open(cached_path(file_path), "r", encoding="utf-8") as f:
        for line in f:
            # 按顺序分别是context, query, rewrite，以及mask、start和end
            # 最后可能会有一个restore_string
            line_list = line.strip().split("\t\t")
            context, query = line_list[0], line_list[1]
            cur_instance = {
                "context": context,
                "query": query,
            }

            if len(line_list) > 2:
                rewrite = line_list[2]
                mask_string, start_string, end_string, *restore_strings = line_list[3:]

                cur_instance["rewrite"] = rewrite
                cur_instance["mask_string"] = mask_string
                cur_instance["start_string"] = start_string
                cur_instance["end_string"] = end_string

                if restore_strings:
                    restore_string = restore_strings[0]
                    cur_instance["restore_string"] = restore_string

            instances.append(cur_instance)

            if len(instances) == batch_size:
                # 将改写的结果保存
                result = predictor.predict_batch_json(instances)
                predictions.extend(result["rewrite_results"])
                instances = []

        # 预测并保存剩余的样本
        if len(instances) > 0:
            result = predictor.predict_batch_json(instances)
            predictions.extend(result["rewrite_results"])

    # 保存预测的结果
    with open(target_file_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")


    # 获取最终的评价指标结果
    metrics = predictor._model.get_metrics(reset=True)

    with open(os.path.dirname(target_file_path) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", default=None, type=str, required=True,
                        help="The file path of the test data.")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The trained model path (should include model.tar.gz and *config.json).")
    parser.add_argument("--target_file_path", default=None, type=str, required=True,
                        help="The path to save the predicted results of the test data.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="The batch size of test loader.")
    parser.add_argument("--predictor_name", default="bert_span_resolution", type=str,
                        help="The predictor name corresponds to the model.")

    args = parser.parse_args()

    metrics = test_restoration_200k(model_path=args.model_path,
                                    predictor_name=args.predictor_name,
                                    file_path=args.file_path,
                                    target_file_path=args.target_file_path,
                                    batch_size=args.batch_size)

    print(metrics)


if __name__ == '__main__':
    main()
