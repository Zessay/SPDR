# 介绍

resolution模块主要用于完成多轮对话中的指代消解（pronoun resolution）和省略补全（omission completion）任务，基于上下文对当前轮用户的query进行补全。

## 模型

resolution模块基于allennlp 1.0实现了基于BERT的消解模型，采用和阅读理解相似的方式，将query中每个位置的token都当做要消解的对象，预测其指代对象在context中的span的范围，并填充到query中该token的前面。

> 示例：

- context：`你好<EOS>我想办理ETC`
- query：`如何办理`
- 模型补全结果：   `如何办理etc`


## resolution模块

AllenNLP定义了数据读取、数据处理、模型、预测等不同的模块，以及模块的注册机制，通过配置文件指定各注册的模块将整体串联起来。本模块文件组织方式：

```
resolution
├── common # 一些整体可公用的库
|       ├── data # 可复用的数据读取模块。
|       ├── ├── preprocessor  # 预处理工具，基于`context`, `query`, `rewrite`(补全结果) 获取训练集
|       ├── ├── reader        # 数据读取和处理
|       ├── ├── token_indexer # 对allennlp中预训练模型对应的类进行改写，适用于中文，并加入turn_id字段
|       ├── ├── tokenizer     # 分词
|       ├── metrics # 多轮对话补全模型的评价指标
|       ├── models # 可复用的模型，能被训练，建议是继承allennlp的Model，但也不强制，可以基于pytorch原生开发。该目录下，可以认为是服务领域可训练的基础模型的集合。
|       ├── modules # 可复用的模块，可以看作是模型处理的一个函数。
|       ├── predictors # model的predictor，可以扩展到model更加贴合应用的封装，即建立基础模型到应用模型的桥梁。比如一些模型基础上，添加了丰富的规则，或者多个model的组合使用，成为一个独立模块的predictor，也可以放到该目录下。在应用层面，被更广泛地理解、接受和复用。
|       ├── utils # 辅助模型的工具函数。
├── test_fixtures # 单测需要用的一些伪造数据和配置文件。
├── tests # 单测的代码，结构同common
```

## 输入数据格式

训练数据放在一个文本文件中，一行为一条数据，每个字段之间用`\t\t`分隔，训练数据字段的顺序为`context`，`query`， `rewrite`，`mask_label`，`start_label`，`end_label`。
其中`context`不同轮次之间用`<EOS>`分隔，比如`你好<EOS>我想办理ETC`（支持多个`<EOS>`）。

如果只有`context`，`query`和`rewrite`，则可以通过提供的`preprocessor`得到模型训练需要的标签，值得注意的是：**多轮对话改写要求`rewrite`中的所有token大部分都要在`context`或者`query`中出现过**。（由于一些待填入的span存在不连续的情况，所以生成标签时允许一定的`tolerance`）

由于对比实验的需要，`mask_label`，`start_label`和`end_label`的长度等于分词之后的`query`的长度加2（预处理时我们默认在`query`前面加上了`[CLS]`，后面加上了`[SEP]`）。

### 数据样例

1. adabrain/summarization/resolution/test_fixtures/test_pointer_resolution.txt

## 模型预测

```python
# span预测的模型加载方式
from resolution.common.predictors.bert_span_resolution_predictor import load_model

model_path = "/home/zs261988/models/online/bert4sr_model"

predictor = load_model(model_path=model_path,
                       predictor_name="bert_span_resolution")

# pointer-gen模型的加载方式
from resolution.common.predictors.pointer_rewrite_predictor import load_model

model_path = "/home/zs261988/models/online/lstm_lstm_pointer"
vocab_path = "/home/zs261988/models/online/lstm_lstm_pointer/vocabulary"

predictor = load_model(vocab_path=vocab_path, 
                       model_path=model_path, 
                       predictor_name="pointer_for_rewrite")


# 预测
context = "你好<EOS>我想办理ETC"
query = "如何办理"

res = predictor.predict(context, query)

print(res["rewrite_results"])

# output: 如何办理etc
```

## 实验结果

| 模型         | 测试集(EM / sEMr) |  inference一次用时 |
| :-------    | :---------:      | :---------:      |
| albert-tiny | 61.69 / 75.27    |  20-30 ms        |
| albert-base | 61.90 / 75.86    |  70-80 ms        |
| bert-wwm    | 66.65 / 82.07    |  80-90 ms        |
| roberta-wwm | 64.99 / 79.98    |  80-90 ms        |
| Pointer-Rewrite模型                                |
| 6层transformer-transformer | 64.85 / 68.45  | 150-180 ms |
| 4层LSTM-LSTM | 73.71 / 75.89    |  70-100 ms       |

对比参照的baseline模型pointer-gen网络最佳指标（73.09 / 75.67），在我们提出的衡量多轮对话补全效果的指标sEMr上有接近7个点的提升，EM指标的逊色则是任务定义本身的限制（resolution任务EM指标在验证集上的上限为82.23）。

> 数据扩充之后，分别对query采用char-level分字和word-level分词模型对应的实验结果

| 模型              | 测试集(EM / sEMr) |
| :-------         | :---------:      |
| albert-tiny-char | 67.18 / 78.92    | 
| albert-tiny-word | 67.32 / 78.84    |
| bert-wwm-char    | 72.50 / 85.36    |
| bert-wwm-word    | 72.16 / 84.90    |

以上实验结果基于少量高质量开源对话改写数据集。

> 关于sEMr指标

EM指标表示两句话精确匹配，而通常来说我们并不一定需要改写的结果和ground-truth完全匹配，而更希望其能够包含主要的关键词，可以容忍一些多余token的加入。于是我们定义：**如果改写的结果中包含了groud-truth中的所有token，并且出现的先后顺序和ground-truth中一致，则计算为一次soft-EM；同时，我们还希望在soft-EM的前提下，多余的词尽可能少，所以使用grouth-truth的长度除以改写之后句子的长度，定义为soft-EM-rate，简称sEMr**。

## 实验结果对比

> [Restoration-200k数据集](https://ai.tencent.com/ailab/nlp/dialogue/#datasets)

| 模型             | f1    | f2    |  f3   | BLEU-1|BLEU-2 |ROUGE-1|ROUGE-2|验证集用时|
| :-------:       | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:  |
| PAC (greedy)    | 61.1  | 46.9  |  37.7 | 89.5  | 85.7  | 91.2  | 82.2  |   ——   |     
| PAC (n_beam=5)  | 63.7  | 49.7  |  40.4 | 89.9  | 86.3  | 91.6  | 82.8  |   ——   |
| T-Ptr (greedy)  | 47.1  | 37.5  |  31.3 | 88.3  | 85.7  | 90.5  | 83.8  | 522 s  |
| T-Ptr (n_beam=5)| 51.0  | 40.4  |  33.3 | 90.3  | 87.4  | 90.1  | 83.0  | 602 s  |
| UniLM (greedy)  | 55.2  | 44.8  |  38.3 | 90.1  | 87.5  | 91.4  | 84.9  | 321 s  |
| UniLM (n_beam=5)| 56.8  | 46.4  |  39.8 | 90.8  | 88.3  | 91.4  | 85.0  | 467 s  |
| SARG (greedy)   | 62.4  | 52.5  |  46.3 | 92.2  | 89.6  | 92.1  | 86.0  | 50 s   |
| SARG (n_beam=5) | 62.3  | 52.5  |  46.4 | 91.4  | 88.9  | 91.9  | 85.7  | 70 s   |
| SPDR (BERT-wwm) | 68.1  | 55.3  |  48.2 | 91.0  | 88.7  | 93.6  | 87.3  | 40 s   |

其中T-Ptr即为上面的Pointer-Rewrite模型，SPDR是我们自己的模型。

> 开源数据+客服场景数据

| 模型               | f1     | f2     |  f3    | BLEU-1 |BLEU-2  |ROUGE-1 |ROUGE-2 | sEMr   |验证集用时  |
| :-------:         | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  |:---:      |
| T-Ptr (n_beam=4)  | 79.6   | 68.8   |**61.8**| 92.2   | 90.3   | 94.0   | 89.9   | 75.1   | 120-180 ms|   |     
| SPDR (BERT-wwm)   |**88.7**|**70.6**|  57.0  |**93.0**|**90.6**|**96.8**|**91.8**|**81.8**| 70-80 ms  |
| SPDR (RBT-3)      | 84.3   | 65.0   |  50.8  |  91.9  | 89.3   | 95.9   | 90.3   | 78.9   | 30-40 ms  |
| SPDR (ALBERT-tiny)| 80.4   | 59.9   |  45.1  |  90.5  | 87.7   | 95.0   | 89.0   | 76.1   | 20-30 ms  |
