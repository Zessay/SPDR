# 介绍

resolution模块主要用于完成多轮对话中的指代消解（pronoun resolution）和省略补全（omission completion）任务，基于上下文对当前轮用户的query进行补全。

## 模型

resolution模块基于allennlp 1.0实现了基于BERT的消解模型，采用和阅读理解相似的方式，将query中每个位置的token都当前要消解的对象，预测其指代对象在context中的span的范围，并填充到query中该token的前面。

> 示例：

- context：`腊八粥<EOS>我想吃腊八粥`&emsp;|&emsp;query：`喝了吗`
- 模型补全结果：   `腊八粥喝了吗`

### 训练好的模型

- 基于中文Bert-wwm训练的模型
    - [BERT4sr (sr为span resolution缩写)](oss://alipay-zark/appspace/ccs/rewrite/bert4sr_model.tar)：
    该文件解压得到`bert_config.json`，`vocab.txt`和`model.tar.gz`，前两个是开源的BERT中文预训练模型中的文件，
    后者是训练好的用于多轮对话补全任务的模型，解压后得到包含这三个文件的文件夹`bert4sr`。

## resolution模块

AllenNLP定义了数据读取、数据处理、模型、预测等不同的模块，以及模块的注册机制，通过配置文件指定各注册的模块将整体串联起来。本模块参考adabrain中common部分文件组织方式：

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
|requirements.txt # 第三方依赖包。
```

## 输入数据格式

训练数据放在一个文本文件中，一行为一条数据，每个字段之间用`\t\t`分隔，训练数据字段的顺序为`context`，`query`， `rewrite`，`mask_label`，`start_label`，`end_label`。
其中`context`不同轮次之间用`<EOS>`分隔，比如`腊八粥<EOS>我想吃腊八粥`。

如果只有`context`，`query`和`rewrite`，则可以通过提供的`preprocessor`得到模型训练需要的标签，值得注意的是：**多轮对话改写要求`rewrite`中的所有token大部分都要在`context`或者`query`中出现过**。

由于对比实验的需要，`mask_label`，`start_label`和`end_label`的长度等于分词之后的`query`的长度加2（预处理时我们默认在`query`前面加上了`[CLS]`，后面加上了`[SEP]`）。

### 数据样例

1. adabrain/summarization/resolution/test_fixtures/test_pointer_resolution.txt

## 模型预测

```python
from resolution.common.predictors.bert_span_resolution_predictor import load_model

model_path = "/home/zs261988/models/online/bert4sr_model"

predictor = load_model(model_path=model_path,
                       predictor_name="bert_span_resolution")

context = "腊八粥<EOS>我想吃腊八粥"
query = "喝了吗"

res = predictor.predict(context, query)

print(res["rewrite_results"])

# output: 喝腊八粥了吗
```

## 实验结果

| 模型 | 测试集(EM / sEMr) |
| :------- | :---------:|
| albert-tiny | 61.69 / 75.27 |
| albert-base | 61.90 / 75.86 | 
| bert-wwm    | 66.65 / 82.07 | 
| roberta-wwm | 64.99 / 79.98 |


对比参照的baseline模型pointer-gen网络最佳指标（73.09 / 75.67），在我们提出的衡量多轮对话补全效果的指标sEMr上有接近7个点的提升，EM指标的逊色则是任务定义本身的限制。


