# coding=utf-8
# @Author: 莫冉
# @Date: 2020-08-06
"""
主要用于BertEncoder-TransformerDecoder的结构
"""
from typing import Dict, Optional, List, Any, Union, Tuple
import torch
from overrides import overrides

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.common.checks import parse_cuda_device, check_for_gpu
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.training.metrics import Metric, Average
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import add_positional_features, masked_softmax
from allennlp.modules.attention import BilinearAttention, DotProductAttention
from allennlp.nn import Activation
from allennlp.common.params import Params

from resolution.common.modules.token_embedders import PretrainedChineseBertEmbedder
from resolution.common.modules.decoders import DecoderNet
from resolution.common.metrics import RewriteEM, TokenBasedROUGE, TokenBasedBLEU, RestorationScore
from resolution.common.utils import seed_everything
from resolution.common.utils.utils_rewrite import run_beam_search, convert_indices_to_string


TupleTensor = Tuple[torch.Tensor, torch.Tensor]


@Model.register("bert_pointer_for_rewrite")
class BertPointerForRewrite(Model):
    """
    其他Pointer Rewrite模型的基类
    :param vocab: `Vocabulary`型，词表
    :param model_name: `str`型，预训练模型的路径
    :param decoder: `DecoderNet`型，解码器
    :param decoder_type: `str`型，解码器的类型，只能是`transformer`或者`lstm`
    :param decoder_num_layers: `int`型，decoder的层数
    :param share_decoder_params: `bool`型，是否共享解码器层参数，只有解码器是`transformer`时有效
    :param text_field_embedder: `TextFieldEmbedder`型，单词到索引的映射
    :param start_token: `str`型，表示起始token
    :param end_token: `str`型，表示结束token
    :param index_name: `str`型，表示训练集数据词表在vocab中的namespace的名称，通常是`tokens`
    :param beam_size: `int`型，表示束搜索时束的大小
    :param min_dec_len: `int`型，表示解码序列的最小长度
    :param max_dec_len: `int`型，表示解码序列的最大长度
    :param coverage_factor: `float`型，表示是否使用coverage机制，以及coverage_loss的系数
    :param device: `int, str, list`型，表示device的类型
    :param trainable: `bool`型，表示Bert编码器是否可训练
    :param metrics: `List[Metric]`，需要观察的metrics
    :param valid_metric_keys: `List[str]`，由于bleu和rouge中存在很多不需要观察的metrics，所以指定有效metrics的键值
    :param seed: `int`型，表示随机种子数，保证模型可复现
    :param initializer: `InitializerApplicator`型，初始化参数使用的正则表达式
    :param regularizer: `RegularizerApplicator`型，使用的正则化类型
    """
    def __init__(self,
                 vocab: Vocabulary,
                 model_name: str,
                 decoder: DecoderNet,
                 decoder_type: str = "lstm",  # `lstm` / `transformer`
                 decoder_num_layers: int = 1,
                 share_decoder_params: bool = True,  # valid for `transformer`
                 text_field_embedder: TextFieldEmbedder = None,
                 start_token: str = "[CLS]",
                 end_token: str = "[SEP]",
                 index_name: str = "bert",
                 beam_size: int = 4,
                 min_dec_len: int = 4,
                 max_dec_len: int = 30,
                 coverage_factor: float = 0.0,
                 device: Union[int, str, List[int]] = -1,
                 trainable: bool = True,   # 表示bert的参数是否可训练
                 metrics: Optional[List[Metric]] = None,
                 valid_metric_keys: List[str] = None,
                 seed: int = 42,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):
        super().__init__(vocab, regularizer)

        # ---------- 定义编码器并获取输出维度 -------------
        if model_name is None and text_field_embedder is None:
            raise ValueError(
                f"`model_name` and `text_field_embedder` can't both equal to None.")

        # 对于预训练模型来说，这里相当于encoder
        self._text_field_embedder = text_field_embedder or BasicTextFieldEmbedder({
            index_name: PretrainedChineseBertEmbedder(model_name,
                                                      train_parameters=trainable,
                                                      return_all=False,
                                                      output_hidden_states=False)})

        # 保存bert编码器的输出维度
        self.encoder_output_dim = self._text_field_embedder.get_output_dim()

        # ---------- 通用初始化过程 -------------
        self.common_init(self.encoder_output_dim, decoder, decoder_type, decoder_num_layers,
                         share_decoder_params, start_token, end_token, index_name, beam_size,
                         min_dec_len, max_dec_len, coverage_factor, device, metrics,
                         valid_metric_keys, seed, initializer)

        # ---------- 不同编码器独特的初始化过程 -------------
        # 由于编码器是bert，所以需要保存编码器的embedding部分
        # 如果是albert，还有embedding到hidden的映射部分
        bert_token_embedder = self._text_field_embedder._token_embedders[index_name]
        self.bert_type = model_name or bert_token_embedder.model_name   # 获取model的名称
        self.word_embeddings = bert_token_embedder.transformer_model.get_input_embeddings()
        if "albert" in self.bert_type:
            # 从embedding层到隐层的映射
            self.embedding_to_hidden = bert_token_embedder.transformer_model.encoder.embedding_hidden_mapping_in

        # 如果解码器是LSTM，则需要使用attention初始化LSTM的初始状态
        # 如果编码器也是LSTM，则不需要
        if self.params["decoder_type"] == "lstm":
            self.h_query = torch.nn.Parameter(torch.randn([self.encoder_output_dim]),
                                              requires_grad=True)
            self.c_query = torch.nn.Parameter(torch.randn([self.encoder_output_dim]),
                                              requires_grad=True)
            # 当编码器是transformer，解码器是LSTM时，需要计算LSTM的初始化状态
            self.init_attention = DotProductAttention()

    def common_init(self,
                    encoder_output_dim: int,
                    decoder: DecoderNet,
                    decoder_type: str,
                    decoder_num_layers: int,
                    share_decoder_params: bool,
                    start_token: str = "[CLS]",
                    end_token: str = "[SEP]",
                    index_name: str = "bert",
                    beam_size: int = 4,
                    min_dec_len: int = 4,
                    max_dec_len: int = 30,
                    coverage_factor: float = 0.0,
                    device: Union[int, str, List[int]] = -1,
                    metrics: Optional[List[Metric]] = None,
                    valid_metric_keys: List[str] = None,
                    seed: int = 42,
                    initializer: InitializerApplicator = InitializerApplicator()):
        """几个不同模型通用的初始化过程"""
        seed_everything(seed)  # 初始化随机种子
        # ----------- metrics相关初始化 -------------
        # 定义metrics
        self._metrics = [TokenBasedBLEU(), TokenBasedROUGE()]
        if metrics is not None:
            self._metrics = metrics
        self._rewrite_em = RewriteEM()
        self._restore_score = RestorationScore(compute_restore_tokens=True)
        self._cov_loss_value = Average()
        self.valid_metric_keys = valid_metric_keys

        # ----------- 参数相关初始化 -------------
        # 定义token以及其他参数
        self._start_token = start_token
        self._end_token = end_token
        self._index_name = index_name
        # 使用bert模型，本质上还是要事先读取词表
        # 所以需要将对应的vocabulary的namespace进行修改
        # 这里非常重要，如果namespace不对，很容易出现assert_trigger_error
        if "bert" in self._index_name:
            self._vocab_namespace = "tokens"
        else:
            self._vocab_namespace = self._index_name
        self.coverage_factor = coverage_factor
        self.decoder_num_layers = decoder_num_layers
        decoder_type = decoder_type.lower()
        # 保存一些重要的参数
        self.params = Params(params={"beam_size": beam_size,
                                     "min_dec_len": min_dec_len,
                                     "max_dec_len": max_dec_len,
                                     "decoder_type": decoder_type})

        # ----------- device相关初始化 -------------
        device = parse_cuda_device(device)
        check_for_gpu(device)    # 检查gpu设置是否超过可用范围
        if isinstance(device, list):
            device = device[0]
        if device < 0:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(f"cuda:{device}")

        # ----------- decoder相关初始化 -------------
        # 定义decoder
        self.decoder = decoder
        self._share_decoder_params = share_decoder_params
        # 如果解码器是lstm，需要判断是否使用coverage机制
        # transformer使用coverage机制比较麻烦，所以直接使用内部计算出来的attention分布
        if self.params['decoder_type'] == 'lstm':
            # 用于LSTM解码器
            if self.coverage_factor > 0.0:
                # 定义用于计算decoder中的每一个step对应encoder结果的attention层
                # 以及计算对于当前轮和历史轮的attention分布的权重
                self.attention = BilinearAttention(vector_dim=encoder_output_dim,
                                                   matrix_dim=encoder_output_dim + 1,
                                                   activation=Activation.by_name('linear')())
                self.lamb_linear = torch.nn.Linear(encoder_output_dim * 3 + 2, 2)
            else:
                self.attention = BilinearAttention(vector_dim=encoder_output_dim,
                                                   matrix_dim=encoder_output_dim,
                                                   activation=Activation.by_name('linear')())
                self.lamb_linear = torch.nn.Linear(encoder_output_dim * 3, 2)
        else:
            # 用于Transformer解码器
            self.lamb_linear = torch.nn.Linear(encoder_output_dim * 3, 2)

        # ----------- 词表相关初始化 -------------
        self._vocab_size = self.vocab.get_vocab_size(namespace=self._vocab_namespace)
        self._unk_id = self.vocab.get_token_index(self.vocab._oov_token,
                                                  namespace=self._vocab_namespace)
        # ----------- 初始化模型参数 -------------
        self._initializer = initializer
        self._initializer(self.lamb_linear)
        self._initializer(self.decoder)

    def _get_embeddings(self, ids: torch.Tensor):
        """对于bert，只有target会通过这个函数获取embeddings"""
        word_embed = self.word_embeddings(ids.to(torch.long))

        if "albert" in self.bert_type:
            word_embed = self.embedding_to_hidden(word_embed)
        return word_embed

    def _run_encoder(self,
                     dialogue_output: torch.Tensor,
                     context_mask: torch.Tensor,
                     query_mask: torch.Tensor):
        """
        编码阶段，主逻辑在forward中实现了，这里主要划分成context和query
        并根据编码结果，计算解码阶段需要使用的一些参数
        """
        # 获取dialogue_mask
        dialogue_mask = torch.cat([context_mask, query_mask], dim=1)
        b, max_context_len = context_mask.size()
        max_query_len = query_mask.size(1)
        context_output, query_output = torch.split(dialogue_output,
                                                   [max_context_len, max_query_len],
                                                   dim=1)
        if self.params['decoder_type'] == 'lstm':
            # compute the initial states of the decoder LSTM
            h_query_expand = self.h_query.unsqueeze(dim=0).expand(b, -1)  # [B, d_model]
            h_attn_weight = self.init_attention(h_query_expand,
                                                dialogue_output,
                                                dialogue_mask)  # [B, total_len]
            h_n = torch.bmm(h_attn_weight.unsqueeze(dim=1),
                            dialogue_output).squeeze(dim=1)  # [B, d_model]
            c_query_expand = self.c_query.unsqueeze(dim=0).expand(b, -1)  # [B, d_model]
            c_attn_weight = self.init_attention(c_query_expand,
                                                dialogue_output,
                                                dialogue_mask)  # [B, total_len]
            c_n = torch.bmm(c_attn_weight.unsqueeze(dim=1),
                            dialogue_output).squeeze(dim=1)  # [B, d_model]
            h_dec_init = h_n.unsqueeze(dim=0).expand(
                self.decoder_num_layers, -1, -1).contiguous()
            c_dec_init = c_n.unsqueeze(dim=0).expand(
                self.decoder_num_layers, -1, -1).contiguous()
            dec_init_state = (h_dec_init, c_dec_init)
        else:
            dec_init_state = None

        return context_output, query_output, dec_init_state

    def _run_lstm_decoder(self,
                          context_output: torch.Tensor,
                          query_output: torch.Tensor,
                          context_mask: torch.Tensor,
                          query_mask: torch.Tensor,
                          context_coverage: torch.Tensor,
                          query_coverage: torch.Tensor,
                          rewrite_embed: torch.Tensor,
                          rewrite_mask: Optional[torch.Tensor] = None,
                          dec_init_state: Optional[TupleTensor] = None):
        """
        实现LSTM解码器的解码过程
        Here rewrite_embed size is [B, d_model]
        :params _output: [B, _len, hidden_size]
        :params _coverage: [B, _len]
        """
        # [B, 1, d_model]
        rewrite_input = rewrite_embed.unsqueeze(dim=1)

        previous_state = {"decoder_hidden": dec_init_state[0],
                          "decoder_context": dec_init_state[1]}
        encoder_outputs = {"context_output": context_output,
                           "query_output": query_output}
        source_mask = {"context_mask": context_mask,
                       "query_mask": query_mask}

        # decoder过程，得到输出和下一步decoder的初始化状态
        dict_new_dec_init_state, dec_output = self.decoder(previous_state,
                                                           encoder_outputs,
                                                           source_mask,
                                                           rewrite_input,
                                                           rewrite_mask)
        # [num_layer, B, hidden_size]
        new_dec_init_state = (dict_new_dec_init_state['decoder_hidden'],
                              dict_new_dec_init_state['decoder_context'])
        # [B, hidden_size]
        dec_output = dec_output.squeeze(dim=1)

        # 如果coverage系数大于0，则使用coverage机制
        if self.coverage_factor > 0.0:
            # [B, _len, hidden_size + 1]
            context_coverage = context_coverage.unsqueeze(dim=-1)
            query_coverage = query_coverage.unsqueeze(dim=-1)
            context_output = torch.cat([context_output, context_coverage],
                                       dim=-1)
            query_output = torch.cat([query_output, query_coverage],
                                     dim=-1)
        # compute cur_step attention result
        # [B, _len]
        context_attn = self.attention(dec_output, context_output, context_mask)
        query_attn = self.attention(dec_output, query_output, query_mask)
        # get the context/query attn output
        # [B, hidden_size]
        dec_context = torch.bmm(context_attn.unsqueeze(dim=1),
                                context_output).squeeze(dim=1)
        dec_query = torch.bmm(query_attn.unsqueeze(dim=1),
                              query_output).squeeze(dim=1)

        # calculate lambda
        # [B, 2]
        lamb = self._compute_lambda(dec_output,
                                    dec_context=dec_context,
                                    dec_query=dec_query)
        return dec_output, context_attn, query_attn, lamb, new_dec_init_state

    def _run_transformer_decoder(self,
                                 context_output: torch.Tensor,
                                 query_output: torch.Tensor,
                                 context_mask: torch.Tensor,
                                 query_mask: torch.Tensor,
                                 rewrite_embed: torch.Tensor,
                                 rewrite_mask: torch.Tensor):
        """
        实现Transformer解码器的decoder过程
        :param _output: [B, _len, d_model]
        :param _mask: [B, _len]
        :param rewrite_embed: [B, cur_dec_len, d_model]
        :param rewrite_mask: [B, cur_dec_len]，这里只是pad的mask，上三角mask在decoder内部实现
        """
        if self._share_decoder_params:
            rewrite_embed = add_positional_features(rewrite_embed)

        previous_state = None
        encoder_outputs = {"context_output": context_output,
                           "query_output": query_output}
        source_mask = {"context_mask": context_mask,
                       "query_mask": query_mask}
        # dec_output: [B, dec_len, d_model]
        # context_attn: [B, num_heads, dec_len, context_len]
        # query_attn: [B, num_heads, dec_len, query_len]
        # x_context: [B, dec_len, d_model]
        # x_query: [B, dec_len, d_model]
        dec_output, context_attn, query_attn, x_context, x_query = self.decoder(
            previous_state, encoder_outputs, source_mask, rewrite_embed, rewrite_mask)
        # 如果共享解码器的参数
        if self._share_decoder_params:
            for _ in range(self.decoder_num_layers - 1):
                dec_output, context_attn, query_attn, x_context, x_query = self.decoder(
                    previous_state, encoder_outputs, source_mask, dec_output, rewrite_mask)

        # sum the attention dists of different heads
        context_attn = torch.sum(context_attn, dim=1, keepdim=False)
        query_attn = torch.sum(query_attn, dim=1, keepdim=False)
        # mask softmax get the final attention dists
        context_attn = masked_softmax(context_attn, context_mask, dim=-1)
        query_attn = masked_softmax(query_attn, query_mask, dim=-1)

        # compute lambda
        # [B, dec_len, 2]
        # 注意这里和LSTM解码器的区别
        # Transformer解码器是一次解码全部输出的，所以需要包含len维度
        lamb = self._compute_lambda(dec_output,
                                    dec_context=x_context,
                                    dec_query=x_query)
        return dec_output, context_attn, query_attn, lamb

    def _run_decoder(self,
                     context_output: torch.Tensor,
                     query_output: torch.Tensor,
                     context_mask: torch.Tensor,
                     query_mask: torch.Tensor,
                     prev_context_coverage: torch.Tensor,
                     prev_query_coverage: torch.Tensor,
                     rewrite_embed: torch.Tensor,
                     rewrite_mask: Optional[torch.Tensor] = None,
                     dec_init_state: Optional[TupleTensor] = None):
        """
        Run decoder phrase with LSTM or Transformer Decoder.
        :param context_output: (B, context_len, d_model), the encoder output of context.
        :param query_output: (B, query_len, d_model), the encoder output of query.
        :param context_mask: (B, context_len), the mask tensor of context.
        :param query_mask: (B, query_len), the mask tensor of query.
        :param prev_context_coverage: (B, context_len), the sum of context attention dists of previous steps.
        :param prev_query_coverage: (B, query_len), the sum of query attention dists of previous steps.
        :param rewrite_embed: (B, dec_len, rewrite_embed), if dec_len = 1, represent current step,
                              else the previous steps.
        :param rewrite_mask: (B, dec_len), the mask of the decoder input.
        :param dec_init_state: tuple, the shape of the two tensor is (B, hidden_size), the first represents
                              hidden init state, the second represents the context init state.
        :return:
        """
        if self.params['decoder_type'] == 'transformer':
            # Transformer解码器一次解码多个位置
            dec_output, context_attn, query_attn, lamb = self._run_transformer_decoder(
                context_output, query_output, context_mask, query_mask, rewrite_embed, rewrite_mask)
        elif self.params['decoder_type'] == 'lstm':
            # LSTM解码器逐个step解码
            max_rewrite_len = rewrite_embed.size(1)

            dec_outputs, context_attn_dists, query_attn_dists, lambs = [], [], [], []
            for cur_step in range(max_rewrite_len):
                (cur_dec_output, cur_context_attn, cur_query_attn,
                 cur_lamb, dec_init_state) = self._run_lstm_decoder(context_output, query_output,
                                                                    context_mask, query_mask,
                                                                    prev_context_coverage,
                                                                    prev_query_coverage,
                                                                    rewrite_embed[:, cur_step],
                                                                    rewrite_mask[:, cur_step],
                                                                    dec_init_state)
                prev_context_coverage += cur_context_attn
                prev_query_coverage += cur_query_attn

                dec_outputs.append(cur_dec_output)
                context_attn_dists.append(cur_context_attn)
                query_attn_dists.append(cur_query_attn)
                lambs.append(cur_lamb)

            # [B, dec_len, *]
            dec_output = torch.stack(dec_outputs, dim=1)
            context_attn = torch.stack(context_attn_dists, dim=1)
            query_attn = torch.stack(query_attn_dists, dim=1)
            lamb = torch.stack(lambs, dim=1)
        else:
            raise ValueError(
                f"{self.params['decoder_type']} can't be recognized, "
                f"only `lstm` and `transformer` are accepted.")

        return dec_output, context_attn, query_attn, lamb, dec_init_state

    def _run_inference(self,
                       context_output: torch.Tensor,
                       query_output: torch.Tensor,
                       context_mask: torch.Tensor,
                       query_mask: torch.Tensor,
                       extend_context_ids: torch.Tensor,
                       extend_query_ids: torch.Tensor,
                       oovs_len: torch.Tensor,
                       dec_init_state: Optional[TupleTensor] = None):
        """
        用于预测阶段
        :param _output: [B, _len, d_model]
        :param _mask: [B, _len]
        :param extend_context_ids: [B, context_len]
        :param extend_query_ids: [B, query_len]
        :param oovs_len: [B, ]
        :param dec_init_state: None或者Tuple[Tensor]，每个Tensor的维度是[n_layers, B, hidden_size]
        :return:
        """
        batch_size = context_output.size(0)
        batch_hyps = []
        # 逐个样例解码
        for i in range(batch_size):
            # expand each sample to [beam_size, *]
            cur_context_output = context_output[i].unsqueeze(dim=0).expand(
                self.params['beam_size'], -1, -1)
            cur_query_output = query_output[i].unsqueeze(dim=0).expand(
                self.params['beam_size'], -1, -1)
            cur_context_mask = context_mask[i].unsqueeze(dim=0).expand(
                self.params['beam_size'], -1)
            cur_query_mask = query_mask[i].unsqueeze(dim=0).expand(
                self.params['beam_size'], -1)
            cur_extend_context_ids = extend_context_ids[i].unsqueeze(
                dim=0).expand(self.params['beam_size'], -1).to(dtype=torch.long)
            cur_extend_query_ids = extend_query_ids[i].unsqueeze(dim=0).expand(
                self.params['beam_size'], -1).to(dtype=torch.long)
            # get h_n and c_n of current sample
            if dec_init_state is not None:
                cur_dec_init_state = (dec_init_state[0][:, i, :],
                                      dec_init_state[1][:, i, :])
            else:
                cur_dec_init_state = None
            cur_max_oovs = oovs_len[i].item()
            # return the sorted hyps
            cur_hyps = run_beam_search(self, self.vocab, self.params,
                                       context_output=cur_context_output,
                                       query_output=cur_query_output,
                                       context_mask=cur_context_mask,
                                       query_mask=cur_query_mask,
                                       extend_context_ids=cur_extend_context_ids,
                                       extend_query_ids=cur_extend_query_ids,
                                       max_oovs=cur_max_oovs,
                                       dec_in_state=cur_dec_init_state,
                                       index_name=self._vocab_namespace,
                                       start_token=self._start_token,
                                       end_token=self._end_token)
            batch_hyps.append(cur_hyps)

        return batch_hyps

    def _compute_lambda(self,
                        dec_output: torch.Tensor,
                        dec_context: torch.Tensor,
                        dec_query: torch.Tensor):
        """
        计算lambda，即context和query对应的attention分布前面的系数
        :param dec_output: [B, hidden_size]或者[B, dec_len, d_model]，前者的LSTM解码，后者是Transformer解码
        :param dec_context: [B, hidden_size]或者[B, dec_len, d_model]
        :param dec_query: [B, hidden_size]或者[B, dec_len, d_model]
        :return:
        """
        lambda_inputs = [dec_output, dec_context, dec_query]
        lambda_inp = torch.cat(lambda_inputs, dim=-1)

        # compute lambda which decides choose from context or query
        lamb = self.lamb_linear(lambda_inp)
        # [B, 2] or [B, dec_len, 2]
        lamb = torch.softmax(lamb, dim=-1)
        return lamb

    def _calc_final_dist(self,
                         context_attn: torch.Tensor,
                         query_attn: torch.Tensor,
                         extend_context_ids: torch.Tensor,
                         extend_query_ids: torch.Tensor,
                         lamb: torch.Tensor,
                         max_oovs: int):
        """
        Calculate the sum attention dists of context and query to the whole vocab.
        :param context_attn: [B, dec_len, context_len], each decode step attends to context dists.
        :param query_attn: [B, dec_len, query_len], each decode step attends to query dists.
        :param extend_context_ids: [B, context_len], extended context ids with oovs.
        :param extend_query_ids: [B, query_len], extended query ids with oovs.
        :param lamb: [B, dec_len, 2], decide each decode step to choose from context or query.
        :param max_oovs: int, the max oov numbers in current batch.
        :return:
        """
        # context_attn and query_attn need to multiply weight lamb
        # [B, dec_len, _len]
        lamb_context, lamb_query = torch.split(lamb, 1, dim=-1)
        context_attn = torch.mul(lamb_context, context_attn)
        query_attn = torch.mul(lamb_query, query_attn)

        # calculate the origin vocab size add oov vocab
        extend_vsize = self._vocab_size + max_oovs

        batch_size, cur_dec_len, _ = context_attn.size()
        shape = [cur_dec_len, batch_size, extend_vsize]
        # [dec_len, B, _len]
        context_attn = context_attn.transpose(0, 1)
        query_attn = query_attn.transpose(0, 1)

        # scatter attended to context probabilities to whole vocab
        context_indices = extend_context_ids.unsqueeze(
            dim=0).expand(cur_dec_len, -1, -1)
        # [dec_len, B, extend_vsize]
        dec_context_prob = torch.zeros(shape, device=self._device).scatter_add(
            dim=-1, index=context_indices, src=context_attn)

        # scatter attended to query probabilities to whole vocab
        query_indices = extend_query_ids.unsqueeze(
            dim=0).expand(cur_dec_len, -1, -1)
        # [dec_len, B, extend_vsize]
        dec_query_prob = torch.zeros(shape, device=self._device).scatter_add(
            dim=-1, index=query_indices, src=query_attn)

        # compute final copy probability distribution
        # [dec_len, B, extend_vsize]
        final_dists = dec_context_prob + dec_query_prob
        return final_dists

    def _calc_loss(self,
                   final_dists: torch.Tensor,
                   rewrite_target_ids: torch.Tensor,
                   rewrite_mask: torch.Tensor,
                   rewrite_len: torch.Tensor):
        """
        计算loss
        :param final_dists: [dec_len, B, extend_vsize]
        :param rewrite_target_ids: [B, dec_len]
        :param rewrite_mask: [B, dec_len]
        :param rewrite_len: [B, ]
        :return:
        """
        # [B, dec_len, extend_vsize]
        final_dists = final_dists.transpose(0, 1)
        # [B, dec_len]
        target_probs = torch.gather(final_dists, dim=-1,
                                    index=rewrite_target_ids.unsqueeze(dim=-1)).squeeze(dim=-1)

        # compute log probabilities of each decode step
        # [B, dec_len]
        loss = -torch.log(torch.clamp(target_probs, min=1e-10, max=1.0))
        loss_wo_pad = loss * rewrite_mask.to(dtype=loss.dtype)

        loss_sum = torch.sum(loss_wo_pad, dim=-1, keepdim=False)
        avg_loss = torch.mean(loss_sum / rewrite_len.to(dtype=loss.dtype))
        return avg_loss

    def _calc_coverage_loss(self,
                            attn_dists: torch.Tensor,
                            rewrite_mask: torch.Tensor,
                            rewrite_len: torch.Tensor):
        """
        计算coverage loss，只用于LSTM解码阶段
        :param attn_dists: [B, dec_len, _len]，最后一维可以是context_len，也可以是query_len
        :param rewrite_mask: [B, dec_len]
        :param rewrite_len: [B, ]
        :return:
        """
        # initial coverage is zero
        coverage = torch.zeros_like(attn_dists[:, 0],
                                    device=self._device,
                                    requires_grad=False)
        # 获取dec的维度
        attn_size = attn_dists.size(1)
        cov_losses = []

        # 计算coverage的主逻辑
        for i in range(attn_size):
            # [batch, ]
            a = attn_dists[:, i]
            cov_loss = torch.sum(torch.min(a, coverage), dim=1)
            cov_losses.append(cov_loss)
            coverage = coverage + a
        rewrite_mask = rewrite_mask.to(dtype=coverage.dtype)
        rewrite_len = rewrite_len.to(dtype=coverage.dtype)
        losses = torch.stack(cov_losses, dim=1) * rewrite_mask
        losses = torch.sum(losses, dim=1, keepdim=False) / rewrite_len
        coverage_loss = torch.mean(losses)
        return coverage_loss

    def _update_metrics(self,
                        predictions: List[str],
                        gold_targets: List[str],
                        queries: List[str]):
        """
        更新所有的metrics
        """
        for metric in self._metrics:
            metric(predictions, gold_targets)
        self._rewrite_em(predictions, gold_targets, queries)
        self._restore_score(predictions, gold_targets, queries)

    def decode_onestep(self,
                       context_output: torch.Tensor,
                       query_output: torch.Tensor,
                       context_mask: torch.Tensor,
                       query_mask: torch.Tensor,
                       extend_context_ids: torch.Tensor,
                       extend_query_ids: torch.Tensor,
                       max_oovs: int,
                       target_input_tokens: List[List[int]],
                       prev_context_coverages: List[torch.Tensor],
                       prev_query_coverages: List[torch.Tensor],
                       dec_init_states: Optional[List[TupleTensor]] = None):
        """验证集和测试集逐个step解码的过程"""
        # get the last decode output before as the new decode input
        if self.params['decoder_type'] == "transformer":
            # [B, cur_dec_len]
            target_input_ids = torch.tensor(target_input_tokens).to(self._device)
        else:
            target_input_ids = torch.tensor(
                [tokens[-1] for tokens in target_input_tokens], device=self._device)
            # shape: [B, 1]
            target_input_ids = target_input_ids.unsqueeze(dim=1)
        # get the mask of the input
        target_mask = torch.ones_like(target_input_ids).to(self._device, dtype=torch.bool)

        # covert decode tokens to embeddings
        # [B, d_model]或者[B, dec_len, d_model]
        target_embed = self._get_embeddings(target_input_ids.to(torch.long))
        # stack the pre coverages to [num_samples, _len]
        prev_context_coverage = torch.stack(prev_context_coverages,
                                            dim=0).to(device=self._device)
        prev_query_coverage = torch.stack(prev_query_coverages,
                                          dim=0).to(device=self._device)
        if dec_init_states is not None:
            dec_init_h_state, dec_init_c_state = [], []
            for dec_state in dec_init_states:
                dec_init_h_state.append(dec_state[0])
                dec_init_c_state.append(dec_state[1])
            # [decoder_num_layers, B, hidden_size]
            dec_init_state = (torch.stack(dec_init_h_state, dim=1).to(device=self._device),
                              torch.stack(dec_init_c_state, dim=1).to(device=self._device))
        else:
            dec_init_state = None

        cur_batch = target_embed.size(0)
        if cur_batch < self.params['beam_size']:
            context_output = context_output[:cur_batch]
            query_output = query_output[:cur_batch]
            context_mask = context_mask[:cur_batch]
            query_mask = query_mask[:cur_batch]
            extend_context_ids = extend_context_ids[:cur_batch]
            extend_query_ids = extend_query_ids[:cur_batch]

        # compute the current step decode output
        dec_output, context_attn, query_attn, lamb, new_dec_init_state = self._run_decoder(
            context_output, query_output, context_mask, query_mask, prev_context_coverage,
            prev_query_coverage, target_embed, rewrite_mask=target_mask, dec_init_state=dec_init_state)
        if self.params['decoder_type'] == 'transformer':
            # 如果解码器是transformer，则取最后一个step
            cur_context_attn, cur_query_attn = context_attn[:, -1:], query_attn[:, -1:]
            cur_lamb = lamb[:, -1:]
        else:
            cur_context_attn, cur_query_attn, cur_lamb = context_attn, query_attn, lamb
        # calculate current step attention dists to the whole vocab
        cur_step_prob_dist = self._calc_final_dist(cur_context_attn,
                                                   cur_query_attn,
                                                   extend_context_ids,
                                                   extend_query_ids,
                                                   cur_lamb,
                                                   max_oovs)
        # [B, extend_vsize]
        cur_step_prob_dist = cur_step_prob_dist.squeeze(dim=0)

        # get top beam_size log prob dist
        # [B, beam_size]
        topk_probs, topk_ids = torch.topk(cur_step_prob_dist,
                                          k=self.params['beam_size'],
                                          dim=-1)
        topk_log_probs = torch.log(topk_probs)

        # 保存coverage的结果
        context_coverages = prev_context_coverage + cur_context_attn.squeeze(dim=1)
        query_coverages = prev_query_coverage + cur_query_attn.squeeze(dim=1)

        output_dict = {}
        output_dict['context_attn_dists'] = context_attn
        output_dict['query_attn_dists'] = query_attn
        output_dict['context_coverages'] = context_coverages
        output_dict['query_coverages'] = query_coverages
        output_dict['new_states'] = new_dec_init_state
        output_dict['topk_ids'] = topk_ids
        output_dict['topk_log_probs'] = topk_log_probs
        output_dict['topk_probs'] = topk_probs

        return output_dict

    def get_rewrite_string(self,
                           output_dict: Dict[str, Any],
                           return_all: bool = False) -> Dict[str, Any]:
        """
        获取rewrite的结果
        :param return_all: bool型，表示返回所有解码结果，还是只返回最好的结果
        """
        # get the current sample predictions
        convert_output_dict = convert_indices_to_string(output_dict['hypothesis'],
                                                        output_dict['metadata'],
                                                        self.vocab,
                                                        end_token=self._end_token,
                                                        return_all=return_all,
                                                        index_name=self._vocab_namespace)
        # update the metrics
        if len(convert_output_dict['gold_target']) > 0:
            rewrites = convert_output_dict["rewrite_token"]
            gold_targets = convert_output_dict["gold_target"]
            queries = convert_output_dict["origin_query"]
            for r, g, q in zip(rewrites, gold_targets, queries):
                self._update_metrics(r, g, q)
        output_dict['rewrite_string'] = convert_output_dict['rewrite_string']
        return output_dict

    def _forward_step(self,
                      context_output: torch.Tensor,
                      query_output: torch.Tensor,
                      context_mask: torch.Tensor,
                      query_mask: torch.Tensor,
                      rewrite_embed: torch.Tensor,
                      rewrite_target_ids: torch.Tensor,
                      rewrite_len: torch.Tensor,
                      rewrite_mask: torch.Tensor,
                      extend_context_ids: torch.Tensor,
                      extend_query_ids: torch.Tensor,
                      oovs_len: torch.Tensor,
                      dec_init_state: Optional[TupleTensor] = None):
        """前向传播的过程"""
        output_dict = {}
        b, max_context_len = context_mask.size()
        max_query_len = query_mask.size(1)
        # initial coverage vector of each sample
        prev_context_coverage = torch.zeros([b, max_context_len],
                                            device=self._device)
        prev_query_coverage = torch.zeros([b, max_query_len],
                                          device=self._device)

        # 基于编码器的结果进行解码
        (dec_output, context_attn, query_attn,
         lamb, dec_init_state) = self._run_decoder(context_output,
                                                   query_output,
                                                   context_mask,
                                                   query_mask,
                                                   prev_context_coverage,
                                                   prev_query_coverage,
                                                   rewrite_embed,
                                                   rewrite_mask,
                                                   dec_init_state)
        # 当前batch中，最大的oov长度
        max_oovs = max(oovs_len).item()
        # [dec_len, B, extend_vsize]
        final_dists = self._calc_final_dist(context_attn,
                                            query_attn,
                                            extend_context_ids,
                                            extend_query_ids,
                                            lamb,
                                            max_oovs)
        output_dict['loss'] = self._calc_loss(final_dists,
                                              rewrite_target_ids,
                                              rewrite_mask,
                                              rewrite_len)

        # compute coverage_loss if necessary
        if self.coverage_factor > 0.0:
            context_coverage_loss = self._calc_coverage_loss(context_attn,
                                                             rewrite_mask, rewrite_len)
            query_coverage_loss = self._calc_coverage_loss(query_attn,
                                                           rewrite_mask, rewrite_len)
            coverage_loss = context_coverage_loss + query_coverage_loss
            output_dict['coverage_loss'] = coverage_loss
            output_dict['loss'] += self.coverage_factor * coverage_loss
            self._cov_loss_value(output_dict['coverage_loss'])
        return output_dict

    @overrides
    def forward(self,
                context_ids: TextFieldTensors,
                query_ids: TextFieldTensors,
                extend_context_ids: torch.Tensor,
                extend_query_ids: torch.Tensor,
                context_len: torch.Tensor,
                query_len: torch.Tensor,
                oovs_len: torch.Tensor,
                rewrite_input_ids: Optional[TextFieldTensors] = None,
                rewrite_target_ids: Optional[TextFieldTensors] = None,
                extend_rewrite_ids: Optional[torch.Tensor] = None,
                rewrite_len: Optional[torch.Tensor] = None,
                metadata: Optional[List[Dict[str, Any]]] = None):
        """
        这里的通用的id都是allennlp中默认的TextFieldTensors类型
        而extend_context_ids则是我们在数据预处理时转换好的
        context, query和rewrite等_len，主要用于获取mask向量
        """
        # 获取context和query的token_ids
        context_token_ids = context_ids[self._index_name]["token_ids"]
        query_token_ids = query_ids[self._index_name]["token_ids"]

        context_mask = context_ids[self._index_name]["mask"]
        query_mask = query_ids[self._index_name]["mask"]

        # get the extended context and query ids
        extend_context_ids = context_token_ids + extend_context_ids.to(dtype=torch.long)
        extend_query_ids = query_token_ids + extend_query_ids.to(dtype=torch.long)
        # ---------- bert编码器计算输出 ---------------
        # 需要将context和query拼接在一起编码
        indexers = context_ids.keys()
        dialogue_ids = {}
        for indexer in indexers:
            # get the various variables of context and query
            dialogue_ids[indexer] = {}
            for key in context_ids[indexer].keys():
                context = context_ids[indexer][key]
                query = query_ids[indexer][key]
                # concat the context and query in the length dim
                dialogue = torch.cat([context, query], dim=1)
                dialogue_ids[indexer][key] = dialogue

        # 计算编码
        dialogue_output = self._text_field_embedder(dialogue_ids)
        context_output, query_output, dec_init_state = self._run_encoder(dialogue_output,
                                                                         context_mask,
                                                                         query_mask)

        output_dict = {"metadata": metadata}
        if self.training:
            rewrite_input_token_ids = rewrite_input_ids[self._index_name]["token_ids"]
            rewrite_input_mask = rewrite_input_ids[self._index_name]["mask"]
            rewrite_target_ids = rewrite_target_ids[self._index_name]["token_ids"]
            rewrite_target_ids = rewrite_target_ids + extend_rewrite_ids.to(dtype=torch.long)

            # [B, rewrite_len, encoder_output_dim]
            rewrite_embed = self._get_embeddings(rewrite_input_token_ids)
            new_output_dict = self._forward_step(context_output, query_output, context_mask, query_mask,
                                                 rewrite_embed, rewrite_target_ids, rewrite_len,
                                                 rewrite_input_mask, extend_context_ids,
                                                 extend_query_ids, oovs_len, dec_init_state)
            output_dict.update(new_output_dict)
        else:
            batch_hyps = self._run_inference(context_output,
                                             query_output,
                                             context_mask,
                                             query_mask,
                                             extend_context_ids,
                                             extend_query_ids,
                                             oovs_len,
                                             dec_init_state=dec_init_state)
            # get the result of each instance
            output_dict['hypothesis'] = batch_hyps
            output_dict = self.get_rewrite_string(output_dict)
            output_dict["loss"] = torch.tensor(0)
        return output_dict

    @overrides
    def make_output_human_readable(self,
                                   output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_output_dict = {}
        new_output_dict["rewrite_results"] = output_dict["rewrite_string"]
        return new_output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if self.coverage_factor > 0.0:
            metrics['cov_loss'] = sanitize(self._cov_loss_value.get_metric(reset))
        for metric in self._metrics:
            metrics.update(metric.get_metric(reset))
        if self.valid_metric_keys is not None:
            all_keys = list(metrics.keys())
            for key in all_keys:
                if key not in self.valid_metric_keys:
                    del metrics[key]
        # 更新em相关指标
        metrics.update(self._rewrite_em.get_metric(reset))
        # 更新restoration score指标
        metrics.update(self._restore_score.get_metric(reset))
        return metrics