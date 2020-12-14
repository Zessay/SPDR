# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-06
from typing import Optional
from overrides import overrides
import torch

from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn import util

from resolution.common.modules.token_embedders import PretrainedChineseBertEmbedder


@TokenEmbedder.register("pretrained_chinese_bert_mismatched")
class PretrainedChineseBertMismatchedEmbedder(TokenEmbedder):
    """
    Use this embedder to embed wordpieces given by `PretrainedBertMismatchedIndexer`
    and to pool the resulting vectors to get word-level representations.

    Registered as a `TokenEmbedder` with name "pretrained_chinese_bert_mismatched".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedBertMismatchedIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedBertMismatchedIndexer`.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    """

    def __init__(self,
                 model_name: str,
                 max_length: int = None,
                 train_parameters: bool = True,
                 **kwargs) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = PretrainedChineseBertEmbedder(model_name, max_length=max_length,
                                                               train_parameters=train_parameters,
                                                               **kwargs)

        # Whether to return the outputs of all of the layers
        self.return_all = kwargs.get("return_all", False)

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    @overrides
    def forward(self,
                token_ids: torch.LongTensor,
                mask: torch.BoolTensor,
                offsets: torch.LongTensor,
                wordpiece_mask: torch.BoolTensor,
                type_ids: Optional[torch.LongTensor] = None,
                segment_concat_mask: Optional[torch.BoolTensor] = None,
                **kwargs) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedBertEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedBertEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        outputs = self._matched_embedder(token_ids, wordpiece_mask,
                                         type_ids=type_ids,
                                         segment_concat_mask=segment_concat_mask,
                                         **kwargs)

        if self.return_all:
            new_outputs = tuple()
            for i in range(len(outputs)):
                if i == 0:
                    output = self._get_orig_token_embeddings(
                        outputs[i], offsets)
                    new_outputs += (output, )
                elif i == 2:
                    new_hidden_states = tuple()
                    for j, hidden_state in enumerate(outputs[2]):
                        new_hidden_state = self._get_orig_token_embeddings(
                            hidden_state, offsets)
                        new_hidden_states += (new_hidden_state, )
                    new_outputs += (new_hidden_states, )
                else:
                    new_outputs += (outputs[i], )
            outputs = new_outputs
        else:
            outputs = self._get_orig_token_embeddings(outputs, offsets)

        return outputs

    def _get_orig_token_embeddings(self,
                                   embeddings: torch.Tensor,
                                   offsets: torch.LongTensor):
        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(
            embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / span_embeddings_len

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(
            orig_embeddings.shape)] = 0

        return orig_embeddings
