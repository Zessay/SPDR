# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-03
from typing import Optional
from overrides import overrides
import torch
from transformers import AutoModel, AutoModelWithLMHead
from transformers import BertModel, AlbertModel, ElectraModel
from transformers import AutoConfig

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

from resolution.common.data.tokenizer import PretrainedChineseBertTokenizer


@TokenEmbedder.register("pretrained_chinese_bert")
class PretrainedChineseBertEmbedder(PretrainedTransformerEmbedder):
    """
    The subclass of `PretrainedTransformerEmbedder` but don't use the super().__init__().
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    Registered as a `TokenEmbedder` with name "pretrained_chinese_bert".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedBertIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedBertIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    """

    def __init__(self,
                 model_name: str,
                 *,
                 max_length: int = None,
                 sub_module: str = None,
                 train_parameters: bool = True,
                 **kwargs) -> None:
        torch.nn.Module.__init__(self)
        model_name = model_name.rstrip("/")
        postfix = model_name.split("/")[-1]
        if postfix.endswith("config.json"):
            config = AutoConfig.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_config(config)
        else:
            if "albert" in postfix:
                self.transformer_model = AlbertModel.from_pretrained(model_name)
            elif "bert" in postfix:
                self.transformer_model = BertModel.from_pretrained(model_name)
            elif "xlnet" in postfix:
                self.transformer_model = AutoModelWithLMHead.from_pretrained(model_name)
            elif "electra" in postfix:
                self.transformer_model = ElectraModel.from_pretrained(model_name)
            else:
                self.transformer_model = AutoModel.from_pretrained(model_name)

        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)

        # whether to output attentions and hidden_states
        # Valid in transformers <= 2.11.0, but invalid in transformers >= 3.0.0
        self.output_attentions = kwargs.get("output_attentions", False)
        self.output_hidden_states = kwargs.get("output_hidden_states", False)

        self.transformer_model.encoder.output_attentions = self.output_attentions
        self.transformer_model.encoder.output_hidden_states = self.output_hidden_states
        self.config = self.transformer_model.config
        self._max_length = max_length

        # get the embedding module of the pretrained model
        if "albert" in model_name:
            embedding_size = self.config.embedding_size
        else:
            embedding_size = self.config.hidden_size

        self.embeddings = self.transformer_model.embeddings

        # get the max_turn_length
        self._max_turn_length = kwargs.get("max_turn_length", 3)
        self.turn_embedding = torch.nn.Embedding(self._max_turn_length, embedding_size)

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        tokenizer = PretrainedChineseBertTokenizer(model_name)
        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        # Whether to return the output of all of the layers
        self.return_all = kwargs.get("return_all", False)

    @overrides
    def forward(self,
                token_ids: torch.LongTensor,
                mask: torch.BoolTensor,
                type_ids: Optional[torch.LongTensor] = None,
                segment_concat_mask: Optional[torch.BoolTensor] = None,
                **kwargs) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, embedding_size]`.

        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError(
                        "Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        fold_long_sequences = self._max_length is not None and token_ids.size(
            1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids)

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # get token embeddings and turn embeddings
        turn_ids = kwargs.get("turn_ids", None)

        inputs_embeds = self.embeddings(token_ids)
        # add turn embedding if necessary
        if turn_ids is not None:
            turn_embeds = self.turn_embedding(turn_ids)
            inputs_embeds = inputs_embeds + turn_embeds

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        # parameters = {"inputs_embeds": inputs_embeds, "attention_mask": transformer_mask.float(),
        #               "output_attentions": self.output_attentions,
        #               "output_hidden_states": self.output_hidden_states}
        parameters = {"inputs_embeds": inputs_embeds,
                      "attention_mask": transformer_mask.float()}
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids
        # (sequence_output, pooled_output, other_layer_outputs)
        outputs = self.transformer_model(**parameters)
        embeddings = outputs[0]

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces)
            new_outputs = (embeddings, outputs[1], )
            if self.return_all:
                if len(outputs) > 2:
                    new_hidden_states = tuple()
                    for i, hidden_state in enumerate(outputs[2]):
                        new_hidden_state = self._unfold_long_sequences(
                            hidden_state, segment_concat_mask, batch_size, num_segment_concat_wordpieces)
                        new_hidden_states += (new_hidden_state, )
                    new_outputs += (new_hidden_states, )
                    # if exist attention, we need to add attentions
                    if self.output_attentions:
                        new_outputs += outputs[3:]
                outputs = new_outputs

        if self.return_all:
            return outputs
        else:
            return embeddings
