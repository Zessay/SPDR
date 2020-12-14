# coding=utf-8
# @Author: 莫冉
# @Date: 2020-06-17
from typing import Union, List, Tuple, Dict, Any
import numpy as np
import torch

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.common.params import Params


TupleTensor = Tuple[torch.Tensor, torch.Tensor]


class Hypothesis(object):
    def __init__(self,
                 tokens: List[int],
                 target_input_tokens: List[int],
                 log_probs: List[float],
                 context_attn_dists: Union[List[torch.Tensor], List[np.ndarray]],
                 query_attn_dists: Union[List[torch.Tensor], List[np.ndarray]],
                 context_coverage: Union[np.ndarray, torch.Tensor],
                 query_coverage: Union[np.ndarray, torch.Tensor],
                 state: Union[np.ndarray, torch.Tensor, TupleTensor] = None):
        """
        The hypothesis which save useful information for beam search.
        用来保存beam search时有用信息的类
        :param tokens: `list`, shape: [dec_len, ], represent current decoder tokens id (the oov will be converted)
        :param target_input_tokens: `list`, shape: [dec_len, ], represent the input tokens id to the decoder
        :param log_probs: `list`, shape: [dec_len, ], the log probability of each token
        :param context_attn_dists: shape: [context_len], dists attended to context
        :param query_attn_dists: shape: [query_len], dists attended to query
        :param context_coverage: shape: [context_len], dists coverage attended to context
        :param query_coverage: shape: [query_len], dists coverage attended to query
        :param state: shape: [hidden_size], current step final states (only valid is rnn)
        """
        self.tokens = tokens
        self.target_input_tokens = target_input_tokens
        self.log_probs = log_probs
        self.context_attn_dists = context_attn_dists
        self.query_attn_dists = query_attn_dists
        self.context_coverage = context_coverage
        self.query_coverage = query_coverage
        self.state = state

    def extend(self, token: int, target_input_token: int, log_prob: float,
               context_attn_dist: Union[np.ndarray, torch.Tensor],
               query_attn_dist: Union[np.ndarray, torch.Tensor],
               context_coverage: Union[np.ndarray, torch.Tensor],
               query_coverage: Union[np.ndarray, torch.Tensor],
               state: Union[np.ndarray, torch.Tensor, TupleTensor] = None):

        return Hypothesis(
            tokens=self.tokens + [token],
            target_input_tokens=self.target_input_tokens + [target_input_token],
            log_probs=self.log_probs + [log_prob],
            context_attn_dists=self.context_attn_dists + [context_attn_dist],
            query_attn_dists=self.query_attn_dists + [query_attn_dist],
            context_coverage=context_coverage,
            query_coverage=query_coverage,
            state=state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)


def sort_hyps(hyps):
    """Sort the hypothesis according to average log probabilities."""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


def run_beam_search(model: Model,
                    vocab: Vocabulary,
                    params: Params,
                    context_output: torch.Tensor,
                    query_output: torch.Tensor,
                    context_mask: torch.Tensor,
                    query_mask: torch.Tensor,
                    extend_context_ids: torch.Tensor,
                    extend_query_ids: torch.Tensor,
                    max_oovs: int,
                    dec_in_state: Tuple[torch.Tensor, torch.Tensor] = None,
                    index_name: str = "tokens",
                    start_token: str = "[CLS]",
                    end_token: str = "[SEP]"):
    """
    Run beam search function.
    运行beam search的主逻辑
    """
    # save the unk_id
    unk_id = vocab.get_token_index(vocab._oov_token, namespace=index_name)
    vocab_size = vocab.get_vocab_size(namespace=index_name)
    beam_size = params['beam_size']

    # initial hyps (beam_size)
    hyps = [
        Hypothesis(tokens=[],
                   target_input_tokens=[vocab.get_token_index(start_token, namespace=index_name)],
                   log_probs=[0.0],
                   context_attn_dists=[],
                   query_attn_dists=[],
                   context_coverage=torch.zeros([context_output.size(1)]),
                   query_coverage=torch.zeros([query_output.size(1)]),
                   state=dec_in_state)
        for _ in range(beam_size)
    ]

    # get the decoder result of an instance
    results = []
    steps = 0

    while steps < params['max_dec_len'] and len(results) < beam_size:
        # take the output token ids before as new input
        target_input_tokens = [h.target_input_tokens for h in hyps]

        dec_init_states, prev_context_coverage, prev_query_coverage = [], [], []
        for h in hyps:
            dec_init_states.append(h.state)
            prev_context_coverage.append(h.context_coverage)
            prev_query_coverage.append(h.query_coverage)
        if params['decoder_type'] == 'transformer':
            dec_init_states = None
        # get the result of current step
        output_dict = model.decode_onestep(context_output,
                                           query_output,
                                           context_mask,
                                           query_mask,
                                           extend_context_ids,
                                           extend_query_ids,
                                           max_oovs,
                                           target_input_tokens,
                                           prev_context_coverage,
                                           prev_query_coverage,
                                           dec_init_states)

        # extend each hyp, get (beam_size) * beam_size hyps, and sort
        all_hyps = []
        # when step equals to 0, each hyp in hyps is same
        num_origin_hyps = 1 if steps == 0 else len(hyps)

        # [beam_size, _len]
        context_attn_dists = output_dict['context_attn_dists']
        query_attn_dists = output_dict['query_attn_dists']
        new_context_coverages = output_dict['context_coverages']
        new_query_coverages = output_dict['query_coverages']
        new_states = output_dict['new_states']
        topk_ids = output_dict['topk_ids'].detach().cpu().numpy()
        topk_log_probs = output_dict['topk_log_probs'].detach().cpu().numpy()

        for i in range(num_origin_hyps):
            # get the new decoder result of the hyp i
            # context_attn_dist: [context_len,]
            # query_attn_dist: [query_len,]
            h, context_attn_dist, query_attn_dist = hyps[i], context_attn_dists[i], query_attn_dists[i]
            new_context_coverage, new_query_coverage = new_context_coverages[i], new_query_coverages[i]
            if new_states is None:
                new_state = None
            else:
                new_state = (new_states[0][:, i, :],
                             new_states[1][:, i, :])

            # get beam_size results of each hyp
            for j in range(beam_size):
                cur_token_id = topk_ids[i, j]
                new_hyp = h.extend(token=cur_token_id,
                                   target_input_token=cur_token_id if cur_token_id < vocab_size else unk_id,
                                   log_prob=topk_log_probs[i, j],
                                   context_attn_dist=context_attn_dist,
                                   query_attn_dist=query_attn_dist,
                                   context_coverage=new_context_coverage,
                                   query_coverage=new_query_coverage,
                                   state=new_state)
                all_hyps.append(new_hyp)

        # get the result the current step and sort
        hyps = []
        for h in sort_hyps(all_hyps):
            if h.latest_token == vocab.get_token_index(end_token,
                                                       namespace=index_name):
                if steps >= params['min_dec_len']:
                    results.append(h)
            else:
                hyps.append(h)
            # preserve beam_size hyps and go to next iter
            if len(hyps) == beam_size or len(results) == beam_size:
                break
        steps += 1

    # if no result finally, return current hyps
    if len(results) == 0:
        results = hyps

    # sort the final results and return
    hyps_sorted = sort_hyps(results)
    return hyps_sorted


def convert_indices_to_string(hyps: List[List[Hypothesis]],
                              metadata: List[Dict[str, Any]],
                              vocab: Vocabulary,
                              end_token: str = "[SEP]",
                              return_all: bool = False,
                              index_name: str = "tokens"):
    """Convert the token ids in hyps to result rewrite string."""
    vocab_size = vocab.get_vocab_size(namespace=index_name)

    rewrite_tokens = []
    rewrite_strings = []
    origin_rewrite_strings = []
    origin_query_strings = []
    other_rewrite_strings = []
    # for each instance
    for hyp, mdata in zip(hyps, metadata):
        oovs = mdata['oovs']
        if 'rewrite' in mdata:
            origin_query_words = mdata['query_words']
            origin_rw_words = mdata['rewrite']
        other_rw_string = []
        for i, h in enumerate(hyp):
            word_ids = h.tokens
            words = []
            for wid in word_ids:
                try:
                    w = vocab.get_token_from_index(wid, namespace=index_name)
                except Exception:
                    assert oovs is not None, "Error: No oov words in the dialogue!"
                    dialogue_oov_idx = wid - vocab_size
                    try:
                        w = oovs[dialogue_oov_idx]
                    except Exception:
                        raise ValueError(
                            "Error: model produce word ID %i corresponds to dialogue OOV %i "
                            "but this example only has %i OOV words." %
                            (wid, dialogue_oov_idx, len(oovs)))
                words.append(w)
            if i == 0:
                if 'rewrite' in mdata:
                    origin_query_strings.append(origin_query_words)
                    origin_rewrite_strings.append(origin_rw_words)

                # find the end position
                try:
                    stop_idx = words.index(end_token)
                    words = words[:stop_idx]
                except ValueError:
                    pass
                rewrite_tokens.append(words)
                rewrite_strings.append("".join(words))
                if not return_all:
                    break
            else:
                other_rw_string.append("".join(words))
        other_rewrite_strings.append(other_rw_string)

    # return result string, rewrite_token, gold_target and origin_query
    output_dict = {}
    output_dict['rewrite_string'] = rewrite_strings
    output_dict['rewrite_token'] = rewrite_tokens
    output_dict['gold_target'] = origin_rewrite_strings
    output_dict['origin_query'] = origin_query_strings
    # if return_all return other rewrite results (not only the best)
    if return_all:
        output_dict['other_rewrites'] = other_rewrite_strings
    return output_dict
