# coding=utf-8
# @Author: 莫冉
# @Date: 2020-07-08
import torch
from allennlp.nn.util import replace_masked_values, min_value_of_dtype


def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    """
    mask.dim() should be equal to x.dim()
    """
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))


def get_best_span(span_start_logits: torch.Tensor,
                  span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    We call the inputs "logits" - they could either be un-normalized logits or normalized log
    probabilities. A log-softmax operation is a constant shifting of the entire logit vector,
    so taking an argmax over either one gives the same result.
    :param span_start_logits: [B, _len]
    :param span_end_logits: [B, _len]
    :return:
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")

    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries
    # where the span ends before is starts.
    # the lower triangle is `0`, and it's log is `-inf`
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax. We
    # can recover the start and end indices from this flattened list using simple modular arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(dim=-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)