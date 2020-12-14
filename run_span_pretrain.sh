allennlp train ./experiments/bert_span_pretrain.jsonnet \
  -s ../../models/spdr/restoration_200k_pretrain \
  --include-package resolution \
  -f