allennlp train ./experiments/bert_span_pointer_resolution.jsonnet \
  -s ../../models/spdr/restoration_200k \
  --include-package resolution \
  -f