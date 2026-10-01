# Data and evaluation plan

## Dataset workflow

1. Download and map Bitext into explicit query-type and fine-intent labels.
2. Build the documented technical-support extension because Bitext has no dedicated technical class.
3. Create the stratified held-out split before training.
4. Fit the fine-intent baseline only on `data/generic/train.csv` and evaluate once on `data/generic/test.csv`.

## Metrics

- Fine intent: accuracy, macro F1, weighted F1, per-intent precision/recall/F1, confusion matrix.
- Query type: accuracy, macro F1, per-type report.
- Retrieval: Recall@5 over labelled policy queries.
- Safety policy: priority accuracy plus escalation precision, recall, and F1 on a separate authored safety set.
- Model comparison: use the same held-out split for TF-IDF + Logistic Regression and DistilBERT.

## Error analysis

Review likely confusion pairs such as delayed delivery versus tracking, payment failure versus duplicate charge, refund request versus pending refund, and website error versus checkout error. Separately inspect false negatives for critical safety language because those errors are more consequential than routine routing mistakes.

## Current limitations

Results reflect the mapped public dataset and project-owned synthetic technical extension, not live company traffic. Sentiment is English-oriented and rule patterns must be reviewed before use with another company or language.
