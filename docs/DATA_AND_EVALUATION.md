# Data and evaluation plan

## Dataset plan

Use BANKING77 as the primary benchmark. It supplies an official train/test split with 77 fine-grained online-banking intents. Preserve those original intent labels for training and evaluate on the untouched official test split. `data/safety_eval.csv` is a separate project-authored banking safety test set for escalation policy; it is never used to train the classifier.

## Evaluation

- Create stratified train/validation/test splits before training.
- Report accuracy only as a secondary metric; report macro precision, recall, F1, per-class F1, and confusion matrix.
- Keep the manual safety set out of training and measure priority/escalation precision and recall separately.
- Conduct error analysis for confused fine-grained labels such as unrecognized cash withdrawals versus cash-withdrawal charges, and card-payment fees versus exchange-rate issues.
- Compare DistilBERT and TF-IDF + Logistic Regression on the same official test split and report quality, latency, and failure modes.

## Current limitations

The benchmark is English-only and represents online-banking queries, not all real banking traffic. The small safety set is project-authored and should be described as a policy test, not a population-level performance claim. Add a second annotator and report inter-annotator agreement before making stronger claims.
