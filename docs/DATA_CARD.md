# Data card and provenance

## Training sources

1. **Bitext customer-support dataset**: public English customer-support utterances. `scripts/prepare_bitext_data.py` downloads it, maps source labels to the project taxonomy, and preserves a `query_type` and `intent` column.
2. **Project-owned technical-support extension**: `scripts/create_technical_extension.py` deterministically creates labelled prototype examples for login, app, checkout, website, device-setup, and product-malfunction issues. This data is synthetic and must always be described as such.
3. **Project-owned realistic paraphrase extension**: `scripts/create_realistic_paraphrases.py` creates natural, short support requests for training-only robustness augmentation. It is excluded from the held-out test split and must always be described as synthetic.
4. **`demo_tickets.csv` and `annotated_extension.csv`**: small project-authored development examples. They support repeatable local demos only.

## Taxonomy

Broad query types are payment, delivery, refund, account access, technical support, and other. Fine intents refine those operational queues. The two-level labels let the project separately evaluate broad routing and detailed classification.

## Split and leakage control

`scripts/build_training_set.py` creates a deterministic stratified 80/20 train/test split with `random_state=42`. The test CSV is not used for model fitting or rule development. Retrieval and safety evaluation CSVs are separate from classifier training.

## Limitations and responsible claims

The dataset is English-focused, and the technical-support extension is synthetic. Report the exact training mix and do not present its results as real-company or production performance. Before a production study, replace synthetic examples with consented, de-identified tickets; double-annotate priority/escalation labels; and report inter-annotator agreement.
