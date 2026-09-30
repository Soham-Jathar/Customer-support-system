# Data card and provenance

## Training sources

1. BANKING77: the primary public benchmark. It contains 13,083 English online-banking customer-service queries with 77 fine-grained intent labels. `scripts/prepare_banking77.py` downloads its official train and test splits and writes reproducible CSV exports.
2. `demo_tickets.csv` and `annotated_extension.csv`: early project-authored examples retained only for UI demonstration. They must not be mixed with BANKING77 results or described as real customer data.

## Mapping limits

The intent classifier retains the original 77 BANKING77 labels. Broad departments are assigned only after classification using `src.config.department_for`; this separates predictive NLP from operational business rules. The current benchmark is English-only.

## Labelling protocol for priority and escalation

Annotate priority independently from sentiment. Mark `critical` for suspected fraud, unauthorized transactions, or account compromise. Mark escalation for sensitive, low-confidence, multi-issue, or explicitly human-requested tickets. Have a second annotator label at least 20% of the safety test set and report Cohen's kappa.

## Prohibited claims

Do not claim BANKING77 represents every bank or real production traffic. Do not report results from the seed UI examples as production performance. Keep the BANKING77 test split untouched until final evaluation.
