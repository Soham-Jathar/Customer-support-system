# Faculty project proposal

## Title

**Intelligent Banking Customer Support System Using NLP**

## Problem statement

Banking-support teams manually read large volumes of unstructured customer messages before identifying the underlying issue, its urgency, and the responsible department. This delays resolution and can cause high-risk cases—such as unrecognized card payments, suspicious cash withdrawals, or account compromise—to be missed.

This project develops an explainable NLP-based triage system for online-banking support. It classifies a customer message into one of 77 fine-grained BANKING77 intents, analyses sentiment, extracts operational references, predicts priority through machine learning plus explicit safety rules, retrieves supporting policy documents, and routes the ticket to the appropriate department. Fraud-related, low-confidence, or human-requested cases are escalated to a human agent with a structured handoff summary. The project evaluates intent classification, retrieval, and escalation separately.

## NLP contribution

1. Baseline supervised intent classifier using TF-IDF word and character n-grams with Logistic Regression.
2. Sentiment analysis as a contextual signal, not a substitute for urgency detection.
3. Explainable priority and escalation policy combining intent, classifier confidence, and risk language.
4. Regex-based entity extraction for transaction IDs, card references, amounts, and dates.
5. Semantic policy retrieval using Sentence Transformers and FAISS, with a TF-IDF fallback.
6. Fine-tuned DistilBERT comparison using the same BANKING77 official test set.
7. FastAPI, React agent workspace, persistent ticket storage, and human-resolution workflow.

## Implemented evaluation results

The models were evaluated on BANKING77's untouched 3,080-message official test split.

| Component | Measure | Result |
| --- | --- | ---: |
| TF-IDF + Logistic Regression | Accuracy / macro F1 | **0.9049 / 0.9049** |
| Fine-tuned DistilBERT | Accuracy / macro F1 | 0.8724 / 0.8671 |
| Policy retrieval | Recall@5 on 14 labelled banking queries | 1.0000 |
| Safety policy | Escalation F1 on 20 authored safety cases | 0.7059 |

The baseline is the deployed classifier because it obtained the best official-test macro F1. Retrieval and safety figures are reported separately because they measure different components. The retrieval and safety sets are small project-authored evaluation sets, not production-performance claims.

## Ethical and safety constraints

The system must not invent account balances, transaction status, refund approvals, exchange rates, or card-delivery dates. It retains only the minimum ticket data necessary for routing, masks personal information in real deployments, and defers security-sensitive or uncertain cases to people. BANKING77 is English-only and may not represent all banks or real customer traffic.
