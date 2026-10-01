# Faculty project proposal

## Title

**Intelligent Customer Support System Using Explainable NLP and Grounded Retrieval**

## Problem statement

Customer-support teams receive large volumes of unstructured messages about payments, delayed deliveries, refunds, account access, and technical failures. Manually identifying the complaint type, urgency, responsible department, and appropriate policy is slow and can delay high-risk cases.

This project develops an explainable NLP-based customer-support triage system. It predicts a broad query type and a fine-grained intent, analyses sentiment, extracts operational references, applies a transparent priority and escalation policy, retrieves relevant support policies, and routes each ticket to the appropriate queue. Suspected fraud, low-confidence predictions, and explicit requests for an agent are escalated to a human with a structured handoff. Classification, retrieval, and escalation are evaluated separately.

## NLP contribution

1. Hierarchical supervised classification: six broad query types and fine-grained customer-support intents.
2. TF-IDF word/character n-gram + Logistic Regression baseline, compared with an optional fine-tuned DistilBERT model.
3. VADER sentiment analysis used as a contextual feature, not a replacement for urgency detection.
4. Explainable priority detection combining query type, classifier confidence, sentiment, and explicit safety rules.
5. Entity extraction for order, tracking, and transaction IDs, amounts, and dates.
6. Semantic policy retrieval using Sentence Transformers and FAISS, with an evaluated TF-IDF fallback.
7. FastAPI, React customer/agent separation, ticket persistence, and a human-resolution feedback workflow.

## Evaluation plan

The project uses an 80/20 stratified held-out split created before fitting from mapped public Bitext customer-support data plus a clearly labelled project-owned technical-support extension. It reports accuracy, macro F1, weighted F1, per-intent results, and a confusion matrix. Query type, retrieval Recall@5, and safety escalation precision/recall/F1 are evaluated independently. The application exposes saved metrics in the restricted agent console.

## Current reproducible results

The current held-out set contains 1,026 examples. The TF-IDF + Logistic Regression baseline was selected for the running prototype because it outperformed the fine-tuned DistilBERT comparison on the same split.

| Component | Measure | Result |
| --- | --- | ---: |
| TF-IDF + Logistic Regression | Accuracy / macro F1 | **0.9844 / 0.9783** |
| Fine-tuned DistilBERT | Accuracy / macro F1 | 0.9659 / 0.8981 |
| Broad query-type classifier | Accuracy / macro F1 | **0.9912 / 0.9880** |
| Policy retrieval | Recall@5 on 18 labelled queries | **1.0000** |
| Safety policy | Escalation F1 on 19 authored cases | **0.9333** |

These results are reproducible from the scripts in the project README. They are prototype results, not claims about production customer traffic.

## Ethical and safety constraints

The system must not invent delivery dates, order status, refund approvals, or account outcomes. It stores only the minimum routing data and masks common sensitive data before persistence. Security-sensitive, uncertain, and customer-requested human cases are deferred to people. The current dataset is English-focused and prototype-only; it must not be presented as production performance or as representative of every company.
