# System architecture

```text
React + Vite customer and agent interface
        │ HTTP POST /tickets/analyse
        ▼
FastAPI backend ──► NLP pipeline ──► intent, sentiment, entities
        │                    │
        │                    └──► priority + safety escalation rules
        │
        ├──► knowledge retrieval (Sentence Transformers + FAISS; TF-IDF fallback)
        ├──► grounded policy suggestion / optional local Ollama draft / human escalation
        └──► SQLite development database / PostgreSQL deployment database
```

## API

- `GET /health` — liveness check.
- `POST /tickets/analyse` — triage, retrieve policy evidence, and persist a ticket.
- `GET /tickets` — support-agent queue.
- `PATCH /tickets/{id}` — record a human outcome and ticket status.

## Grounding rule

The response component may summarize retrieved policy text, but it may not assert live delivery, refund, transaction, or account facts. Low-confidence and sensitive cases bypass automated advice and enter the human queue.

## Optional local LLM

Set `ENABLE_LLM_GENERATION=true` only after running Ollama locally. The generator receives the customer message and retrieved policy chunks, is instructed to cite policy IDs, and is bypassed for human-escalated tickets. This is a drafting feature, not the project's primary NLP contribution.
