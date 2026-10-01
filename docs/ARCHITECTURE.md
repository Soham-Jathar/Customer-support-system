# System architecture

```text
React + Vite customer portal        React restricted agent console
              │                                  │
              └──────────── FastAPI ─────────────┘
                              │
              ▼
      Text normalization and entity extraction
              │
              ├── Fine-intent classifier: TF-IDF + Logistic Regression
              ├── Query-type classifier: six support queues
              ├── Sentiment: VADER (contextual signal)
              └── Priority and escalation policy rules
                              │
              ├── Sentence Transformers + FAISS policy retrieval
              │      (explicit TF-IDF fallback)
              ├── Grounded response or human-agent handoff
              └── SQLite development database / PostgreSQL deployment database
                              │
                              ▼
      Agent-reviewed corrections and outcomes → feedback CSV for retraining
```

## API

- `GET /health` — liveness check.
- `POST /tickets/analyse` — analyse a customer request, retrieve policies, and persist a masked ticket.
- `POST /agent/login` — verify a local prototype agent key.
- `GET /tickets` — restricted agent queue.
- `GET /tickets/{ticket_id}/status` — customer-safe status lookup by opaque reference.
- `PATCH /tickets/{id}` — restricted human outcome and ticket-status update.
- `GET /feedback/export` — restricted export of reviewed labels for future retraining.
- `GET /metrics` — restricted saved evaluation summary.

## Grounding boundary

The response component can only summarize retrieved policy documents. It cannot assert a live delivery, payment, refund, or account fact. Low-confidence, security-sensitive, and explicitly human-requested tickets are routed to the human escalation queue.

## Optional local LLM

When `ENABLE_LLM_GENERATION=true` and Ollama is running, a local model receives only the customer message and retrieved policy excerpts. It is instructed to cite policy IDs and is bypassed whenever a human escalation is required. The LLM is a constrained drafting layer, not the main NLP contribution.

## Feedback retraining safeguard

Agent corrections are exported separately. `scripts/prepare_feedback_for_retraining.py` validates complete, taxonomy-consistent labels before `scripts/build_retraining_set.py` appends them to training data. The held-out test split remains unchanged, so each retrained version can still be evaluated fairly.
