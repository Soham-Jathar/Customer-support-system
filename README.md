# Intelligent Customer Support System Using NLP

An explainable NLP system that understands customer messages, classifies the problem, estimates priority, retrieves policy evidence, and routes sensitive or uncertain tickets to human support agents.

## What it implements

- A hierarchical classifier: **six query types** (`payment`, `delivery`, `refund`, `account_access`, `technical_support`, `other`) and fine-grained issue intents inside those queues.
- TF-IDF word and character n-grams with Logistic Regression as the reproducible baseline; fine-tuned DistilBERT is an optional comparison.
- VADER sentiment analysis, used as context rather than as the sole urgency signal.
- Transparent priority and escalation rules for suspected fraud, high-impact delivery/payment issues, low confidence, and explicit human requests.
- Operational entity extraction for order, tracking, and transaction references, amounts, and dates.
- Sentence Transformers + FAISS policy retrieval when available, with a visible TF-IDF fallback.
- Separate React customer portal and access-key-protected agent console, FastAPI API, SQLite development persistence, and optional PostgreSQL deployment.
- End-to-end ticket lifecycle: customer-safe status tracking, agent-reviewed corrections, resolution notes, and restricted CSV feedback export for future retraining.

## Start the application

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-advanced.txt
uvicorn api.main:app --reload --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API documentation is at `http://127.0.0.1:8000/docs`.

## Build the generic NLP models

Run these from the project root after activating `.venv`:

```powershell
# Public Bitext data, mapped to the project's explicit two-level taxonomy.
python scripts/prepare_bitext_data.py

# Project-owned, transparently synthetic technical-support extension.
python scripts/create_technical_extension.py

# Training-only paraphrases for more natural demo wording.
python scripts/create_realistic_paraphrases.py

# Deterministic 80/20 train/held-out split.
python scripts/build_training_set.py

# Baseline fine-intent and broad query-type models.
python -m src.train --data data/generic/train.csv
python scripts/train_query_type.py --data data/generic/train.csv

# Held-out evaluation and safety/retrieval evaluation.
python scripts/evaluate.py --data data/generic/test.csv
python scripts/evaluate_query_type.py --test data/generic/test.csv
python scripts/evaluate_safety.py
python scripts/evaluate_retrieval.py
```

Optional DistilBERT comparison:

```powershell
python scripts/train_transformer.py --data data/generic/train.csv --output models/distilbert-generic-intent
python scripts/evaluate_transformer.py --model models/distilbert-generic-intent --test data/generic/test.csv
python scripts/compare_models.py
```

## Agent access

Customer submission is public in this local prototype. The agent queue, stored ticket history, resolution endpoint, and model metrics require a local access key.

```powershell
Copy-Item .env.example .env
notepad .env
```

Set a long `AGENT_ACCESS_KEY`, restart FastAPI, then use that key only on the Agent Console sign-in screen. This is prototype authorization, not production identity management.

## Human-feedback retraining loop

An agent can export reviewed corrections from the Operations/Agent Console. Validate the export before using it in retraining; the held-out test CSV is never modified.

```powershell
python scripts/prepare_feedback_for_retraining.py --input "$env:USERPROFILE\Downloads\agent_feedback.csv"
python scripts/build_retraining_set.py
python -m src.train --data data/retraining/train_with_feedback.csv
python scripts/train_query_type.py --data data/retraining/train_with_feedback.csv
```

## Deliberate safety boundary

The system never invents delivery dates, order status, refund approval, or account outcomes. It can only summarize retrieved policy text. Live operational facts require a company-system lookup or human review.

See [the proposal](docs/PROJECT_PROPOSAL.md), [data card](docs/DATA_CARD.md), [evaluation plan](docs/DATA_AND_EVALUATION.md), and [architecture](docs/ARCHITECTURE.md).
