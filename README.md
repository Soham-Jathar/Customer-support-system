# Intelligent Customer Support Triage

An explainable NLP system for banking customer-ticket classification, sentiment analysis, priority detection, knowledge-base retrieval, routing, ticket persistence, and human escalation.

## Features

- BANKING77's 77 fine-grained banking intents, retained without collapsing labels.
- TF-IDF word/character n-grams + Logistic Regression baseline with confidence scores.
- VADER sentiment analysis with an offline fallback.
- Transparent priority policy: fraud/security language is **critical** regardless of sentiment.
- Human escalation for suspected fraud, low-confidence predictions, and explicit agent requests.
- Regex-based extraction of order IDs, transaction IDs, amounts, and dates.
- FastAPI API with persistent ticket history; SQLite runs locally and PostgreSQL is configurable for deployment.
- Sentence Transformers + FAISS retrieval when installed, with an explicit TF-IDF fallback for offline demos.
- Streamlit UI, reproducible training, intent/retrieval evaluation, confusion matrix, and safety tests.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-advanced.txt
python scripts/prepare_banking77.py
python -m src.train --data data/banking77/banking77_train.csv
python scripts/evaluate.py --data data/banking77/banking77_train.csv
python scripts/evaluate_banking77_test.py
python scripts/evaluate_retrieval.py
uvicorn api.main:app --reload --port 8000
```

Open a second terminal for the primary React customer-and-agent interface:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The React frontend calls FastAPI at `http://127.0.0.1:8000` and displays the live ticket queue.

The Streamlit `app.py` remains available as a lightweight fallback dashboard, but React/Vite is the primary frontend.

## Agent access

Customer ticket submission is public in this local prototype. The Agent Console, ticket history, resolution endpoint, and model-quality metrics require an agent access key.

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `AGENT_ACCESS_KEY` in `.env` to a long local secret, install updated dependencies with `pip install -r requirements.txt`, then restart FastAPI. Use that same key only on the Agent Console sign-in screen. This is local prototype authorization, not production identity management.

## Advanced experiments

```powershell
python scripts/train_transformer.py --data data/banking77/banking77_train.csv
python scripts/evaluate_transformer.py
docker compose up -d postgres
```

Copy `.env.example` to `.env` and set `DATABASE_URL` before running FastAPI with PostgreSQL.

## Dataset workflow

```powershell
# Download BANKING77's official train/test split and preserve all 77 labels.
python scripts/prepare_banking77.py
python -m src.train --data data/banking77/banking77_train.csv
python scripts/evaluate.py --data data/banking77/banking77_train.csv
python scripts/evaluate_banking77_test.py
python scripts/evaluate_safety.py
python scripts/evaluate_retrieval.py
python scripts/compare_models.py
```

Use `data/banking77/banking77_test.csv` as the untouched final test set. Read `docs/DATA_CARD.md` before reporting results.

The seed CSV only makes the interface reproducible. For faculty evaluation, use the documented benchmark split and separately annotated safety test set. See `docs/ARCHITECTURE.md`, `docs/DATA_AND_EVALUATION.md`, and `docs/PROJECT_PROPOSAL.md`.

## Deliberate scope

This prototype does **not** generate factual answers about account balances, transaction status, refunds, or card-delivery dates. Its response module only summarizes retrieved policy text; live facts require an approved banking-system lookup.
