from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.database import SessionLocal, Ticket, initialise_database, save_ticket
from src.config import ROOT
from src.auth import require_agent, verify_agent_key
from src.service import analyse_ticket


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialise_database()
    yield


app = FastAPI(title="Intelligent Customer Support API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    message: str = Field(min_length=3, max_length=5000)
    save: bool = True


class TicketResolution(BaseModel):
    status: str = Field(pattern="^(open|in_progress|resolved)$")
    agent_outcome: str | None = Field(default=None, max_length=3000)


class AgentLogin(BaseModel):
    access_key: str = Field(min_length=8, max_length=256)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(require_agent)])
def metrics() -> dict:
    """Expose reproducible evaluation outputs for the agent dashboard."""
    outputs = ROOT / "outputs"

    def read_metric(filename: str) -> dict | None:
        path = outputs / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return {
        "baseline": read_metric("banking77_test_metrics.json"),
        "transformer": read_metric("distilbert_banking77_test_metrics.json"),
        "retrieval": read_metric("retrieval_eval_metrics.json"),
        "safety": read_metric("safety_eval_metrics.json"),
    }


@app.post("/tickets/analyse")
def analyse(request: TicketRequest) -> dict:
    result = analyse_ticket(request.message)
    if request.save:
        result["ticket"] = save_ticket(result, request.message)
    return result


@app.post("/agent/login")
def login_agent(login: AgentLogin) -> dict:
    verify_agent_key(login.access_key)
    return {"authenticated": True}


@app.get("/tickets", dependencies=[Depends(require_agent)])
def list_tickets(limit: int = 50) -> list[dict]:
    with SessionLocal() as session:
        tickets = session.scalars(select(Ticket).order_by(Ticket.created_at.desc()).limit(min(max(limit, 1), 200))).all()
        return [ticket.as_dict() for ticket in tickets]


@app.patch("/tickets/{ticket_id}", dependencies=[Depends(require_agent)])
def resolve_ticket(ticket_id: str, update: TicketResolution) -> dict:
    with SessionLocal() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        ticket.status = update.status
        ticket.agent_outcome = update.agent_outcome
        session.commit()
        session.refresh(ticket)
        return ticket.as_dict()
