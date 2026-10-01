from __future__ import annotations

from contextlib import asynccontextmanager
import csv
import json
from io import StringIO
from pathlib import Path
from collections import Counter

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select

from src.database import SessionLocal, Ticket, TicketReview, initialise_database, save_ticket
from src.config import ROOT
from src.auth import require_agent, verify_agent_key
from src.service import analyse_ticket
from src.attachments import extract_attachment_text


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
    reviewer: str = Field(default="Support agent", min_length=2, max_length=80)
    final_query_type: str | None = Field(default=None, max_length=64)
    final_intent: str | None = Field(default=None, max_length=96)


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
        "baseline": read_metric("generic_intent_test_metrics.json"),
        "transformer": read_metric("distilbert_generic_intent_test_metrics.json"),
        "retrieval": read_metric("retrieval_eval_metrics.json"),
        "safety": read_metric("safety_eval_metrics.json"),
    }


@app.get("/analytics", dependencies=[Depends(require_agent)])
def analytics() -> dict:
    """Operational summary for the restricted support-agent dashboard."""
    with SessionLocal() as session:
        tickets = session.scalars(select(Ticket)).all()
        reviews = session.scalars(select(TicketReview)).all()
        query_types = Counter(ticket.as_dict()["query_type"] for ticket in tickets)
        priorities = Counter(ticket.priority for ticket in tickets)
        statuses = Counter(ticket.status for ticket in tickets)
        total = len(tickets)
        return {
            "total_tickets": total,
            "open_tickets": total - statuses.get("resolved", 0),
            "resolved_tickets": statuses.get("resolved", 0),
            "escalated_tickets": sum(ticket.escalated == "true" for ticket in tickets),
            "reviewed_tickets": len(reviews),
            "feedback_coverage": round(len(reviews) / total, 4) if total else 0,
            "by_query_type": dict(sorted(query_types.items())),
            "by_priority": dict(sorted(priorities.items())),
            "by_status": dict(sorted(statuses.items())),
        }


@app.post("/tickets/analyse")
def analyse(request: TicketRequest) -> dict:
    result = analyse_ticket(request.message)
    if request.save:
        result["ticket"] = save_ticket(result, request.message)
    return result


@app.post("/tickets/analyse-attachment")
async def analyse_attachment(file: UploadFile = File(...), save: bool = Form(True)) -> dict:
    """Extract a supported attachment in memory, then use the normal pipeline."""
    attachment = await extract_attachment_text(file)
    result = analyse_ticket(attachment.text)
    result["attachment"] = {
        "name": attachment.filename,
        "format": attachment.extension.upper(),
        "text_characters": len(attachment.text),
        "stored": False,
    }
    if save:
        result["ticket"] = save_ticket(result, attachment.text, attachment_name=attachment.filename)
    return result


@app.post("/agent/login")
def login_agent(login: AgentLogin) -> dict:
    verify_agent_key(login.access_key)
    return {"authenticated": True}


def ticket_payload(ticket: Ticket, session) -> dict:
    payload = ticket.as_dict()
    review = session.scalar(select(TicketReview).where(TicketReview.ticket_id == ticket.id))
    payload["review"] = review.as_dict() if review else None
    return payload


@app.get("/tickets", dependencies=[Depends(require_agent)])
def list_tickets(limit: int = 50) -> list[dict]:
    with SessionLocal() as session:
        tickets = session.scalars(select(Ticket).order_by(Ticket.created_at.desc()).limit(min(max(limit, 1), 200))).all()
        return [ticket_payload(ticket, session) for ticket in tickets]


@app.get("/tickets/{ticket_id}/status")
def ticket_status(ticket_id: str) -> dict:
    """Customer-safe status lookup by the opaque ticket reference."""
    with SessionLocal() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket reference not found")
        return {
            "id": ticket.id,
            "created_at": ticket.created_at.isoformat(),
            "status": ticket.status,
            "priority": ticket.priority,
            "assigned_queue": ticket.department,
            "human_review": ticket.escalated == "true",
        }


@app.patch("/tickets/{ticket_id}", dependencies=[Depends(require_agent)])
def resolve_ticket(ticket_id: str, update: TicketResolution) -> dict:
    with SessionLocal() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        ticket.status = update.status
        ticket.agent_outcome = update.agent_outcome
        if update.final_query_type or update.final_intent or update.agent_outcome:
            review = session.scalar(select(TicketReview).where(TicketReview.ticket_id == ticket.id))
            if review is None:
                review = TicketReview(ticket_id=ticket.id)
                session.add(review)
            review.reviewer = update.reviewer
            review.final_query_type = update.final_query_type
            review.final_intent = update.final_intent
            review.notes = update.agent_outcome
        session.commit()
        session.refresh(ticket)
        return ticket_payload(ticket, session)


@app.get("/feedback/export", dependencies=[Depends(require_agent)])
def export_feedback() -> Response:
    """Export human-reviewed labels for an auditable retraining workflow."""
    with SessionLocal() as session:
        reviews = session.scalars(select(TicketReview).order_by(TicketReview.reviewed_at.desc())).all()
        buffer = StringIO()
        fields = ["ticket_id", "customer_message", "predicted_query_type", "predicted_intent", "final_query_type", "final_intent", "reviewer", "notes", "status", "reviewed_at"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for review in reviews:
            ticket = session.get(Ticket, review.ticket_id)
            if ticket is None:
                continue
            writer.writerow({
                "ticket_id": ticket.id, "customer_message": ticket.customer_message,
                "predicted_query_type": ticket.as_dict()["query_type"], "predicted_intent": ticket.intent,
                "final_query_type": review.final_query_type or "", "final_intent": review.final_intent or "",
                "reviewer": review.reviewer, "notes": review.notes or "", "status": ticket.status,
                "reviewed_at": review.reviewed_at.isoformat(),
            })
        return Response(content=buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=agent_feedback.csv"})
