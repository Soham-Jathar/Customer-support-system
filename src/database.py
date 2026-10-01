"""Ticket persistence. SQLite is the development default; PostgreSQL is supported via DATABASE_URL."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import ROOT
from src.preprocess import mask_entities, mask_sensitive_data
from src.taxonomy import query_type_for

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'support_generic.db').as_posix()}")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    customer_message: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[str] = mapped_column(String(16))
    sentiment: Mapped[str] = mapped_column(String(16))
    priority: Mapped[str] = mapped_column(String(16))
    department: Mapped[str] = mapped_column(String(64))
    escalated: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(24), default="open")
    escalation_reasons: Mapped[str] = mapped_column(Text, default="[]")
    entities: Mapped[str] = mapped_column(Text, default="{}")
    retrieved_sources: Mapped[str] = mapped_column(Text, default="[]")
    attachment_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    agent_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "created_at": self.created_at.isoformat(),
            "customer_message": self.customer_message, "intent": self.intent,
            "query_type": query_type_for(self.intent),
            "confidence": float(self.confidence), "sentiment": self.sentiment,
            "priority": self.priority, "department": self.department,
            "escalated": self.escalated == "true", "status": self.status,
            "escalation_reasons": json.loads(self.escalation_reasons),
            "entities": json.loads(self.entities), "retrieved_sources": json.loads(self.retrieved_sources),
            "attachment_name": self.attachment_name,
            "agent_outcome": self.agent_outcome,
        }


class TicketReview(Base):
    """Human feedback retained separately from the model's original decision.

    Keeping predictions and reviewed labels side by side creates an auditable
    feedback dataset without overwriting what the system originally predicted.
    """
    __tablename__ = "ticket_reviews"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    reviewer: Mapped[str] = mapped_column(String(80), default="Support agent")
    final_query_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    final_intent: Mapped[str | None] = mapped_column(String(96), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict:
        return {
            "reviewer": self.reviewer,
            "final_query_type": self.final_query_type,
            "final_intent": self.final_intent,
            "notes": self.notes,
            "reviewed_at": self.reviewed_at.isoformat(),
        }


def initialise_database() -> None:
    Base.metadata.create_all(bind=engine)
    # SQLite create_all does not add new columns to an existing local database.
    if DATABASE_URL.startswith("sqlite") and "attachment_name" not in {column["name"] for column in inspect(engine).get_columns("tickets")}:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE tickets ADD COLUMN attachment_name VARCHAR(255)"))


def save_ticket(result: dict, customer_message: str, attachment_name: str | None = None) -> dict:
    with SessionLocal() as session:
        ticket = Ticket(
            customer_message=mask_sensitive_data(customer_message), intent=result["intent"], confidence=str(result["confidence"]),
            sentiment=result["sentiment"], priority=result["priority"], department=result["department"],
            escalated=str(result["escalate_to_human"]).lower(),
            escalation_reasons=json.dumps(result["escalation_reasons"]), entities=json.dumps(mask_entities(result["entities"])),
            retrieved_sources=json.dumps(result.get("retrieved_sources", [])),
            attachment_name=attachment_name,
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket.as_dict()
