"""
Postgres-backed memory store:
  - short-term conversational turns (per session)
  - long-term / compacted summaries (per session, after N turns)
  - audit log (every query, retrieved chunks, model used, decision trace)

Swap Postgres for RDS in production -- SQLAlchemy means the app code doesn't
change, only the connection string.
"""
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, create_engine, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

Base = declarative_base()
engine = create_engine(settings.postgres_dsn, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SessionSummary(Base):
    """Long-term memory: compacted summary once a session exceeds N turns."""
    __tablename__ = "session_summaries"

    session_id = Column(String, primary_key=True)
    summary = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trace_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(JSON, nullable=False)
    model_used = Column(String, nullable=False)
    needs_human_review = Column(Integer, nullable=False, default=0)  # 0/1
    review_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)


# --- Short-term memory ---

MAX_TURNS_BEFORE_COMPACTION = 12


def add_turn(session_id: str, role: str, content: str):
    with SessionLocal() as db:
        db.add(ConversationTurn(session_id=session_id, role=role, content=content))
        db.commit()


def get_recent_turns(session_id: str, limit: int = 10) -> list[dict]:
    with SessionLocal() as db:
        rows = (
            db.query(ConversationTurn)
            .filter(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.created_at.desc())
            .limit(limit)
            .all()
        )
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def count_turns(session_id: str) -> int:
    with SessionLocal() as db:
        return db.query(ConversationTurn).filter(
            ConversationTurn.session_id == session_id
        ).count()


# --- Long-term memory / compaction ---

def get_summary(session_id: str) -> str | None:
    with SessionLocal() as db:
        row = db.query(SessionSummary).filter(
            SessionSummary.session_id == session_id
        ).first()
        return row.summary if row else None


def upsert_summary(session_id: str, summary: str):
    with SessionLocal() as db:
        row = db.query(SessionSummary).filter(
            SessionSummary.session_id == session_id
        ).first()
        if row:
            row.summary = summary
            row.updated_at = datetime.now(timezone.utc)
        else:
            db.add(SessionSummary(session_id=session_id, summary=summary))
        db.commit()


def new_session_id() -> str:
    return str(uuid.uuid4())


# --- Audit log ---

def write_audit_log(
    trace_id: str,
    session_id: str,
    username: str,
    role: str,
    question: str,
    retrieved_chunk_ids: list[str],
    model_used: str,
    needs_human_review: bool,
    review_reason: str | None,
):
    with SessionLocal() as db:
        db.add(AuditLog(
            trace_id=trace_id,
            session_id=session_id,
            username=username,
            role=role,
            question=question,
            retrieved_chunk_ids=json.dumps(retrieved_chunk_ids),
            model_used=model_used,
            needs_human_review=1 if needs_human_review else 0,
            review_reason=review_reason,
        ))
        db.commit()
