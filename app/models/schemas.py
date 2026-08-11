from typing import Optional
from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = None


class Citation(BaseModel):
    chunk_id: str
    source_doc: str
    text_snippet: str
    classification: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    needs_human_review: bool
    review_reason: Optional[str] = None
    session_id: str
    trace_id: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_doc: str
    text: str
    classification: str  # e.g. "public", "front_office_only"
    score: float
