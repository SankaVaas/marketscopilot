import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app.agents.graph import get_graph
from app.memory import compaction, store
from app.memory.store import add_turn, init_db, new_session_id, write_audit_log
from app.models.schemas import QueryRequest, QueryResponse, TokenResponse
from app.retrieval import bm25_index
from app.security.auth import authenticate_user, create_access_token, get_current_user
from app.security.prompt_safety import detect_prompt_injection, redact_pii
from app.security.rbac import allowed_classifications

app = FastAPI(
    title="MarketsCopilot",
    description="Agentic RAG research & compliance assistant (open-source, zero-cost stack)",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup():
    init_db()
    loaded = bm25_index.load_index_from_disk()
    if not loaded:
        print(
            "WARNING: no BM25 index found on disk. Run "
            "`python -m app.retrieval.ingest data/sample_docs` before querying."
        )
    # Tracing is optional in local dev if no collector is running -- wrap defensively.
    try:
        from app.observability.tracing import setup_tracing
        setup_tracing(app)
    except Exception as e:  # noqa: BLE001
        print(f"Tracing setup skipped (collector not reachable?): {e}")


@app.post("/auth/token", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(user["username"], user["role"])
    return TokenResponse(access_token=token, role=user["role"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, current_user: dict = Depends(get_current_user)):
    trace_id = str(uuid.uuid4())
    session_id = request.session_id or new_session_id()

    # --- Prompt-safety screen (before anything touches the LLM or retrieval) ---
    if detect_prompt_injection(request.question):
        raise HTTPException(
            status_code=400,
            detail="Your message was flagged by the prompt-safety filter. Please rephrase.",
        )
    redacted_question, pii_found = redact_pii(request.question)
    if pii_found:
        # Log but proceed with the redacted version -- don't let PII reach the LLM/logs.
        print(f"[trace={trace_id}] Redacted PII types in query: {pii_found}")

    role = current_user["role"]
    username = current_user["username"]

    add_turn(session_id, "user", redacted_question)
    compaction.maybe_compact(session_id)

    graph = get_graph()
    initial_state = {
        "question": redacted_question,
        "session_id": session_id,
        "username": username,
        "role": role,
        "allowed_classifications": allowed_classifications(role),
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )

    add_turn(session_id, "assistant", result["answer"])

    write_audit_log(
        trace_id=trace_id,
        session_id=session_id,
        username=username,
        role=role,
        question=redacted_question,
        retrieved_chunk_ids=[c["chunk_id"] for c in result.get("citations", [])],
        model_used="see LLM_MODEL env var",
        needs_human_review=result.get("needs_human_review", False),
        review_reason=result.get("review_reason"),
    )

    return QueryResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        needs_human_review=result.get("needs_human_review", False),
        review_reason=result.get("review_reason"),
        session_id=session_id,
        trace_id=trace_id,
    )
