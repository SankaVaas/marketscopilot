# MarketsCopilot

An agentic, retrieval-augmented research & compliance assistant built for a **zero recurring-cost stack** — no commercial LLM APIs, no paid SaaS trials. Every component is open-source and self-hostable, and designed so each piece has a drop-in commercial/managed equivalent for when the startup is ready to pay for scale (see `docs/path_to_production.md`).

## What it does

A user asks a research or compliance question. A multi-agent LangGraph pipeline:
1. **Planner agent** — decides what needs to be retrieved and whether the query is in-scope.
2. **Retrieval agent** — runs hybrid search (BM25 + dense vectors) over an internal document store, with a retrieval-evaluation harness for quality tracking.
3. **Compliance agent** — checks retrieved content and the draft answer against simple policy rules (entitlement/classification tags) before anything is returned.
4. **Composer agent** — writes the final answer with inline citations back to source chunks, and flags low-confidence answers for human review.

Every request is authenticated (JWT + RBAC), logged to an audit trail (Postgres), traced (OpenTelemetry), and screened by a lightweight prompt-safety / DLP filter.

## Stack (all free / self-hosted)

| Purpose | Tool |
|---|---|
| LLM inference (dev) | Ollama |
| LLM inference (prod) | vLLM, serving open weights from Hugging Face |
| Embeddings | `sentence-transformers` (BAAI/bge-large-en-v1.5) |
| Vector search | Qdrant |
| Lexical search | `rank_bm25` |
| Orchestration | LangGraph |
| Memory / audit log | Postgres |
| Object storage | MinIO (S3-API compatible) |
| Tracing | OpenTelemetry (console/Jaeger exporter) |
| API | FastAPI |

## Quick start (local dev, CPU-friendly)

```bash
# 1. Copy env file and adjust if needed
cp .env.example .env

# 2. Pull a small local model with Ollama (run this on the host, or use the ollama service in compose)
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text   # optional, only if you want Ollama for embeddings too

# 3. Bring up the full stack
docker compose up --build

# 4. Ingest the sample documents into the vector + lexical indexes
docker compose exec app python -m app.retrieval.ingest data/sample_docs

# 5. Call the API
curl -X POST http://localhost:8000/auth/token -d "username=analyst1&password=analyst1" 
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our current view on rate cuts this quarter?"}'
```

Swagger docs: `http://localhost:8000/docs`

## Switching from Ollama (dev) to vLLM (prod)

Nothing in `app/` changes. Both expose an OpenAI-compatible endpoint at `/v1/chat/completions`.
Just point `LLM_BASE_URL` in `.env` at the vLLM server instead of Ollama, e.g.:

```bash
# Dev
LLM_BASE_URL=http://ollama:11434/v1
LLM_MODEL=qwen2.5:7b-instruct-q4_K_M

# Prod (vLLM, launched separately — see docs/path_to_production.md)
LLM_BASE_URL=http://vllm:8001/v1
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
```

## Running the evaluation harness

```bash
docker compose exec app python -m eval.retrieval_eval
```
This scores retrieval precision@k and answer groundedness against `eval/sample_queries.json` and writes results to `eval/results/`. Wired into CI via `.github/workflows/eval.yml` — every PR runs the eval automatically (GitHub Actions free tier).

## Repo layout

```
marketscopilot/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py
│   ├── models/schemas.py       # Pydantic request/response models
│   ├── security/                # auth, RBAC, DLP redaction, prompt-safety filter
│   ├── memory/                  # session memory + audit log (Postgres)
│   ├── retrieval/               # chunking, embeddings, BM25, vector store, hybrid search, ingest script
│   ├── agents/                  # LangGraph graph + individual agent nodes
│   ├── mcp_servers/              # example MCP tool server
│   └── observability/            # OpenTelemetry setup
├── eval/                        # retrieval + groundedness evaluation harness
├── data/sample_docs/             # sample market research / compliance docs for demo
├── docs/                        # architecture + path-to-production notes
└── tests/
```

## Documents in this repo worth reading first
- `docs/architecture.md` — full system diagram and component rationale
- `docs/path_to_production.md` — exactly what changes (and what doesn't) when this moves to paid managed infra
