from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM serving (OpenAI-compatible — Ollama in dev, vLLM in prod)
    llm_base_url: str = "http://ollama:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    llm_api_key: str = "not-needed-for-local-serving"

    # Embeddings
    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # Vector store
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "markets_research"

    # Postgres
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "marketscopilot"
    postgres_user: str = "marketscopilot"
    postgres_password: str = "change-me-locally"

    # MinIO / S3-compatible storage
    s3_endpoint_url: str = "http://minio:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "marketscopilot-docs"


settings = Settings()
