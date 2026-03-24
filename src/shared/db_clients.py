"""
Database client factories for ExpertiseRAG.

Provides singleton connection factories for:
  - Memgraph  : openCypher graph database accessed via Neo4j bolt driver
  - PostgreSQL : pgvector chunk store accessed via psycopg2

Connections are cached at module level and reused across warm Lambda invocations.
Credentials are read from AWS Secrets Manager on first access.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Secrets Manager helpers
# ─────────────────────────────────────────────────────────────────────────────

_secrets_cache: dict[str, dict] = {}


def _get_secret(secret_arn: str) -> dict:
    """Fetch and cache a Secrets Manager secret as a parsed dict."""
    if secret_arn in _secrets_cache:
        return _secrets_cache[secret_arn]

    import boto3
    from botocore.config import Config
    client = boto3.client(
        "secretsmanager",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(connect_timeout=5, retries={"max_attempts": 2}),
    )
    response = client.get_secret_value(SecretId=secret_arn)
    secret = json.loads(response["SecretString"])
    _secrets_cache[secret_arn] = secret
    return secret


# ─────────────────────────────────────────────────────────────────────────────
# Memgraph (Neo4j bolt driver)
# ─────────────────────────────────────────────────────────────────────────────

_memgraph_driver: Any = None


def get_memgraph_driver() -> Any:
    """
    Return a cached Neo4j bolt driver connected to Memgraph.

    Reads connection details from MEMGRAPH_SECRET_ARN (host, port) or
    falls back to MEMGRAPH_HOST / MEMGRAPH_PORT environment variables.

    Memgraph Community Edition runs without auth by default.
    """
    global _memgraph_driver

    if _memgraph_driver is not None:
        # Verify the driver is still live before returning
        try:
            _memgraph_driver.verify_connectivity()
            return _memgraph_driver
        except Exception:
            logger.warning("Memgraph driver connectivity check failed – reconnecting")
            _memgraph_driver = None

    from neo4j import GraphDatabase

    secret_arn = os.environ.get("MEMGRAPH_SECRET_ARN", "")
    if secret_arn:
        secret = _get_secret(secret_arn)
        host = secret.get("host", "localhost")
        port = int(secret.get("port", 7687))
    else:
        host = os.environ.get("MEMGRAPH_HOST", "localhost")
        port = int(os.environ.get("MEMGRAPH_PORT", "7687"))

    uri = f"bolt://{host}:{port}"
    logger.info("Connecting to Memgraph at %s", uri)
    _memgraph_driver = GraphDatabase.driver(
        uri,
        auth=("", ""),
        connection_timeout=5,
        max_connection_lifetime=300,
    )
    return _memgraph_driver


def run_graph_query(query: str, parameters: dict | None = None) -> list[dict]:
    """
    Execute an openCypher query against Memgraph; returns [] on any error.

    Parameters are passed as keyword args to driver.session().run().
    """
    try:
        driver = get_memgraph_driver()
        with driver.session() as session:
            result = session.run(query, **(parameters or {}))
            return [dict(record) for record in result]
    except Exception as exc:
        logger.warning("Memgraph query error: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL / pgvector (psycopg2)
# ─────────────────────────────────────────────────────────────────────────────

_pg_conn: Any = None


def get_pg_connection() -> Any:
    """
    Return a cached psycopg2 connection to PostgreSQL.

    Reads connection details from POSTGRES_SECRET_ARN or falls back to
    individual POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB /
    POSTGRES_USER / POSTGRES_PASSWORD environment variables.
    """
    global _pg_conn

    if _pg_conn is not None:
        try:
            # Lightweight ping to verify connection is still alive
            _pg_conn.cursor().execute("SELECT 1")
            return _pg_conn
        except Exception:
            logger.warning("PostgreSQL connection lost – reconnecting")
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None

    import psycopg2

    secret_arn = os.environ.get("POSTGRES_SECRET_ARN", "")
    if secret_arn:
        secret = _get_secret(secret_arn)
        host = secret.get("host", "localhost")
        port = int(secret.get("port", 5432))
        dbname = secret.get("dbname", secret.get("db", "expertiserag"))
        user = secret.get("username", secret.get("user", "expertiserag"))
        password = secret.get("password", "")
    else:
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = int(os.environ.get("POSTGRES_PORT", "5432"))
        dbname = os.environ.get("POSTGRES_DB", "expertiserag")
        user = os.environ.get("POSTGRES_USER", "expertiserag")
        password = os.environ.get("POSTGRES_PASSWORD", "")

    logger.info("Connecting to PostgreSQL at %s:%d/%s", host, port, dbname)
    _pg_conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=10,
    )
    _pg_conn.autocommit = False
    return _pg_conn


def init_pgvector_schema() -> None:
    """
    Idempotently create the pgvector extension and chunks table.
    Safe to call on every cold start; uses IF NOT EXISTS throughout.
    """
    conn = get_pg_connection()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                doc_id         TEXT NOT NULL,
                source_file    TEXT NOT NULL,
                doc_type       TEXT,
                strategy       TEXT,
                content        TEXT NOT NULL,
                embedding      vector(1024),
                parent_content TEXT,
                is_child       BOOLEAN DEFAULT FALSE,
                metadata       JSONB DEFAULT '{}',
                created_at     TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_doc_type_idx ON chunks (doc_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_source_file_idx ON chunks (source_file)")

        # ── Query cache (CAG) ────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                question_embedding  vector(1024)  NOT NULL,
                response_json       JSONB         NOT NULL,
                question_type       TEXT,
                hit_count           INTEGER       DEFAULT 0,
                created_at          TIMESTAMPTZ   DEFAULT NOW(),
                expires_at          TIMESTAMPTZ   NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS qcache_embedding_idx
                ON query_cache USING ivfflat (question_embedding vector_cosine_ops)
                WITH (lists = 50)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS qcache_expires_idx
                ON query_cache (expires_at)
        """)
    conn.commit()
    logger.info("pgvector schema initialised")
