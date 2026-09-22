"""The health store: a different database, reached by a different role.

Separate from `db_clients` on purpose, and the separation is the feature. Four
properties of the expertise corpus make it the wrong place for a medical
record, and each is answered here rather than by convention:

1. **`POST /query-expertise` has no authorizer.** Every `Events:` block on the
   expertise API is unauthenticated, which is fine for architecture notes and
   not for a discharge summary. The health routes sit behind the platform's
   SIWE authorizer.

2. **Retrieval has no namespace.** `retriever.py` searches one undifferentiated
   `chunks` table -- its only filters are the query's own and
   `embedding IS NOT NULL`. So a LinkedIn post generator asking about work
   under pressure could retrieve a chunk of a medical record and nothing in the
   pipeline would notice. Health chunks live in their own **database**, reached
   with their own credential; the expertise role has no CONNECT on it, so that
   retrieval cannot reach these rows even by mistake.

3. **There was no delete.** The only per-document removal in the expertise path
   is re-ingestion replacing a file's own chunks, and the graph's only removal
   is `MATCH (n) DETACH DELETE n` -- everything. `delete_document` here removes
   one document's rows, and it exists before the first record does, because
   retrofitting deletion is the part nobody gets to later.

4. **Health documents never enter the knowledge graph.** Memgraph and Neptune
   are shared with the expertise path and have no per-document delete, so a
   graph write would be the one thing this module could not take back. The
   trade is real: no entity expansion for health questions, only vector
   retrieval over the record itself.

The instance is shared with the expertise database; the database and the role
are not. A separate RDS instance would be stronger and costs another instance,
so this is named as a decision rather than left implicit.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Its own table name as well as its own database, so a mis-set connection
# string fails loudly on a missing relation rather than silently reading or
# writing the expertise corpus.
HEALTH_CHUNKS_TABLE = "health_chunks"
EMBED_DIM = 1024

_conn = None


def _secret(secret_arn: str) -> dict:
    import boto3

    client = boto3.client("secretsmanager")
    raw = client.get_secret_value(SecretId=secret_arn)["SecretString"]
    return json.loads(raw)


def get_connection() -> Any:
    """A cached connection to the health database.

    Deliberately not `db_clients.get_pg_connection`: that one caches a
    connection opened with the expertise credential, and sharing it would undo
    the isolation this module exists for.
    """
    global _conn

    if _conn is not None:
        try:
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            logger.warning("health postgres connection lost - reconnecting")
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

    # Checked before the driver import: with the import first, a missing
    # secret surfaces as ModuleNotFoundError in any environment without
    # psycopg2, which sends the reader to the wrong problem entirely.
    secret_arn = os.environ.get("HEALTH_POSTGRES_SECRET_ARN", "")
    if not secret_arn:
        # Refused rather than defaulted. A fallback to the expertise
        # credentials would put medical records in the corpus the LinkedIn
        # writer reads, which is the whole failure this module prevents.
        raise RuntimeError(
            "HEALTH_POSTGRES_SECRET_ARN is not set. The health store has its own "
            "database and its own role; there is no fallback to the expertise "
            "connection, because that corpus is read by the content teams."
        )

    import psycopg2

    secret = _secret(secret_arn)
    dbname = secret.get("dbname") or secret.get("db") or ""
    if not dbname:
        raise RuntimeError("health secret carries no dbname")

    _conn = psycopg2.connect(
        host=secret.get("host", ""),
        port=int(secret.get("port", 5432)),
        dbname=dbname,
        user=secret.get("username") or secret.get("user", ""),
        password=secret.get("password", ""),
        connect_timeout=10,
    )
    _conn.autocommit = False
    logger.info("connected to the health database %s", dbname)
    return _conn


def init_schema() -> None:
    """Idempotent. Safe on every cold start."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {HEALTH_CHUNKS_TABLE} (
                id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                doc_id       TEXT NOT NULL,
                source_key   TEXT NOT NULL,
                content      TEXT NOT NULL,
                embedding    vector({EMBED_DIM}),
                metadata     JSONB DEFAULT '{{}}',
                created_at   TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS health_chunks_embedding_idx
                ON {HEALTH_CHUNKS_TABLE} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
        """)
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS health_chunks_doc_id_idx "
            f"ON {HEALTH_CHUNKS_TABLE} (doc_id)"
        )
    conn.commit()


def replace_document(doc_id: str, source_key: str, rows: list[tuple[str, list[float]]],
                     metadata: Optional[dict] = None) -> int:
    """Write one document's chunks, replacing any earlier version of it.

    Delete-then-insert in one transaction: a re-ingest that inserted first
    would leave both versions retrievable, and a health record answering from
    a superseded copy is worse than one that answers from nothing.
    """
    conn = get_connection()
    payload = json.dumps(metadata or {})
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {HEALTH_CHUNKS_TABLE} WHERE doc_id = %s", (doc_id,))
        for content, embedding in rows:
            cur.execute(
                f"INSERT INTO {HEALTH_CHUNKS_TABLE} "
                f"(doc_id, source_key, content, embedding, metadata) "
                f"VALUES (%s, %s, %s, %s, %s)",
                (doc_id, source_key, content, embedding, payload),
            )
    conn.commit()
    return len(rows)


def delete_document(doc_id: str) -> int:
    """Remove one document. Returns the number of chunks removed.

    The count is returned rather than discarded so a caller can tell "deleted"
    from "there was nothing there" -- an interface that answers the same way to
    both is one you cannot use to confirm a record is gone.
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {HEALTH_CHUNKS_TABLE} WHERE doc_id = %s", (doc_id,))
        removed = cur.rowcount
    conn.commit()
    return removed


def list_documents() -> list[dict]:
    """What is in the store, without returning any of its content."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT doc_id, source_key, COUNT(*), MAX(created_at) "
            f"FROM {HEALTH_CHUNKS_TABLE} GROUP BY doc_id, source_key ORDER BY 4 DESC"
        )
        return [
            {"docId": r[0], "sourceKey": r[1], "chunks": r[2],
             "ingestedAt": r[3].isoformat() if r[3] else ""}
            for r in cur.fetchall()
        ]


def search(embedding: list[float], top_k: int = 6) -> list[dict]:
    """Cosine search over the health store only."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT content, doc_id, source_key, 1 - (embedding <=> %s::vector) "
            f"FROM {HEALTH_CHUNKS_TABLE} WHERE embedding IS NOT NULL "
            f"ORDER BY embedding <=> %s::vector LIMIT %s",
            (embedding, embedding, int(top_k)),
        )
        return [
            {"content": r[0], "docId": r[1], "sourceKey": r[2], "score": float(r[3])}
            for r in cur.fetchall()
        ]
