"""The health endpoint. Authenticated, uncached, and it does not diagnose.

Three routes, all behind the platform's SIWE authorizer -- which is the first
authorizer on this API at all. Every expertise route is open, and that is
tolerable for architecture notes; it is not for a medical record.

    POST   /health/query               ask a question of your own record
    GET    /health/documents           what is in the store (never its content)
    DELETE /health/documents/{docId}   remove one, and confirm how much went

**Nothing here is cached.** The expertise path writes every answer to
`query_cache` for seven days, keyed by question embedding. An answer drawn from
a medical record would sit in a second table for a week, outliving a delete of
the record it came from. `write_cache` is not called and `cacheHit` is always
false.

**It answers about the record; it does not practise medicine.** The synthesis
prompt forbids diagnosis, treatment and dosage, and says so in the response so
a caller cannot present the answer as clinical advice by omission.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..shared import health_db

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

DISCLAIMER = (
    "This answers from your own uploaded records and is not medical advice, "
    "a diagnosis, or a reason to change any treatment. Take it to a clinician."
)

SYSTEM_PROMPT = (
    "You answer questions about the person's own health records, which are "
    "provided to you as excerpts.\n"
    "Rules, in order of priority:\n"
    "1. Never diagnose, never name a condition as the cause of anything, never "
    "suggest starting, stopping or changing a medicine or a dose.\n"
    "2. Answer only from the excerpts. If they do not contain the answer, say "
    "so plainly. An invented value in a medical answer is the worst possible "
    "failure here, and a confident wrong number is indistinguishable from a "
    "right one to the person reading it.\n"
    "3. Quote the figure and say which document it came from, so the person can "
    "check it against the original.\n"
    "4. If the excerpts suggest something that should be seen urgently, say "
    "that first, before answering the question that was asked."
)


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _synthesize(question: str, excerpts: list[dict], *, client=None) -> dict:
    if client is None:
        import boto3
        client = boto3.client("bedrock-runtime")

    context = "\n\n".join(
        f"[{i + 1}] from {e['sourceKey']}:\n{e['content']}" for i, e in enumerate(excerpts)
    )
    response = client.converse(
        modelId=os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0"),
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [
            {"text": f"EXCERPTS FROM YOUR RECORDS:\n{context}\n\nQUESTION: {question}"}
        ]}],
        inferenceConfig={"maxTokens": 1200, "temperature": 0.2},
    )
    blocks = (((response or {}).get("output") or {}).get("message") or {}).get("content") or []
    return {"answer": "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()}


def query(body: dict, *, bedrock=None, embed=None) -> dict:
    question = str(body.get("question") or "").strip()
    if not question:
        return _response(400, {"error": "question is required"})

    # Injected, like `bedrock` and `s3` elsewhere here, and imported lazily
    # when it is not: listing and deleting a record must not depend on the
    # embedding stack being importable, and a test must not have to import
    # Bedrock and the observability wrapper to check a disclaimer.
    if embed is None:
        from ..shared.embedder import embed_text as embed

    embedding = embed(question)
    if embedding is None:
        return _response(503, {"error": "the question could not be embedded", "disclaimer": DISCLAIMER})

    top_k = int(body.get("topK") or 6)
    excerpts = health_db.search(embedding, top_k=top_k)
    if not excerpts:
        # Distinct from an answer. "Nothing in your records covers this" is a
        # real result, and letting a model answer anyway from general knowledge
        # would present its recollection as the person's own chart.
        return _response(200, {
            "answer": "",
            "found": False,
            "note": "No excerpt in your uploaded records is relevant to that question.",
            "sources": [],
            "cacheHit": False,
            "disclaimer": DISCLAIMER,
        })

    result = _synthesize(question, excerpts, client=bedrock)
    return _response(200, {
        "answer": result["answer"],
        "found": True,
        # Which documents, never the excerpt text: the caller can look them up,
        # and a response body is a thing that gets logged and forwarded.
        "sources": sorted({e["sourceKey"] for e in excerpts}),
        "cacheHit": False,
        "disclaimer": DISCLAIMER,
    })


def lambda_handler(event: dict, _context: Any = None) -> dict:
    method = ((event.get("requestContext") or {}).get("http") or {}).get("method", "GET").upper()
    path = event.get("rawPath") or ""

    try:
        if method == "POST" and path.endswith("/health/query"):
            raw = event.get("body") or "{}"
            return query(json.loads(raw) if isinstance(raw, str) else raw)

        if method == "GET" and path.endswith("/health/documents"):
            return _response(200, {"documents": health_db.list_documents()})

        if method == "DELETE" and "/health/documents/" in path:
            doc_id = path.rsplit("/", 1)[-1]
            if not doc_id:
                return _response(400, {"error": "docId is required"})
            removed = health_db.delete_document(doc_id)
            # 404 on nothing removed, so "it is gone" and "it was never here"
            # are different answers. Deleting a record you cannot confirm is
            # gone is not deleting it.
            if removed == 0:
                return _response(404, {"docId": doc_id, "deleted": 0,
                                       "note": "no such document in the health store"})
            return _response(200, {"docId": doc_id, "deleted": removed})

        return _response(404, {"error": f"no route for {method} {path}"})
    except Exception:
        # The message is never echoed: a psycopg2 error can carry a row, and a
        # row here is a medical record.
        logger.exception("health api call failed method=%s path=%s", method, path)
        return _response(500, {"error": "the request could not be completed"})
