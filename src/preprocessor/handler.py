"""
ExpertiseRAG – Preprocessor Lambda Handler

Triggered by:
  1. S3 ObjectCreated events on the raw/ prefix
  2. Direct invocation with { "repo_prefix": "raw/<repo>/", "bucket": "<bucket>" }
  3. Step Functions task input

For each invocation the handler:
  1. Lists all files under the raw/<repo>/ prefix (or processes the single key from S3 event)
  2. Downloads each file
  3. Dispatches to the appropriate extractor (Markdown, YAML, PlantUML, text)
  4. Chunks extracted text and generates Bedrock Titan V2 embeddings
  5. Writes chunks + embeddings to PostgreSQL pgvector
  6. Builds graph entities + edges via GraphBuilder and writes to Memgraph
  7. Writes derived artifacts to S3 under derived/<repo>/
     - <filename>.derived.json  : full extraction output
     - graph_entities.json      : all nodes for this repo pass
     - graph_edges.json         : all edges for this repo pass
     - expertise_signals.json   : aggregated signals
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from typing import Any
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

# With CodeUri: src/, Lambda adds /var/task/ to sys.path but the sibling
# modules live in /var/task/preprocessor/. Insert this directory so that
# bare module names (extractors, graph_builder, etc.) resolve correctly.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from extractors import extract
from graph_builder import build_graph_from_extractions, GraphBuilder
from models import DerivedArtifact, get_source_weight
from routing_analyzer import analyze_document

# Shared utilities – imported lazily so the module loads even without
# the DB drivers installed (unit-test environments)
def _get_shared_module(name: str):
    import importlib, sys, os
    shared_dir = os.path.join(os.path.dirname(__file__), "..", "shared")
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    return importlib.import_module(name)

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# ─────────────────────────────────────────────────────────────────────────────
# AWS clients (instantiated once per Lambda container for connection reuse)
# ─────────────────────────────────────────────────────────────────────────────
_S3 = boto3.client("s3")

ARTIFACTS_BUCKET = os.environ["ARTIFACTS_BUCKET"]
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw")
DERIVED_PREFIX = os.environ.get("DERIVED_PREFIX", "derived")

# Well-known repo files and their evidence weights
KNOWN_FILE_WEIGHTS: dict[str, float] = {
    "architecture.md": 1.0,
    "CLAUDE.md": 1.0,
    "repo-signals.yaml": 0.7,
    "README.md": 0.6,
    "readme.md": 0.6,
}


# ─────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_s3_text(bucket: str, key: str) -> str | None:
    """Download an S3 object and return its content as a UTF-8 string."""
    try:
        response = _S3.get_object(Bucket=bucket, Key=key)
        raw_bytes = response["Body"].read()
        return raw_bytes.decode("utf-8", errors="replace")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "NoSuchKey":
            logger.warning("Key not found: s3://%s/%s", bucket, key)
            return None
        raise


def _write_s3_json(bucket: str, key: str, data: Any) -> None:
    """Serialise data as JSON and write to S3."""
    body = json.dumps(data, indent=2, default=str)
    _S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="aws:kms",
    )
    logger.info("Wrote s3://%s/%s (%d bytes)", bucket, key, len(body))


def _list_s3_prefix(bucket: str, prefix: str) -> list[str]:
    """Return all object keys under a given S3 prefix."""
    keys: list[str] = []
    paginator = _S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not k.endswith("/"):  # skip "folder" markers
                keys.append(k)
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# Event parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_s3_event(event: dict) -> list[tuple[str, str]]:
    """
    Parse an S3 notification event and return list of (bucket, key) tuples.
    """
    results: list[tuple[str, str]] = []
    for record in event.get("Records", []):
        s3 = record.get("s3", {})
        bucket = s3.get("bucket", {}).get("name", "")
        key = unquote_plus(s3.get("object", {}).get("key", ""))
        if bucket and key:
            results.append((bucket, key))
    return results


def _extract_repo_prefix(s3_key: str) -> str:
    """
    Given raw/<repo>/something/file.md, return raw/<repo>/.
    Falls back to the first two path components.
    """
    parts = s3_key.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2]) + "/"
    return s3_key


def _derived_key(repo_prefix: str, source_key: str, suffix: str) -> str:
    """
    Map a raw/ key to a derived/ key.
    raw/myrepo/docs/arch.md → derived/myrepo/docs/arch.md.derived.json
    """
    relative = source_key[len(repo_prefix):]
    base = relative.rstrip("/")
    derived_repo = repo_prefix.replace(f"{RAW_PREFIX}/", "", 1).rstrip("/")
    return f"{DERIVED_PREFIX}/{derived_repo}/{base}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_file(
    bucket: str, key: str, repo_prefix: str
) -> dict[str, Any] | None:
    """
    Download, extract, and write a single file's derived artifact to S3.
    Returns the extraction metadata (for graph aggregation) or None on failure.
    """
    file_name = os.path.basename(key)
    logger.info("Processing file: s3://%s/%s", bucket, key)

    content = _read_s3_text(bucket, key)
    if content is None:
        return None

    # Determine evidence weight from filename or source type
    weight = KNOWN_FILE_WEIGHTS.get(file_name) or get_source_weight(file_name)

    extraction = extract(content, file_name, weight)
    file_type = (
        "plantuml" if file_name.endswith((".puml", ".plantuml", ".pu", ".wsd"))
        else "yaml" if file_name.endswith((".yaml", ".yml"))
        else "markdown" if file_name.endswith((".md", ".markdown"))
        else "text"
    )

    # Build the derived artifact envelope
    artifact = DerivedArtifact(
        source_bucket=bucket,
        source_key=key,
        repo_prefix=repo_prefix,
        file_type=file_type,
        extracted_text=extraction["extracted_text"],
        summary=extraction["summary"],
        expertise_signals=extraction["expertise_signals"],
        graph_nodes=[],   # populated in aggregate pass
        graph_edges=[],
        metadata={
            **extraction.get("metadata", {}),
            "source_weight": weight,
        },
    )

    # Write per-file derived artifact
    derived_key = _derived_key(repo_prefix, key, ".derived.json")
    _write_s3_json(bucket, derived_key, artifact.to_json())

    # Also write the clean extracted text as a .txt for Bedrock KB ingestion
    text_key = _derived_key(repo_prefix, key, ".extracted.txt")
    try:
        _S3.put_object(
            Bucket=bucket,
            Key=text_key,
            Body=(extraction["summary"] + "\n\n" + extraction["extracted_text"]).encode("utf-8"),
            ContentType="text/plain",
            ServerSideEncryption="aws:kms",
        )
        logger.info("Wrote extracted text: s3://%s/%s", bucket, text_key)
    except ClientError:
        logger.warning("Failed to write extracted text for %s", key, exc_info=True)

    # Routing analysis – classify document type and recommend chunking strategy
    routing_analysis = analyze_document(
        text=extraction["extracted_text"],
        filename=file_name,
        signals=extraction["expertise_signals"],
    )
    logger.info(
        "Routing analysis for %s: doc_type=%s chunking=%s",
        file_name,
        routing_analysis.doc_type,
        routing_analysis.chunking_strategy,
    )

    return {
        "source_file": file_name,
        "source_key": key,
        "file_type": file_type,
        "expertise_signals": extraction["expertise_signals"],
        "extracted_text": extraction["extracted_text"],
        "weight": weight,
        "routing_analysis": routing_analysis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chunk + embed + write to pgvector
# ─────────────────────────────────────────────────────────────────────────────

def _write_chunks_to_pgvector(
    extractions: list[dict],
    repo_name: str,
    errors: list | None = None,
) -> int:
    """
    Chunk each extracted document, generate Titan V2 embeddings, and insert
    into the pgvector chunks table.

    Returns the total number of chunks written (0 on failure).
    """
    try:
        chunker = _get_shared_module("chunker")
        embedder = _get_shared_module("embedder")
        db_clients = _get_shared_module("db_clients")
    except Exception as exc:
        logger.warning("Shared modules unavailable – skipping pgvector write: %s", exc)
        return 0

    conn = db_clients.get_pg_connection()
    total_written = 0

    for ext in extractions:
        source_file = ext.get("source_file", "")
        extracted_text = ext.get("extracted_text", "")
        routing_analysis = ext.get("routing_analysis")
        strategy = routing_analysis.chunking_strategy if routing_analysis else "fixed_512"
        doc_type = routing_analysis.doc_type if routing_analysis else "narrative"
        doc_id = f"{repo_name}/{source_file}"

        if not extracted_text.strip():
            continue

        chunks = chunker.chunk_text(extracted_text, strategy)
        if not chunks:
            continue

        # Generate embeddings for all chunk contents
        contents = [c.content for c in chunks]
        embeddings = embedder.embed_texts(contents)

        none_count = sum(1 for e in embeddings if e is None)
        if none_count > 0:
            logger.error(
                "Embedding failed for %d/%d chunks in %s – check Bedrock model access for %s",
                none_count, len(embeddings), source_file, "amazon.titan-embed-text-v2:0",
            )

        from psycopg2.extras import Json as _PgJson
        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is None:
                continue
            rows.append((
                doc_id,
                source_file,
                doc_type,
                strategy,
                chunk.content,
                embedding,
                chunk.parent_content,
                chunk.is_child,
                _PgJson({}),
            ))

        if not rows:
            continue

        try:
            with conn.cursor() as cur:
                # Delete existing chunks for this source_file to allow re-processing
                cur.execute(
                    "DELETE FROM chunks WHERE source_file = %s AND doc_id = %s",
                    (source_file, doc_id),
                )
                from psycopg2.extras import execute_values
                execute_values(
                    cur,
                    """
                    INSERT INTO chunks
                        (doc_id, source_file, doc_type, strategy, content,
                         embedding, parent_content, is_child, metadata)
                    VALUES %s
                    """,
                    [
                        (r[0], r[1], r[2], r[3], r[4],
                         r[5], r[6], r[7], r[8])
                        for r in rows
                    ],
                    template="(%s, %s, %s, %s, %s, %s::vector, %s, %s, %s)",
                )
            conn.commit()
            total_written += len(rows)
            logger.info(
                "pgvector: wrote %d chunks for %s (strategy=%s)",
                len(rows), source_file, strategy,
            )
        except Exception as exc:
            conn.rollback()
            msg = f"pgvector write failed for {source_file}: {exc}"
            logger.error(msg)
            if errors is not None:
                errors.append(msg)

    return total_written


# ─────────────────────────────────────────────────────────────────────────────
# Write graph to Memgraph
# ─────────────────────────────────────────────────────────────────────────────

def _write_graph_to_memgraph(
    extractions: list[dict],
    repo_name: str,
    errors: list | None = None,
) -> dict[str, int]:
    """
    Build the knowledge graph from extractions and write it directly to Memgraph.

    Returns {"nodes_written": int, "edges_written": int}.
    """
    try:
        db_clients = _get_shared_module("db_clients")
    except Exception as exc:
        logger.warning("db_clients unavailable – skipping Memgraph write: %s", exc)
        return {"nodes_written": 0, "edges_written": 0}

    from graph_builder import GraphBuilder
    import re as _re

    repo_id_label = f"repository_{_re.sub(r'[^a-z0-9]', '_', repo_name.lower()).strip('_')}"
    builder = GraphBuilder(
        person_name=os.environ.get("PERSON_NAME", "Developer"),
        repo_id=repo_id_label,
    )

    routing_analyses = {
        ext["source_file"]: ext["routing_analysis"]
        for ext in extractions
        if "routing_analysis" in ext
    }

    for ext in extractions:
        source_file = ext.get("source_file", "")
        builder.add_extraction(
            source_file=source_file,
            file_type=ext.get("file_type", "text"),
            expertise_signals=ext.get("expertise_signals", []),
            extracted_text=ext.get("extracted_text", ""),
            weight=float(ext.get("weight", 0.5)),
        )
        if source_file in routing_analyses:
            import re as _re2
            doc_nid = f"document_{_re2.sub(r'[^a-z0-9]', '_', source_file.lower()).strip('_')}"
            builder.add_routing_metadata(doc_nid, routing_analyses[source_file])

    try:
        driver = db_clients.get_memgraph_driver()
        result = builder.write_to_memgraph(driver)
        logger.info("Memgraph write result: %s", result)
        return result
    except Exception as exc:
        msg = f"Memgraph write failed: {exc}"
        logger.error(msg, exc_info=True)
        if errors is not None:
            errors.append(msg)
        return {"nodes_written": 0, "edges_written": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Main Lambda handler
# ─────────────────────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:
    """
    Entry point.

    Accepted event shapes:
      S3 notification : { "Records": [{ "s3": { "bucket": {..}, "object": {..} } }] }
      Direct invoke   : { "repo_prefix": "raw/myrepo/", "bucket": "...", "trigger_source": "..." }
      Step Functions  : same as direct invoke
    """
    logger.info("Event received: %s", json.dumps(event, default=str))

    bucket = ARTIFACTS_BUCKET
    keys_to_process: list[str] = []
    repo_prefix = ""

    # ── Parse event ───────────────────────────────────────────────────────────
    if "Records" in event:
        # S3 notification
        file_pairs = _parse_s3_event(event)
        for ev_bucket, key in file_pairs:
            bucket = ev_bucket
            keys_to_process.append(key)
            if not repo_prefix:
                repo_prefix = _extract_repo_prefix(key)
    else:
        # Direct / Step Functions invocation
        repo_prefix = event.get("repo_prefix", "")
        if event.get("bucket"):
            bucket = event["bucket"]
        if not repo_prefix:
            repo_prefix = f"{RAW_PREFIX}/"
        # List all files under the prefix
        keys_to_process = _list_s3_prefix(bucket, repo_prefix)
        logger.info(
            "Direct invocation: found %d files under s3://%s/%s",
            len(keys_to_process),
            bucket,
            repo_prefix,
        )

    if not keys_to_process:
        logger.warning("No files to process for prefix: %s", repo_prefix)
        return {"status": "no_files", "repo_prefix": repo_prefix, "files_processed": 0}

    # ── Process files ─────────────────────────────────────────────────────────
    extractions: list[dict[str, Any]] = []
    errors: list[str] = []

    for key in keys_to_process:
        try:
            result = _process_file(bucket, key, repo_prefix)
            if result:
                extractions.append(result)
        except Exception as exc:
            err_msg = f"Failed to process {key}: {exc}"
            logger.error(err_msg, exc_info=True)
            errors.append(err_msg)

    # ── Build aggregate graph ─────────────────────────────────────────────────
    repo_name = repo_prefix.strip("/").split("/")[-1] or "unknown_repo"
    graph_nodes, graph_edges = [], []

    if extractions:
        try:
            # Collect routing analyses keyed by source_file
            routing_analyses = {
                ext["source_file"]: ext["routing_analysis"]
                for ext in extractions
                if "routing_analysis" in ext
            }
            graph_nodes, graph_edges = build_graph_from_extractions(
                extractions,
                person_name=os.environ.get("PERSON_NAME", "Developer"),
                repo_prefix=repo_name,
                routing_analyses=routing_analyses,
            )
        except Exception as exc:
            logger.error("Graph build failed: %s", exc, exc_info=True)
            errors.append(f"Graph build failed: {exc}")

    # ── Aggregate expertise signals ────────────────────────────────────────────
    all_signals: list[dict[str, Any]] = []
    signal_index: dict[tuple[str, str], dict[str, Any]] = {}

    for ext in extractions:
        for sig in ext.get("expertise_signals", []):
            k = (sig.get("signal_type", ""), sig.get("value", ""))
            if k in signal_index:
                signal_index[k]["frequency"] = signal_index[k].get("frequency", 1) + sig.get("frequency", 1)
                signal_index[k]["weight"] = max(signal_index[k].get("weight", 0), sig.get("weight", 0))
            else:
                signal_index[k] = {**sig}

    all_signals = sorted(
        signal_index.values(),
        key=lambda s: (s.get("frequency", 1) * s.get("weight", 0.5)),
        reverse=True,
    )

    # ── Write chunks + embeddings to pgvector ─────────────────────────────────
    chunks_written = 0
    if extractions:
        try:
            chunks_written = _write_chunks_to_pgvector(extractions, repo_name, errors=errors)
            logger.info("pgvector: total %d chunks written", chunks_written)
        except Exception as exc:
            err_msg = f"pgvector write failed: {exc}"
            logger.error(err_msg, exc_info=True)
            errors.append(err_msg)

    # ── Write graph to Memgraph ────────────────────────────────────────────────
    memgraph_result: dict = {"nodes_written": 0, "edges_written": 0}
    if extractions:
        try:
            memgraph_result = _write_graph_to_memgraph(extractions, repo_name, errors=errors)
        except Exception as exc:
            err_msg = f"Memgraph write failed: {exc}"
            logger.error(err_msg, exc_info=True)
            errors.append(err_msg)

    # ── Write aggregate derived artifacts ─────────────────────────────────────
    derived_repo = repo_prefix.replace(f"{RAW_PREFIX}/", "", 1).rstrip("/")
    derived_base = f"{DERIVED_PREFIX}/{derived_repo}"

    _write_s3_json(bucket, f"{derived_base}/graph_entities.json", graph_nodes)
    _write_s3_json(bucket, f"{derived_base}/graph_edges.json", graph_edges)
    _write_s3_json(bucket, f"{derived_base}/expertise_signals.json", all_signals)
    # Summarise routing decisions for the manifest
    routing_summary: dict[str, int] = {}
    for ext in extractions:
        ra = ext.get("routing_analysis")
        if ra:
            key_label = f"{ra.doc_type}:{ra.chunking_strategy}"
            routing_summary[key_label] = routing_summary.get(key_label, 0) + 1

    _write_s3_json(bucket, f"{derived_base}/processing_manifest.json", {
        "repo_prefix": repo_prefix,
        "files_processed": len(extractions),
        "files_skipped": len(keys_to_process) - len(extractions),
        "graph_node_count": len(graph_nodes),
        "graph_edge_count": len(graph_edges),
        "signal_count": len(all_signals),
        "chunks_written": chunks_written,
        "memgraph_nodes_written": memgraph_result.get("nodes_written", 0),
        "memgraph_edges_written": memgraph_result.get("edges_written", 0),
        "routing_summary": routing_summary,
        "errors": errors,
    })

    logger.info(
        "Preprocessing complete: %d files, %d nodes, %d edges, %d signals, %d chunks",
        len(extractions),
        len(graph_nodes),
        len(graph_edges),
        len(all_signals),
        chunks_written,
    )

    return {
        "status": "success" if not errors else "partial",
        "repo_prefix": repo_prefix,
        "derived_prefix": derived_base,
        "files_processed": len(extractions),
        "graph_node_count": len(graph_nodes),
        "graph_edge_count": len(graph_edges),
        "signal_count": len(all_signals),
        "chunks_written": chunks_written,
        "memgraph_nodes_written": memgraph_result.get("nodes_written", 0),
        "memgraph_edges_written": memgraph_result.get("edges_written", 0),
        "errors": errors,
    }
