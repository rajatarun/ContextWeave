"""Bedrock batch inference for the bulk generation and judge calls.

Each input line is ``{"recordId", "modelInput"}``. ``modelInput`` is that
model's InvokeModel body: Anthropic Messages, Llama prompt, or Nova
``messages-v1``. The on-demand path stays in the caller.

The role ARN and the bucket come from ``batch.role_arn`` / ``batch.bucket``
in the config, or from the environment variables named there. Neither value
is hard-coded. A missing one stops the run.

A job is resumable. The local state file is named from the sha256 of the
input JSONL and is written before upload. A second run with the same input
polls the stored job ARN instead of creating another job. ``ConflictException``
looks the job up by name. Completed outputs are parsed into the same fields
the on-demand path records. A successful line with no token counts stops the
run. An error line with no usage stores zero tokens and ``usage_observed:
false``, which is the output saying nothing was metered, not an estimate.

The caller checks the spend cap with the projected batch cost before this
module creates a job.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from experiments.common import ProtocolError, now_iso, read_json, write_json

TERMINAL_OK = {"Completed", "PartiallyCompleted"}
TERMINAL_BAD = {"Failed", "Stopped", "Expired"}
RECORD_ID_LEN = 11


def model_family(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "anthropic" in mid or "claude" in mid:
        return "anthropic"
    if "llama" in mid:
        return "llama"
    if "nova" in mid:
        return "nova"
    raise ProtocolError(
        f"no batch modelInput format for {model_id!r}. "
        "Refusing to guess a request body."
    )


def record_id_for(*parts: str) -> str:
    """11 hex characters. Bedrock's documented recordId examples are this short.

    Callers must reject a collision rather than merge two rows.
    """
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:RECORD_ID_LEN]


def model_input(
    family: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """InvokeModel body for one JSONL line."""
    if family == "anthropic":
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system": system,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": user}],
            }],
        }
    if family == "llama":
        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return {
            "prompt": prompt,
            "max_gen_len": int(max_tokens),
            "temperature": float(temperature),
        }
    if family == "nova":
        return {
            "schemaVersion": "messages-v1",
            "system": [{"text": system}],
            "messages": [{
                "role": "user",
                "content": [{"text": user}],
            }],
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
            },
        }
    raise ProtocolError(f"no batch modelInput format for family {family!r}")


def build_jsonl(records: Sequence[dict[str, Any]]) -> str:
    """JSONL sorted by recordId. Each line has only recordId and modelInput."""
    lines = []
    seen: set[str] = set()
    ordered = sorted(records, key=lambda row: row["recordId"])
    for row in ordered:
        rid = row.get("recordId")
        if not isinstance(rid, str) or not rid:
            raise ProtocolError("batch record is missing recordId")
        if rid in seen:
            raise ProtocolError(f"duplicate batch recordId {rid}. Refusing to merge two rows.")
        seen.add(rid)
        if "modelInput" not in row or not isinstance(row["modelInput"], dict):
            raise ProtocolError(f"batch record {rid} is missing modelInput")
        lines.append(json.dumps(
            {"recordId": rid, "modelInput": row["modelInput"]},
            ensure_ascii=False,
            sort_keys=True,
        ))
    if not lines:
        raise ProtocolError("batch input has no records")
    return "\n".join(lines) + "\n"


def job_name_for(payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"cw{digest}"


def resolve_batch_location(cfg: dict[str, Any]) -> dict[str, str]:
    """Role ARN and bucket from config, then from the named environment variables."""
    batch = cfg.get("batch")
    if not isinstance(batch, dict):
        raise ProtocolError("config batch is missing. Refusing to guess a bucket or role.")
    role = str(batch.get("role_arn") or "").strip()
    bucket = str(batch.get("bucket") or "").strip()
    role_env = str(batch.get("role_arn_env") or "")
    bucket_env = str(batch.get("bucket_env") or "")
    if not role and role_env:
        role = os.environ.get(role_env, "").strip()
    if not bucket and bucket_env:
        bucket = os.environ.get(bucket_env, "").strip()
    if not role or not bucket:
        raise ProtocolError(
            "batch inference needs a role ARN and an S3 bucket. Set "
            f"{role_env or 'batch.role_arn_env'} and {bucket_env or 'batch.bucket_env'}, "
            "or set batch.role_arn and batch.bucket in the config. "
            "Refusing to guess either value."
        )
    prefix = str(batch.get("prefix") or "").strip().strip("/")
    if not prefix:
        raise ProtocolError("batch.prefix is empty. Refusing to guess an S3 prefix.")
    return {"role_arn": role, "bucket": bucket, "prefix": prefix}


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", type(exc).__name__))
    return type(exc).__name__


def _error_message(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Message", exc))
    return str(exc)


def _s3_body(resp: dict[str, Any]) -> bytes:
    body = resp["Body"]
    if hasattr(body, "read"):
        data = body.read()
    else:
        data = body
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def _list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for item in page.get("Contents") or []:
            key = item.get("Key")
            if key:
                keys.append(key)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            raise ProtocolError(
                f"s3 list of s3://{bucket}/{prefix} was truncated and returned no continuation token"
            )
    return keys


def upload_input(s3: Any, bucket: str, key: str, payload: str) -> str:
    s3.put_object(Bucket=bucket, Key=key, Body=payload.encode("utf-8"))
    return f"s3://{bucket}/{key}"


def create_or_find_job(
    bedrock: Any,
    *,
    job_name: str,
    model_id: str,
    role_arn: str,
    input_uri: str,
    output_uri: str,
    timeout_hours: int,
) -> str:
    try:
        resp = bedrock.create_model_invocation_job(
            jobName=job_name,
            roleArn=role_arn,
            modelId=model_id,
            clientRequestToken=job_name,
            modelInvocationType="InvokeModel",
            inputDataConfig={"s3InputDataConfig": {
                "s3Uri": input_uri,
                "s3InputFormat": "JSONL",
            }},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}},
            timeoutDurationInHours=int(timeout_hours),
        )
    except Exception as exc:
        if _error_code(exc) != "ConflictException":
            raise ProtocolError(
                f"create_model_invocation_job failed ({_error_code(exc)}): {_error_message(exc)}"
            ) from exc
        return _job_arn_by_name(bedrock, job_name)
    arn = resp.get("jobArn")
    if not arn:
        raise ProtocolError("create_model_invocation_job returned no jobArn")
    return str(arn)


def _job_arn_by_name(bedrock: Any, job_name: str) -> str:
    resp = bedrock.list_model_invocation_jobs(nameContains=job_name, maxResults=100)
    matches = [
        item for item in resp.get("invocationJobSummaries") or []
        if item.get("jobName") == job_name and item.get("jobArn")
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"batch job name {job_name} already exists and list_model_invocation_jobs "
            f"returned {len(matches)} exact matches. Refusing to guess which job to resume."
        )
    return str(matches[0]["jobArn"])


def poll_job(
    bedrock: Any,
    job_arn: str,
    sleep: Callable[[float], None],
    poll_seconds: float,
    max_polls: int,
) -> dict[str, Any]:
    polls = 0
    while True:
        job = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        status = str(job.get("status") or "")
        if status in TERMINAL_OK or status in TERMINAL_BAD:
            return job
        if not status:
            raise ProtocolError(f"batch job {job_arn} returned no status")
        polls += 1
        if polls >= max_polls:
            raise ProtocolError(
                f"batch job {job_arn} still {status} after {max_polls} polls. "
                "Refusing to treat it as finished."
            )
        sleep(poll_seconds)


def download_output_lines(s3: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    keys = [key for key in _list_keys(s3, bucket, prefix) if key.endswith(".jsonl.out")]
    if not keys:
        raise ProtocolError(
            f"batch output s3://{bucket}/{prefix} has no .jsonl.out object. "
            "Refusing to invent rows."
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in sorted(keys):
        raw = _s3_body(s3.get_object(Bucket=bucket, Key=key))
        for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"s3://{bucket}/{key}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict) or not obj.get("recordId"):
                raise ProtocolError(f"s3://{bucket}/{key}:{line_no}: missing recordId")
            rid = str(obj["recordId"])
            if rid in seen:
                raise ProtocolError(f"batch output repeats recordId {rid}")
            seen.add(rid)
            rows.append(obj)
    return rows


def parse_model_output(family: str, output: dict[str, Any]) -> dict[str, Any]:
    """Text, token counts, and stop reason from one model's InvokeModel response."""
    if family == "anthropic":
        parts = []
        for block in output.get("content") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        return {
            "text": "".join(parts),
            "input_tokens": _token(usage, "input_tokens", "inputTokens"),
            "output_tokens": _token(usage, "output_tokens", "outputTokens"),
            "stop_reason": output.get("stop_reason") or output.get("stopReason"),
        }
    if family == "llama":
        text = output.get("generation")
        return {
            "text": text if isinstance(text, str) else "",
            "input_tokens": _token(output, "prompt_token_count"),
            "output_tokens": _token(output, "generation_token_count"),
            "stop_reason": output.get("stop_reason") or output.get("stopReason"),
        }
    if family == "nova":
        parts = []
        message = (output.get("output") or {}).get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
        return {
            "text": "".join(parts),
            "input_tokens": _token(usage, "inputTokens", "input_tokens"),
            "output_tokens": _token(usage, "outputTokens", "output_tokens"),
            "stop_reason": output.get("stopReason") or output.get("stop_reason"),
        }
    raise ProtocolError(f"no batch output parser for family {family!r}")


def _token(obj: dict[str, Any], *names: str) -> int | None:
    for name in names:
        if name in obj and obj[name] is not None:
            value = obj[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProtocolError(f"batch usage field {name} is {value!r}")
            return int(value)
    return None


def parsed_output_rows(family: str, lines: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map recordId to text, tokens, stop reason, and error.

    A success line without token counts stops the run. An error line stores
    the error and, when the line has no usage, zero tokens with
    ``usage_observed`` false.
    """
    out: dict[str, dict[str, Any]] = {}
    for line in lines:
        rid = str(line["recordId"])
        if line.get("error"):
            err = line["error"]
            message = err.get("errorMessage") if isinstance(err, dict) else str(err)
            code = err.get("errorCode") if isinstance(err, dict) else "error"
            parsed = {"text": "", "input_tokens": None, "output_tokens": None, "stop_reason": None}
            if isinstance(line.get("modelOutput"), dict):
                parsed = parse_model_output(family, line["modelOutput"])
            observed = parsed["input_tokens"] is not None and parsed["output_tokens"] is not None
            out[rid] = {
                "text": parsed["text"] or "",
                "input_tokens": parsed["input_tokens"] if observed else 0,
                "output_tokens": parsed["output_tokens"] if observed else 0,
                "stop_reason": parsed["stop_reason"],
                "usage_observed": observed,
                "error": {"type": str(code), "message": str(message)},
            }
            continue
        model_output = line.get("modelOutput")
        if not isinstance(model_output, dict):
            raise ProtocolError(f"batch record {rid} has no modelOutput and no error")
        parsed = parse_model_output(family, model_output)
        if parsed["input_tokens"] is None or parsed["output_tokens"] is None:
            raise ProtocolError(
                f"batch record {rid} succeeded and reported no token counts. "
                "Refusing to invent a cost."
            )
        out[rid] = {
            "text": parsed["text"],
            "input_tokens": parsed["input_tokens"],
            "output_tokens": parsed["output_tokens"],
            "stop_reason": parsed["stop_reason"] if isinstance(parsed["stop_reason"], str) else None,
            "usage_observed": True,
            "error": None,
        }
    return out


def state_path(directory: Path, job_name: str) -> Path:
    return directory / f"{job_name}.json"


def write_state(path: Path, body: dict[str, Any]) -> None:
    write_json(path, body)


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        raise ProtocolError(f"{path}: batch state is not an object")
    return data


def run_job(
    *,
    cfg: dict[str, Any],
    model_id: str,
    records: Sequence[dict[str, Any]],
    state_dir: Path,
    s3: Any,
    bedrock: Any,
    sleep: Callable[[float], None],
    seed: int,
    max_polls: int | None = None,
) -> dict[str, Any]:
    """Upload, create or resume, poll, and parse. Returns parsed rows and the state.

    ``records`` items are ``recordId``, ``modelInput``, and ``meta`` (kept in
    the state file, not uploaded).
    """
    family = model_family(model_id)
    payload = build_jsonl(records)
    name = job_name_for(payload)
    location = resolve_batch_location(cfg)
    path = state_path(state_dir, name)
    existing = load_state(path)
    if existing is not None and existing.get("input_sha256") not in (None, hashlib.sha256(payload.encode("utf-8")).hexdigest()):
        raise ProtocolError(f"{path}: stored input hash does not match this submission")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    input_key = f"{location['prefix']}/jobs/{name}/input.jsonl"
    output_prefix = f"{location['prefix']}/jobs/{name}/out/"
    meta = {
        row["recordId"]: row.get("meta") or {}
        for row in records
    }
    state = existing or {
        "job_name": name,
        "job_arn": None,
        "model_id": model_id,
        "family": family,
        "input_sha256": digest,
        "seed": seed,
        "created_at": now_iso(),
        "s3_input_uri": f"s3://{location['bucket']}/{input_key}",
        "s3_output_prefix": output_prefix,
        "bucket": location["bucket"],
        "records": meta,
        "pricing": "batch",
    }
    state["seed"] = seed
    state["records"] = meta
    write_state(path, state)
    upload_input(s3, location["bucket"], input_key, payload)
    if not state.get("job_arn"):
        timeout = int(cfg["batch"]["timeout_hours"])
        arn = create_or_find_job(
            bedrock,
            job_name=name,
            model_id=model_id,
            role_arn=location["role_arn"],
            input_uri=state["s3_input_uri"],
            output_uri=f"s3://{location['bucket']}/{output_prefix}",
            timeout_hours=timeout,
        )
        state["job_arn"] = arn
        write_state(path, state)
    poll_seconds = float(cfg["batch"]["poll_seconds"])
    if max_polls is None:
        window = float(cfg["batch"]["timeout_hours"]) * 3600.0
        max_polls = max(1, int(window / poll_seconds) + 1)
    job = poll_job(bedrock, state["job_arn"], sleep, poll_seconds, max_polls)
    status = str(job.get("status") or "")
    state["status"] = status
    write_state(path, state)
    if status in TERMINAL_BAD:
        message = job.get("message") or job.get("statusMessage") or status
        raise ProtocolError(f"batch job {state['job_arn']} ended {status}: {message}")
    lines = download_output_lines(s3, location["bucket"], output_prefix)
    parsed = parsed_output_rows(family, lines)
    missing = [rid for rid in meta if rid not in parsed]
    if missing:
        raise ProtocolError(
            f"batch job {state['job_arn']} returned no output for recordId {missing[0]}. "
            "Refusing to invent that row."
        )
    return {"job_name": name, "job_arn": state["job_arn"], "status": status, "state_path": str(path), "rows": parsed}
