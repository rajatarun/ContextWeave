"""Lexical grounding, NLI grounding, and the sampled judge.

Lexical support and the judge prompt are the deployed ones
(``verified_reward.lexical_support``, ``verified_reward.JUDGE_PROMPT``).
The claim is supported when some passage reaches tau = 0.6. Numbers in the
claim must occur verbatim in the passage. Grounding is missing when there are
no passages or no checkable claims.

The judge runs on a deterministic 5% sample of question ids: the first 8 hex
characters of SHA-256(qid) as an integer, divided by 2**32, included when that
value is below the rate (``verified_reward.should_judge``). The sample does
not depend on ``--seed``. Every arm of a sampled question is judged. A question
that is out of the sample is written with reason ``not_sampled``.

The judge model must differ from the generator model unless ``--allow-same-judge``
is passed. That flag is stored on the artifact either way.

NLI is ``cross-encoder/nli-deberta-v3-small``. The score is the entailment
class probability after softmax, the same reading as
``verified_reward_bench.nli_verifier``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))
sys.path.insert(0, str(_ROOT / "scripts"))

import verified_reward as V  # noqa: E402

from experiments.common import ProtocolError
from experiments.generate_stage import (
    FATAL_CODES, THROTTLE_CODES, _error_code, _error_message,
)

JUDGE_PROMPT = V.JUDGE_PROMPT


def judge_sampled(qid: str, rate: float) -> bool:
    return V.should_judge(qid, rate)


def lexical_value(answer: str, passages: Sequence[str], threshold: float, weight: float) -> tuple[float | None, str | None, dict[str, Any]]:
    signal = V.grounding_signal(answer, passages, verifier=V.lexical_support, threshold=threshold, weight=weight)
    reason = signal.detail.get("reason") if signal.value is None else None
    return signal.value, reason, signal.detail


def nli_value(answer: str, passages: Sequence[str], verifier: V.Verifier, threshold: float, weight: float) -> tuple[float | None, str | None, dict[str, Any]]:
    signal = V.grounding_signal(answer, passages, verifier=verifier, threshold=threshold, weight=weight)
    reason = signal.detail.get("reason") if signal.value is None else None
    return signal.value, reason, signal.detail


def load_nli(model_name: str) -> tuple[V.Verifier, dict[str, Any]]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ProtocolError(
            "sentence-transformers is not installed; pip install -r experiments/requirements.txt"
        ) from exc
    model = CrossEncoder(model_name, device="cpu")
    labels = [str(l).lower() for l in getattr(model.config, "id2label", {}).values()] or [
        "contradiction", "entailment", "neutral",
    ]
    if "entailment" not in labels:
        raise ProtocolError(f"NLI model {model_name} has no entailment label: {labels}")
    ent = labels.index("entailment")
    revision = None
    try:
        revision = model.model.config._commit_hash
    except Exception:
        revision = None

    def verify(claim: str, passage: str) -> float:
        probs = model.predict([(passage, claim)], apply_softmax=True)[0]
        return float(probs[ent])

    return verify, {"nli_model": model_name, "revision": revision, "labels": labels}


def check_judge_model(judge_model: str, generator_model: str, allow_same: bool) -> None:
    if judge_model == generator_model and not allow_same:
        raise ProtocolError(
            f"judge model {judge_model} equals the generator model. "
            "Pass --allow-same-judge to run anyway. Refusing to grade a model with itself."
        )


def judge_one(
    client: Any,
    model_id: str,
    question: str,
    answer: str,
    passages: Sequence[str],
    max_tokens: int,
    *,
    sleep: Callable[[float], None],
    max_attempts: int = 6,
) -> tuple[float | None, str | None, str]:
    evidence = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1) if p)
    if not evidence:
        return None, "no_passages", ""
    prompt = JUDGE_PROMPT.format(question=question, evidence=evidence[:12000], answer=answer)
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
            raw = ""
            for block in resp.get("output", {}).get("message", {}).get("content", []):
                if isinstance(block, dict) and "text" in block:
                    raw += block["text"]
            score = V.parse_judge_score(raw)
            if score is None:
                return None, "unparseable", raw
            return score, None, raw
        except Exception as exc:
            code = _error_code(exc)
            if code in FATAL_CODES or "AccessDenied" in code:
                raise ProtocolError(f"fatal judge error ({code}): {_error_message(exc)}") from exc
            if code in THROTTLE_CODES and attempt + 1 < max_attempts:
                sleep(delay)
                delay *= 2
                continue
            if code == "ValidationException":
                return None, "validation_exception", _error_message(exc)
            if code in THROTTLE_CODES:
                raise ProtocolError(f"judge still throttled after {max_attempts} attempts: {code}") from exc
            raise ProtocolError(f"unhandled judge error ({code}): {_error_message(exc)}") from exc
    raise ProtocolError("judge retries exhausted")


def signal_row(base: dict[str, Any], value: float | None, reason: str | None, **extra: Any) -> dict[str, Any]:
    row = {
        "qid": base["qid"],
        "dataset": base["dataset"],
        "arm": base["arm"],
        "question_type": base["question_type"],
        "value": value,
        "reason": reason,
    }
    row.update(extra)
    return row
