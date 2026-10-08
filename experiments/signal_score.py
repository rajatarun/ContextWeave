"""Claim grounding, seeded samples, self percentile, and the holdout logistic.

Lexical and NLI score the stored ``claim``, not the answer text. A claim is
supported when a single retrieved passage or a pair of retrieved passages
reaches tau. NLI splits each passage and each pair into sentence windows that
fit the token limit. A window that cannot fit is cut down to the limit and
counted. The count is part of the row.

The judge sample and the logistic holdout are seeded prefixes of two
independent shuffles. The stream name is mixed into the seed and stored on
the artifact. ``k`` is ``n * rate`` rounded half up, so a larger rate with
the same stream keeps the smaller sample as a prefix.

Self percentile compares an ok confidence with the other ok confidences in
the same dataset. Ties count half. A dataset with no other ok confidence
stores a null.

The logistic uses token-F1 correctness as the training label and
``self``, ``lexical``, and ``nli`` as features. It is fit per dataset on the
questions outside the holdout. Holdout rows get the predicted probability.
Fit-fold rows are not scored with that model. Features are the three signals
present on every complete row.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "query_api"))

import verified_reward as V  # noqa: E402

from experiments.common import ProtocolError
from experiments.metrics import score_answer

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

LOGISTIC_FEATURES = ("self", "lexical", "nli")
UnitScore = Callable[[str, str], tuple[float, int, int]]


def stream_seed(seed: int, name: str) -> int:
    """Independent deterministic stream. The artifact stores ``seed`` and ``name``."""
    if not isinstance(seed, int):
        raise ProtocolError(f"seed must be an int, got {seed!r}")
    if not name:
        raise ProtocolError("a random stream needs a name")
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def take_count(n: int, rate: float) -> int:
    """How many items ``rate`` keeps out of ``n``. Half rounds up.

    ``int(n * rate)`` drops a question when the product lands just below an
    integer. Decimal half-up keeps 60% of 10 equal to 6.
    """
    if n < 0:
        raise ProtocolError(f"sample size is negative: {n}")
    if rate < 0 or rate > 1:
        raise ProtocolError(f"rate must be in [0, 1], got {rate}")
    if n == 0 or rate == 0:
        return 0
    if rate == 1:
        return n
    scaled = n * rate
    nearest = math.floor(scaled + 0.5)
    # A product that is just under an integer (0.6 * 5 == 2.9999999999999996
    # in some rates) still belongs to that integer. 1e-9 is below one item.
    if abs(scaled - round(scaled)) < 1e-9:
        nearest = int(round(scaled))
    return int(nearest)


def seeded_ids(ids: Sequence[str], rate: float, seed: int) -> list[str]:
    """First ``take_count`` of ``sorted(ids)`` after ``Random(seed).shuffle``.

    The returned list is shuffle order, which is the prefix order. A higher
    rate with the same seed and the same ids keeps this list as a prefix.
    """
    ordered = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:take_count(len(ordered), rate)]


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_RE.split(text or "") if part.strip()]
    return parts


def pack_windows(
    passage: str,
    claim: str,
    count_tokens: Callable[[str], int],
    truncate: Callable[[str, int], str],
    limit: int,
    special_tokens: int,
) -> dict[str, Any]:
    """Sentence windows whose token count, plus the claim and specials, fits ``limit``.

    A sentence longer than the remaining budget is cut with ``truncate`` and
    counted. The model is then given the cut text, so the cut is recorded
    rather than left to the encoder.
    """
    if limit < 8:
        raise ProtocolError(f"nli token limit {limit} is below 8")
    if special_tokens < 0 or special_tokens >= limit:
        raise ProtocolError(f"nli special-token overhead {special_tokens} does not fit in {limit}")
    claim_truncated = False
    claim_text = claim or ""
    budget = limit - special_tokens - count_tokens(claim_text)
    if budget < 1:
        claim_truncated = True
        claim_text = truncate(claim_text, max(limit - special_tokens - 1, 1))
        budget = limit - special_tokens - count_tokens(claim_text)
        if budget < 1:
            return {
                "windows": [],
                "n_windows": 0,
                "n_truncated": 1,
                "claim_truncated": True,
                "claim_text": claim_text,
            }
    windows: list[dict[str, Any]] = []
    n_truncated = 1 if claim_truncated else 0
    buf: list[str] = []
    buf_n = 0

    def flush() -> None:
        nonlocal buf, buf_n
        if not buf:
            return
        text = " ".join(buf)
        if count_tokens(text) > budget:
            text = truncate(text, budget)
            windows.append({"text": text, "truncated": True})
            nonlocal_truncated[0] += 1
        else:
            windows.append({"text": text, "truncated": False})
        buf = []
        buf_n = 0

    nonlocal_truncated = [n_truncated]
    for sentence in split_sentences(passage):
        n = count_tokens(sentence)
        if n == 0:
            continue
        if n > budget:
            flush()
            windows.append({"text": truncate(sentence, budget), "truncated": True})
            nonlocal_truncated[0] += 1
            continue
        if buf and buf_n + n > budget:
            flush()
        buf.append(sentence)
        buf_n += n
    flush()
    n_truncated = nonlocal_truncated[0]
    if not windows and (passage or "").strip():
        windows.append({"text": truncate(passage, budget), "truncated": True})
        n_truncated += 1
    return {
        "windows": windows,
        "n_windows": len(windows),
        "n_truncated": n_truncated,
        "claim_truncated": claim_truncated,
        "claim_text": claim_text,
    }


def _passages(passages: Sequence[str]) -> list[str]:
    return [p for p in passages if isinstance(p, str) and p.strip()]


def _pair_texts(texts: Sequence[str]) -> list[str]:
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append(texts[i].rstrip() + "\n" + texts[j].lstrip())
    return pairs


def score_stored_claim(
    claim: str | None,
    passages: Sequence[str],
    unit_score: UnitScore,
    threshold: float,
) -> dict[str, Any]:
    """Fraction of checkable claim sentences supported by a passage or a pair.

    ``unit_score(claim, text)`` returns ``(score, n_truncated_windows, n_windows)``.
    Lexical scoring passes zeros for the window counts.
    """
    if claim is None or not str(claim).strip():
        return {"value": None, "reason": "no_claim", "detail": {"claims": 0}}
    texts = _passages(passages)
    if not texts:
        return {"value": None, "reason": "no_passages", "detail": {"claims": 0, "n_passages": 0}}
    claims = V.split_claims(claim)
    if not claims:
        return {
            "value": None,
            "reason": "no_claims",
            "detail": {"claims": 0, "n_passages": len(texts), "stored_claim": claim},
        }
    pairs = _pair_texts(texts)
    supported = 0
    by_single = 0
    by_pair = 0
    n_truncated = 0
    n_windows = 0
    best_single = None
    best_pair = None
    for sentence in claims:
        single_best = None
        for text in texts:
            score, trunc, windows = unit_score(sentence, text)
            n_truncated += trunc
            n_windows += windows
            single_best = score if single_best is None else max(single_best, score)
        pair_best = None
        for text in pairs:
            score, trunc, windows = unit_score(sentence, text)
            n_truncated += trunc
            n_windows += windows
            pair_best = score if pair_best is None else max(pair_best, score)
        if single_best is not None:
            best_single = single_best if best_single is None else max(best_single, single_best)
        if pair_best is not None:
            best_pair = pair_best if best_pair is None else max(best_pair, pair_best)
        hit_single = single_best is not None and single_best >= threshold
        hit_pair = pair_best is not None and pair_best >= threshold
        if hit_single:
            by_single += 1
        if hit_pair:
            by_pair += 1
        if hit_single or hit_pair:
            supported += 1
    return {
        "value": supported / len(claims),
        "reason": None,
        "detail": {
            "claims": len(claims),
            "supported": supported,
            "supported_by_single": by_single,
            "supported_by_pair": by_pair,
            "best_single": None if best_single is None else round(best_single, 4),
            "best_pair": None if best_pair is None else round(best_pair, 4),
            "n_passages": len(texts),
            "n_pairs": len(pairs),
            "n_windows": n_windows,
            "n_truncated_windows": n_truncated,
            "threshold": threshold,
        },
    }


def lexical_unit(claim: str, text: str) -> tuple[float, int, int]:
    return V.lexical_support(claim, text), 0, 1


def windowed_unit(
    predict: Callable[[str, str], float],
    count_tokens: Callable[[str], int],
    truncate: Callable[[str, int], str],
    limit: int,
    special_tokens: int,
) -> UnitScore:
    """NLI unit score. ``predict(claim, window)`` is the entailment probability."""

    def unit(claim: str, text: str) -> tuple[float, int, int]:
        packed = pack_windows(text, claim, count_tokens, truncate, limit, special_tokens)
        scores = [
            float(predict(packed["claim_text"], window["text"]))
            for window in packed["windows"]
            if window["text"]
        ]
        score = max(scores) if scores else 0.0
        return score, int(packed["n_truncated"]), int(packed["n_windows"])

    return unit


def self_percentile(value: float, peers: Sequence[float]) -> float | None:
    """Fraction of ``peers`` below ``value``, with ties counting half.

    ``peers`` are the other ok confidences. An empty peer list is null.
    """
    if not peers:
        return None
    less = sum(1 for item in peers if item < value)
    ties = sum(1 for item in peers if item == value)
    return (less + 0.5 * ties) / len(peers)


def assign_self_percentiles(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One output row per input row. Peers are other ok rows in the same dataset."""
    ok: dict[str, list[float]] = {}
    for row in rows:
        if row.get("self_status") == "ok" and row.get("self_confidence") is not None:
            ok.setdefault(row["dataset"], []).append(float(row["self_confidence"]))
    out = []
    for row in rows:
        status = row.get("self_status")
        if status != "ok" or row.get("self_confidence") is None:
            reason = status if isinstance(status, str) and status else "self_missing"
            out.append({**_key(row), "value": None, "reason": reason})
            continue
        value = float(row["self_confidence"])
        pool = ok.get(row["dataset"], [])
        peers = list(pool)
        # Drop one copy of this row's own value so a point is not its own peer.
        if value in peers:
            peers.remove(value)
        scored = self_percentile(value, peers)
        out.append({
            **_key(row),
            "value": scored,
            "reason": None if scored is not None else "no_peers",
            "self_confidence": value,
        })
    return out


def _key(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": row["qid"],
        "dataset": row["dataset"],
        "arm": row["arm"],
        "question_type": row.get("question_type"),
    }


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination. A vanishing pivot means the features are collinear."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ProtocolError(
                "logistic features are collinear. Refusing to invent a coefficient."
            )
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def fit_logistic(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> list[float]:
    """IRLS logistic coefficients ``[intercept, *features]``. No random step."""
    if len(features) != len(labels) or not features:
        raise ProtocolError("logistic fit needs one label per row")
    width = len(features[0])
    if any(len(row) != width for row in features):
        raise ProtocolError("logistic feature width differs across rows")
    classes = {int(label) for label in labels}
    if classes - {0, 1}:
        raise ProtocolError(f"logistic labels must be 0 or 1, got {sorted(classes)}")
    if len(classes) < 2:
        raise ProtocolError(
            "logistic fit has one class. Refusing to invent a coefficient."
        )
    if len(features) < width + 1:
        raise ProtocolError(
            f"logistic fit has {len(features)} rows and {width} features. "
            "Refusing to fit fewer rows than coefficients."
        )
    beta = [0.0] * (width + 1)
    for iteration in range(max_iter):
        dim = width + 1
        xtwx = [[0.0] * dim for _ in range(dim)]
        xtwz = [0.0] * dim
        for row, label in zip(features, labels):
            eta = beta[0] + sum(b * x for b, x in zip(beta[1:], row))
            mu = min(max(_sigmoid(eta), 1e-8), 1.0 - 1e-8)
            weight = mu * (1.0 - mu)
            z = eta + (float(label) - mu) / weight
            xs = [1.0, *row]
            for c in range(dim):
                xtwz[c] += weight * xs[c] * z
                for d in range(dim):
                    xtwx[c][d] += weight * xs[c] * xs[d]
        try:
            nxt = _solve(xtwx, xtwz)
        except ProtocolError:
            # The first step sees equal weights. A singular matrix there means
            # the columns themselves are dependent. A later step goes singular
            # when a separable sample drives some weights to zero; the last
            # finite coefficients are the fit.
            if iteration == 0:
                raise
            break
        delta = sum(abs(nxt[i] - beta[i]) for i in range(dim))
        beta = nxt
        if delta < tol:
            break
    return beta


def predict_logistic(beta: Sequence[float], features: Sequence[float]) -> float:
    if len(beta) != len(features) + 1:
        raise ProtocolError("logistic coefficient width does not match the features")
    eta = beta[0] + sum(b * x for b, x in zip(beta[1:], features))
    return _sigmoid(eta)


def logistic_rows(
    rows: Sequence[dict[str, Any]],
    *,
    holdout_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit per dataset on the complement of ``holdout_ids`` and score the holdout.

    A row needs ``self`` (ok confidence), ``lexical``, and ``nli``. Anything
    else is ``feature_missing`` and is left out of the fit. Fit-fold rows that
    were used stay null with reason ``fit_fold``.
    """
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    written: list[dict[str, Any]] = []
    coefficients: dict[str, Any] = {}
    for dataset, group in by_dataset.items():
        usable = []
        for row in group:
            features = _logistic_features(row)
            if features is None:
                continue
            if row["qid"] in holdout_ids:
                continue
            usable.append((row, features, int(row["correct"])))
        if not usable:
            raise ProtocolError(
                f"{dataset}: logistic fit has no complete rows outside the holdout. "
                "Refusing to invent coefficients."
            )
        beta = fit_logistic([item[1] for item in usable], [item[2] for item in usable])
        coefficients[dataset] = {
            "intercept": beta[0],
            "features": {name: beta[i + 1] for i, name in enumerate(LOGISTIC_FEATURES)},
            "n_fit": len(usable),
        }
        for row in group:
            features = _logistic_features(row)
            base = _key(row)
            if features is None:
                written.append({**base, "value": None, "reason": "feature_missing", "fold": None})
                continue
            if row["qid"] in holdout_ids:
                written.append({
                    **base,
                    "value": predict_logistic(beta, features),
                    "reason": None,
                    "fold": "holdout",
                })
            else:
                written.append({**base, "value": None, "reason": "fit_fold", "fold": "fit"})
    return written, {"features": list(LOGISTIC_FEATURES), "coefficients": coefficients}


def _logistic_features(row: dict[str, Any]) -> list[float] | None:
    if row.get("self_status") != "ok" or row.get("self_confidence") is None:
        return None
    lexical = row.get("lexical")
    nli = row.get("nli")
    if lexical is None or nli is None:
        return None
    return [float(row["self_confidence"]), float(lexical), float(nli)]


def oracle_correct(answer: str, gold_answers: Sequence[str], unanswerable: bool) -> int:
    return int(score_answer(answer, gold_answers, unanswerable)["correct"])


def oracle_retrieval(source_retrieved: bool) -> int:
    return 1 if source_retrieved else 0
