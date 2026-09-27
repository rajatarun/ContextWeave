"""
Verified reward for the routing loop.

The router's reward used to be the synthesiser's own ``confidence``: the model's
opinion of its own answer. Exploration (Thompson sampling) fixed the *search*;
it cannot fix the *objective*. A strategy that retrieves the wrong passages and
gets a confidently wrong answer out of them is reinforced exactly as much as one
that retrieves the right ones, and nothing in the loop can tell the difference.

This module builds the reward from signals that can be checked against
something other than the model's self-report:

  grounding   Is each claim in the answer supported by the passages the chosen
              strategy retrieved? Computed on every answer, no model call. This
              is the signal that is actually *about the strategy*: retrieval is
              the only thing the router chooses, and a strategy whose passages do
              not support the answer has failed at its one job whatever the
              synthesiser thinks of the result.
  judge       A second, independent model grading the answer against the same
              passages, on a deterministic sample of queries (the cost is one
              extra Converse call per sampled query). Weighted above grounding
              because it reads meaning rather than tokens.
  human       ``POST /feedback`` (feedback.py) -- unchanged, delayed, folded in
              when it arrives.
  self        The synthesiser's confidence. Recorded, and used as a reward only
              when ``ROUTER_REWARD_SOURCE`` says so.

Missing is missing
------------------
Every signal is ``value: float | None``. ``None`` means *not observed* -- no
passages to check against, no claims in the answer, a judge call that failed or
was not sampled. ``combine`` averages only what was observed and returns
``None`` when nothing was, and the handler then **does not update the router**.
A default filled in by the code is not evidence about the strategy; the self-
confidence fallbacks (0.7 / 0.5 / 0.0) pulled every posterior toward whichever
constant fired most often before ``confidence_reported`` stopped them, and the
same mistake must not be reintroduced one layer up.

The grounding verifier
----------------------
``lexical_support`` is deliberately simple: the fraction of a claim's content
tokens that appear in a passage, with numbers treated as tokens so a figure the
passages never mention cannot be supported. It is a weak entailment proxy -- it
cannot see negation or paraphrase -- and it is not presented as anything else.
It is the default because it is deterministic, free, and runs inside the Lambda
with no model and no dependency. ``grounding_signal`` takes the verifier as an
argument, so an NLI cross-encoder can be dropped in where one is available
(``scripts/verified_reward_bench.py --verifier nli`` does exactly that, offline),
and the benchmark is how one finds out whether the proxy is good enough.

Environment
-----------
  ROUTER_REWARD_SOURCE        verified (default) | self | verified+self
  ROUTER_GROUNDING_THRESHOLD  float  default 0.6   support a claim needs to count as grounded
  ROUTER_GROUNDING_WEIGHT     float  default 1.0
  ROUTER_JUDGE_SAMPLE_RATE    float  default 0.0   fraction of queries graded by the judge
  ROUTER_JUDGE_WEIGHT         float  default 2.0
  ROUTER_JUDGE_MODEL_ID       str    default ""    judge disabled when empty
  ROUTER_SELF_WEIGHT          float  default 1.0   only used by verified+self
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

REWARD_MODES = ("verified", "self", "verified+self")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning("%s is not a number; using %s", name, default)
        return default


def reward_mode() -> str:
    mode = os.environ.get("ROUTER_REWARD_SOURCE", "verified").strip().lower()
    if mode not in REWARD_MODES:
        # An unknown mode must not silently become "self": that would restore
        # the reward this module exists to replace, with nothing in the logs.
        logger.warning("ROUTER_REWARD_SOURCE=%r is not one of %s; using 'verified'", mode, REWARD_MODES)
        return "verified"
    return mode


# Registry names and envelope sources (contracts/scores.json) for each signal.
_REGISTRY_NAME = {"self": "synthesis_confidence", "grounding": "grounding", "judge": "judge",
                  "human": "human_rating"}
_ENVELOPE_SOURCE = {"self": "self", "grounding": "measurement", "judge": "model", "human": "human"}


@dataclass(frozen=True)
class RewardSignal:
    """One observation of answer quality. ``value is None`` means not observed."""

    source: str
    value: float | None
    weight: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        return self.value is not None and self.weight > 0

    def to_dict(self) -> dict[str, Any]:
        # Shaped as a score envelope (contracts/score_envelope.json): every one
        # of these is an ordering-only, uncalibrated score, and ``observed`` is
        # what distinguishes "not measured" from a measured 0 (R5).
        return {
            "name": f"contextweave.{_REGISTRY_NAME.get(self.source, self.source)}",
            "kind": "score",
            "source": _ENVELOPE_SOURCE.get(self.source, "measurement"),
            "calibrated": False,
            "observed": self.value is not None,
            "value": None if self.value is None else round(self.value, 4),
            "weight": self.weight,
            **({"detail": self.detail} if self.detail else {}),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Grounding: is the answer supported by what the strategy retrieved?
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset("""
a an the and or but if then than so of in on at to for from by with without into onto
over under about as is are was were be been being has have had do does did done can could
will would shall should may might must it its this that these those there here which who
whom whose what when where why how i you he she we they them his her their our your my me
us not no also very more most such any all each both some many much other another own same
just only even still yet via per using used use based across within between through during
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# An answer that declines to answer makes no claim, so it has nothing to ground.
_ABSTAIN_RE = re.compile(
    r"\b(insufficient|not enough (?:evidence|information)|no (?:relevant )?evidence|"
    r"cannot (?:be )?(?:determined|answered|determine|answer)|"
    r"does not (?:mention|say|contain)|unable to (?:find|determine|answer))\b",
    re.IGNORECASE,
)


def _stem(tok: str) -> str:
    # Crude plural/tense folding so "services" matches "service" and "deployed"
    # matches "deploy". Numbers are left alone: 12 must not match 1.
    if tok[0].isdigit():
        return tok
    for suffix in ("ing", "ed", "es", "s"):
        if len(tok) > len(suffix) + 3 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    return tok


def content_tokens(text: str) -> set[str]:
    return {_stem(t) for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1}


def split_claims(answer: str, min_tokens: int = 3) -> list[str]:
    """Split an answer into sentence-level claims worth checking.

    Fragments with fewer than ``min_tokens`` content tokens ("Yes.", a bullet
    heading) carry no checkable content and are dropped rather than counted as
    unsupported, which would punish formatting.
    """
    parts = [p.strip(" -*•\t") for p in _SENTENCE_RE.split(answer or "")]
    return [p for p in parts if len(content_tokens(p)) >= min_tokens and not _ABSTAIN_RE.search(p)]


def lexical_support(claim: str, passage: str) -> float:
    """Fraction of the claim's content tokens present in the passage, in [0, 1].

    A number is a hard requirement rather than one token among many: "serving
    400 teams" against a passage that says 40 shares every other word, and a
    fraction-of-tokens score would call it supported. Any number in the claim
    that the passage does not contain makes the support 0.
    """
    ct = content_tokens(claim)
    if not ct:
        return 0.0
    pt = content_tokens(passage)
    if any(t[0].isdigit() and t not in pt for t in ct):
        return 0.0
    return len(ct & pt) / len(ct)


Verifier = Callable[[str, str], float]


def grounding_signal(
    answer: str,
    passages: Sequence[str],
    *,
    verifier: Verifier = lexical_support,
    threshold: float | None = None,
    weight: float | None = None,
) -> RewardSignal:
    """Fraction of the answer's claims supported by at least one retrieved passage.

    ``None`` when there is nothing to check: no passages (the strategy returned
    nothing, which the synthesiser answers as "insufficient evidence" -- a
    retrieval failure, but one that says nothing about *which claims* were
    grounded), or no checkable claims (an abstention).

    Why no passages is ``None`` and not ``0.0``: an empty retrieval is already
    visible as ``retrieval_count == 0``, and scoring it 0.0 would make grounding
    reward every strategy that happens to retrieve *something* over one that
    correctly retrieves nothing for an out-of-corpus question.
    """
    threshold = _env_float("ROUTER_GROUNDING_THRESHOLD", 0.6) if threshold is None else threshold
    weight = _env_float("ROUTER_GROUNDING_WEIGHT", 1.0) if weight is None else weight
    texts = [p for p in passages if p and p.strip()]
    if not texts:
        return RewardSignal("grounding", None, weight, {"reason": "no_passages"})
    claims = split_claims(answer)
    if not claims:
        return RewardSignal("grounding", None, weight, {"reason": "no_claims"})

    supports = [max(verifier(c, p) for p in texts) for c in claims]
    supported = sum(1 for s in supports if s >= threshold)
    return RewardSignal(
        "grounding",
        supported / len(claims),
        weight,
        {"claims": len(claims), "supported": supported, "meanSupport": round(sum(supports) / len(supports), 4)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Judge: an independent model grades a sample of answers
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented system.
You did not write the answer. Grade only against the EVIDENCE below; do not use outside knowledge.

QUESTION:
{question}

EVIDENCE:
{evidence}

ANSWER:
{answer}

Score how well the EVIDENCE supports the ANSWER and whether the ANSWER addresses the QUESTION.
1.0 = every claim is supported and the question is answered; 0.0 = unsupported or off-question.
An answer that correctly says the evidence is insufficient scores 1.0 if the evidence really is insufficient.
Respond with JSON only: {{"score": <number between 0 and 1>, "unsupported_claims": [<strings>]}}"""


def should_judge(query_id: str, rate: float | None = None) -> bool:
    """Deterministic sample: the same queryId is always in or out.

    Hash-based rather than random so a replay of the decision log selects the
    same queries, and the judged subset is a uniform sample of traffic rather
    than of whatever happened to run while the dice were kind.
    """
    rate = _env_float("ROUTER_JUDGE_SAMPLE_RATE", 0.0) if rate is None else rate
    if rate <= 0 or not query_id:
        return False
    if rate >= 1:
        return True
    h = int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < rate


def parse_judge_score(raw: str) -> float | None:
    """Read ``score`` from the judge's reply; None rather than a guess if absent."""
    if not raw:
        return None
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        score = json.loads(text).get("score")
    except (ValueError, AttributeError):
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    return score if 0.0 <= score <= 1.0 else None


def judge_signal(
    question: str,
    answer: str,
    passages: Sequence[str],
    invoke: Callable[[str], str],
    *,
    weight: float | None = None,
    max_evidence_chars: int = 12000,
) -> RewardSignal:
    """Ask ``invoke(prompt) -> text`` to grade the answer. Any failure is ``None``."""
    weight = _env_float("ROUTER_JUDGE_WEIGHT", 2.0) if weight is None else weight
    evidence = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1) if p)[:max_evidence_chars]
    if not evidence:
        return RewardSignal("judge", None, weight, {"reason": "no_passages"})
    prompt = JUDGE_PROMPT.format(question=question, evidence=evidence, answer=answer)
    try:
        raw = invoke(prompt)
    except Exception as exc:  # a failed call is not evidence about the strategy
        logger.warning("Judge call failed (not a reward): %s", exc)
        return RewardSignal("judge", None, weight, {"reason": "call_failed"})
    score = parse_judge_score(raw)
    if score is None:
        return RewardSignal("judge", None, weight, {"reason": "unparseable"})
    return RewardSignal("judge", score, weight)


def bedrock_judge(model_id: str, runtime_client: Any = None) -> Callable[[str], str]:
    """An ``invoke`` for ``judge_signal`` backed by Bedrock Converse at temperature 0."""

    def invoke(prompt: str) -> str:
        from shared.mcp_observatory import observe_converse_request

        client = runtime_client
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        request_body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 300, "temperature": 0},
        }
        resp = observe_converse_request(
            runtime_client=client,
            model_id=model_id,
            prompt=prompt,
            request_body=request_body,
            source="reward_judge",
            operation="judge_answer",
        )
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                return block["text"]
        return ""

    return invoke


# ─────────────────────────────────────────────────────────────────────────────
# Combining
# ─────────────────────────────────────────────────────────────────────────────

def self_signal(confidence: float | None, reported: bool, weight: float | None = None) -> RewardSignal:
    weight = _env_float("ROUTER_SELF_WEIGHT", 1.0) if weight is None else weight
    value = None if (confidence is None or not reported) else min(1.0, max(0.0, float(confidence)))
    return RewardSignal("self", value, weight)


def combine(signals: Iterable[RewardSignal]) -> tuple[float | None, list[str]]:
    """Weighted mean of the observed signals, and which sources contributed.

    Returns ``(None, [])`` when nothing was observed -- the caller must then
    leave the router alone rather than substitute a default.
    """
    num = den = 0.0
    used: list[str] = []
    for s in signals:
        if s.observed:
            num += s.weight * float(s.value)  # type: ignore[arg-type]
            den += s.weight
            used.append(s.source)
    if den <= 0:
        return None, []
    return min(1.0, max(0.0, num / den)), used


@dataclass
class RewardAssessment:
    """Everything the handler needs: the reward (or None) and what it was made of."""

    mode: str
    reward: float | None
    sources: list[str]
    signals: dict[str, RewardSignal]

    def value_of(self, source: str) -> float | None:
        s = self.signals.get(source)
        return None if s is None else s.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reward": None if self.reward is None else round(self.reward, 4),
            "sources": self.sources,
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
        }


def assess(
    *,
    query_id: str,
    question: str,
    answer: str,
    passages: Sequence[str],
    self_confidence: float | None,
    self_reported: bool,
    mode: str | None = None,
    judge_invoke: Callable[[str], str] | None = None,
    verifier: Verifier = lexical_support,
) -> RewardAssessment:
    """Compute every signal that applies and the reward the router should learn from.

    The grounding and self signals are always computed and returned, whatever
    the mode, so the decision log holds all of them and the calibration of each
    can be measured against ratings later. ``mode`` decides only which of them
    the *reward* is made of.
    """
    mode = mode or reward_mode()
    signals: dict[str, RewardSignal] = {
        "self": self_signal(self_confidence, self_reported),
        "grounding": grounding_signal(answer, passages, verifier=verifier),
    }

    if judge_invoke is None:
        judge_model = os.environ.get("ROUTER_JUDGE_MODEL_ID", "").strip()
        if judge_model:
            judge_invoke = bedrock_judge(judge_model)
    if mode != "self" and judge_invoke is not None and should_judge(query_id):
        signals["judge"] = judge_signal(question, answer, passages, judge_invoke)

    if mode == "self":
        chosen = [signals["self"]]
    elif mode == "verified":
        chosen = [s for k, s in signals.items() if k != "self"]
    else:  # verified+self
        chosen = list(signals.values())
    reward, used = combine(chosen)
    return RewardAssessment(mode=mode, reward=reward, sources=used, signals=signals)
