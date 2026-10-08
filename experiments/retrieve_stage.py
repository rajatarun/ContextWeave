"""Four retrieval arms over one shared passage collection per dataset.

* ``semantic_search``: cosine similarity of a pinned local sentence-transformer.
* ``graph_first``: spaCy entities, ``1/df`` bipartite edges, personalized
  PageRank (``experiments.graph_rank``). No query entity, or zero mass on
  every passage, uses the vector scores and records ``vector_fallback``.
* ``keyword_boosted``: Okapi BM25, reranked by the deployed blend (keyword
  weight 0.25, the rest the vector score) after min-max normalising both.
* ``hybrid``: mean of the min-max normalised vector, graph, and BM25 scores.

Each arm's stored ``score`` is that arm's raw score after a per-query
min-max over the shared collection, so the four arms sit on one scale.
``raw_score`` keeps the value from before that step. Min-max does not
reorder an arm. Ties break by passage id.

The collection is the dev-split passages, the same pool for every question
and every arm. ``gold_in_top_k`` and ``source_retrieved`` are written on
each arm row from ``source_passage_ids``.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Sequence

from experiments.common import ARMS, ProtocolError, snapshot_revision
from experiments.graph_rank import build_graph, graph_or_fallback
from experiments.labels import retrieval_label

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)
_STOP = frozenset("""
a an the and or but if then than so of in on at to for from by with without into onto
over under about as is are was were be been being has have had do does did done can could
will would shall should may might must it its this that these those there here which who
whom whose what when where why how i you he she we they them his her their our your my
""".split())


EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if t.lower() not in _STOP and len(t) > 1]


def graph_scores(
    question: str,
    passages: Sequence[dict[str, str]],
    entity_fn: Callable[[str], Sequence[str]],
    vector_scores: Sequence[float] | None = None,
    *,
    damping: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-10,
) -> list[float]:
    """PageRank scores for ``passages``. Tests pass ``entity_fn`` explicitly.

    The retrieval script uses spaCy. This function does not pick an extractor.
    """
    if vector_scores is None:
        vector_scores = [0.0] * len(passages)
    graph = build_graph(passages, entity_fn)
    scores, _source = graph_or_fallback(
        question, passages, entity_fn, vector_scores, graph, tokenize,
        damping=damping, max_iter=max_iter, tol=tol,
    )
    return scores


class BM25Index:
    """Okapi BM25 over a fixed document set. ``scores`` is the query-only half.

    Document frequency depends only on the documents, so a caller that ranks
    many queries against one corpus (NQ hard negatives, one index per source
    document) builds this once.
    """

    def __init__(self, docs: Sequence[Sequence[str]], k1: float, b: float):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.avgdl = (sum(len(doc) for doc in docs) / self.n) if self.n else 0.0
        self.df: Counter[str] = Counter()
        self.tfs: list[Counter[str]] = []
        self.dls: list[int] = []
        for doc in docs:
            self.df.update(set(doc))
            self.tfs.append(Counter(doc))
            self.dls.append(len(doc))

    def scores(self, query: Sequence[str]) -> list[float]:
        if self.n == 0:
            return []
        out = []
        k1, b, avgdl, n, df = self.k1, self.b, self.avgdl, self.n, self.df
        for tf, dl in zip(self.tfs, self.dls):
            score = 0.0
            for term in query:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                n_q = df[term]
                idf = math.log(1.0 + (n - n_q + 0.5) / (n_q + 0.5))
                denom = freq + k1 * (1.0 - b + b * dl / avgdl) if avgdl else freq + k1
                score += idf * (freq * (k1 + 1.0)) / denom
            out.append(score)
        return out


def bm25_from_tokens(query: Sequence[str], docs: Sequence[Sequence[str]], k1: float, b: float) -> list[float]:
    return BM25Index(docs, k1, b).scores(query)


def bm25_scores(question: str, passages: Sequence[dict[str, str]], k1: float, b: float) -> list[float]:
    return bm25_from_tokens(tokenize(question), [tokenize(p["text"]) for p in passages], k1, b)


def _minmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _rank(
    passages: Sequence[dict[str, str]],
    raw_scores: Sequence[float],
    norm_scores: Sequence[float],
    top_k: int,
) -> list[dict[str, Any]]:
    # Min-max is monotonic, so rank follows the raw score. Passage id breaks ties.
    order = sorted(range(len(passages)), key=lambda i: (-raw_scores[i], passages[i]["id"]))
    chosen = order[:top_k]
    out = []
    for rank, i in enumerate(chosen, start=1):
        src = passages[i]
        out.append({
            "id": src["id"],
            "title": src.get("title", ""),
            "text": src["text"],
            "score": norm_scores[i],
            "raw_score": raw_scores[i],
            "rank": rank,
        })
    return out


def rank_pool(
    question: str,
    passages: Sequence[dict[str, str]],
    embed: EmbedFn,
    top_k: int,
    *,
    entity_fn: Callable[[str], Sequence[str]],
    keyword_weight: float = 0.25,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    damping: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-10,
    graph: dict[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if entity_fn is None:
        raise ProtocolError("rank_pool requires an entity extractor. Refusing to guess one.")
    if top_k < 1:
        raise ProtocolError("top_k must be at least 1")
    if not passages:
        raise ProtocolError("shared passage collection is empty")
    texts = [p["text"] for p in passages]
    vectors = embed([question, *texts])
    if len(vectors) != len(texts) + 1:
        raise ProtocolError("embedder returned the wrong number of vectors")
    qv, pvs = vectors[0], vectors[1:]
    cosine = [_cosine(qv, pv) for pv in pvs]
    built = graph if graph is not None else build_graph(passages, entity_fn)
    graph_raw, source = graph_or_fallback(
        question, passages, entity_fn, cosine, built, tokenize,
        damping=damping, max_iter=max_iter, tol=tol,
    )
    bm25 = bm25_scores(question, passages, bm25_k1, bm25_b)
    bm25_n = _minmax(bm25)
    cos_n = _minmax(cosine)
    graph_n = _minmax(graph_raw)
    kw = keyword_weight
    keyword = [(1.0 - kw) * c + kw * b for c, b in zip(cos_n, bm25_n)]
    hybrid = [(c + g + b) / 3.0 for c, g, b in zip(cos_n, graph_n, bm25_n)]
    raw_by_arm = {
        "semantic_search": cosine,
        "graph_first": graph_raw,
        "keyword_boosted": keyword,
        "hybrid": hybrid,
    }
    # The blend inputs are already min-maxed. The stored score is a second
    # min-max of each arm so a raw cosine and a raw PageRank mass both land
    # on [0, 1] for that query.
    ranked = {
        arm: _rank(passages, raw_by_arm[arm], _minmax(raw_by_arm[arm]), top_k)
        for arm in ARMS
    }
    meta = {
        "graph_score_source": source,
        "score_normalization": "per_query_minmax",
        "pagerank": {
            "damping": damping,
            "max_iter": max_iter,
            "tol": tol,
            "deterministic": True,
            "seed": None,
        },
    }
    return ranked, meta


def load_embedder(model_name: str) -> tuple[EmbedFn, dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ProtocolError(
            "sentence-transformers is not installed; pip install -r experiments/requirements.txt"
        ) from exc
    model = SentenceTransformer(model_name, device="cpu")
    revision = getattr(getattr(model, "model_card_data", None), "base_model", None)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        arr = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return [row.tolist() for row in arr]

    # config._commit_hash is often empty after a cache load. The tokenizer
    # file path still contains snapshots/<commit>/.
    commit = None
    vocab = None
    try:
        commit = model[0].auto_model.config._commit_hash
    except Exception:
        commit = None
    try:
        vocab = model[0].tokenizer.init_kwargs.get("vocab_file")
    except Exception:
        vocab = None
    commit = snapshot_revision(commit, vocab)
    info = {"embedding_model": model_name, "revision": commit, "card_base_model": revision}
    return embed, info


def retrieve_dataset(
    questions: Sequence[dict[str, Any]],
    passages: Sequence[dict[str, str]],
    embed: EmbedFn,
    entity_fn: Callable[[str], Sequence[str]],
    top_k: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank ``passages`` for every question. The collection is shared."""
    if not passages:
        raise ProtocolError("shared passage collection is empty")
    known = {p["id"] for p in passages}
    if len(known) != len(passages):
        raise ProtocolError("shared passage collection has a duplicate passage id")
    graph = build_graph(passages, entity_fn)
    rows = []
    kw = float(cfg["keyword_weight"])
    k1 = float(cfg["bm25_k1"])
    b = float(cfg["bm25_b"])
    damping = float(cfg["graph_damping"])
    max_iter = int(cfg["graph_max_iter"])
    tol = float(cfg["graph_tol"])
    for q in questions:
        source_ids = q.get("source_passage_ids")
        if not source_ids:
            raise ProtocolError(f"{q.get('dataset')}:{q.get('qid')}: source_passage_ids is empty")
        missing = [pid for pid in source_ids if pid not in known]
        if missing:
            raise ProtocolError(
                f"{q.get('dataset')}:{q.get('qid')}: source passage {missing[0]} "
                "is not in the shared collection"
            )
        ranked, meta = rank_pool(
            q["question"], passages, embed, top_k,
            entity_fn=entity_fn, keyword_weight=kw, bm25_k1=k1, bm25_b=b,
            damping=damping, max_iter=max_iter, tol=tol, graph=graph,
        )
        for arm, arm_passages in ranked.items():
            label = retrieval_label(source_ids, [p["id"] for p in arm_passages])
            rows.append({
                "qid": q["qid"],
                "dataset": q["dataset"],
                "arm": arm,
                "question_type": q["question_type"],
                "question": q["question"],
                "pool_size": len(passages),
                "collection": "shared",
                "passages": arm_passages,
                "source_passage_ids": list(source_ids),
                "unanswerable": bool(q["unanswerable"]),
                "yes_no": bool(q.get("yes_no")),
                "graph_score_source": meta["graph_score_source"],
                "score_normalization": meta["score_normalization"],
                **label,
            })
    return rows


def retrieval_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_q: dict[tuple, dict[str, list[str]]] = defaultdict(dict)
    pool_sizes = []
    seen_q = set()
    for row in rows:
        key = (row["dataset"], row["qid"])
        by_q[key][row["arm"]] = [p["id"] for p in row["passages"]]
        if key not in seen_q:
            seen_q.add(key)
            pool_sizes.append(row["pool_size"])
    n = len(by_q)
    differ = 0
    jaccards = []
    for arms in by_q.values():
        tops = [ids[0] for ids in arms.values() if ids]
        if len(set(tops)) > 1:
            differ += 1
        lists = list(arms.values())
        for i in range(len(lists)):
            for j in range(i + 1, len(lists)):
                a, b = set(lists[i]), set(lists[j])
                union = a | b
                jaccards.append((len(a & b) / len(union)) if union else 1.0)
    return {
        "n_questions": n,
        "n_rows": len(rows),
        "pool_size_min": min(pool_sizes) if pool_sizes else None,
        "pool_size_max": max(pool_sizes) if pool_sizes else None,
        "pool_size_mean": (sum(pool_sizes) / len(pool_sizes)) if pool_sizes else None,
        "fraction_top1_differs_across_arms": (differ / n) if n else None,
        "mean_pairwise_topk_jaccard": (sum(jaccards) / len(jaccards)) if jaccards else None,
    }
