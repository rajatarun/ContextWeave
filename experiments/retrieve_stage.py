"""Four retrieval arms over each question's own candidate pool.

* ``semantic_search``: cosine similarity of a pinned local sentence-transformer.
* ``graph_first``: entity/co-occurrence graph built on the pool (below).
* ``keyword_boosted``: Okapi BM25, reranked by the same blend the deployed
  retriever uses (keyword weight 0.25, the rest the vector score).
* ``hybrid``: mean of min-max normalised vector, graph, and BM25 scores.

Graph construction, per question, over that question's pool only:

1. Entities in a passage are maximal capitalised phrases that are not a single
   stopword, plus numeric tokens. Questions in these datasets are usually
   lowercase, so a query term (a content token) matches an entity when the
   term casefolds equal to a word inside the entity.
2. An undirected edge joins two entities that occur in the same passage.
   Edge weight is the number of pool passages they share.
3. Query entities are the entities matched by at least one query term.
   Expanded entities are the query entities plus their graph neighbours.
4. A passage scores ``|E(passage) ∩ query entities| + 0.5 * |E(passage) ∩ neighbours|``.
   Ties break by passage id. A pool with no entities scores every passage 0
   and the ranking is the passage-id order.

The graph is not a corpus knowledge graph. It exists only inside one question's
candidate pool.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Sequence

from experiments.common import ARMS, ProtocolError

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_STOP = frozenset("""
a an the and or but if then than so of in on at to for from by with without into onto
over under about as is are was were be been being has have had do does did done can could
will would shall should may might must it its this that these those there here which who
whom whose what when where why how i you he she we they them his her their our your my
""".split())


EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if t.lower() not in _STOP and len(t) > 1]


def entities_in(text: str) -> set[str]:
    found: set[str] = set()
    for match in _ENTITY.findall(text or ""):
        parts = match.split()
        if len(parts) == 1 and parts[0].lower() in _STOP:
            continue
        found.add(match)
    found.update(_NUMBER.findall(text or ""))
    return found


def _words(entity: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9]+", entity)}


def graph_scores(question: str, passages: Sequence[dict[str, str]]) -> list[float]:
    ent_of = [entities_in(p["text"]) for p in passages]
    # co-occurrence: entity -> set of neighbour entities
    neighbours: dict[str, set[str]] = defaultdict(set)
    for ents in ent_of:
        items = sorted(ents)
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                neighbours[a].add(b)
                neighbours[b].add(a)
    q_terms = set(tokenize(question))
    all_entities = set().union(*ent_of) if ent_of else set()
    query_entities = {e for e in all_entities if _words(e) & q_terms or e in q_terms}
    expanded_extra: set[str] = set()
    for e in query_entities:
        expanded_extra |= neighbours.get(e, set())
    expanded_extra -= query_entities
    scores = []
    for ents in ent_of:
        direct = len(ents & query_entities)
        hop = len(ents & expanded_extra)
        scores.append(float(direct) + 0.5 * float(hop))
    return scores


def bm25_scores(question: str, passages: Sequence[dict[str, str]], k1: float, b: float) -> list[float]:
    docs = [tokenize(p["text"]) for p in passages]
    query = tokenize(question)
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs) / n
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    scores = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            n_q = df[term]
            idf = math.log(1.0 + (n - n_q + 0.5) / (n_q + 0.5))
            denom = freq + k1 * (1.0 - b + b * dl / avgdl) if avgdl else freq + k1
            score += idf * (freq * (k1 + 1.0)) / denom
        scores.append(score)
    return scores


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


def _rank(passages: Sequence[dict[str, str]], scores: Sequence[float], top_k: int) -> list[dict[str, Any]]:
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], passages[i]["id"]))
    chosen = order[:top_k]
    out = []
    for rank, i in enumerate(chosen, start=1):
        src = passages[i]
        out.append({
            "id": src["id"],
            "title": src.get("title", ""),
            "text": src["text"],
            "score": scores[i],
            "rank": rank,
        })
    return out


def rank_pool(
    question: str,
    passages: Sequence[dict[str, str]],
    embed: EmbedFn,
    top_k: int,
    *,
    keyword_weight: float = 0.25,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, list[dict[str, Any]]]:
    if top_k < 1:
        raise ProtocolError("top_k must be at least 1")
    if not passages:
        raise ProtocolError("candidate pool is empty")
    texts = [p["text"] for p in passages]
    vectors = embed([question, *texts])
    if len(vectors) != len(texts) + 1:
        raise ProtocolError("embedder returned the wrong number of vectors")
    qv, pvs = vectors[0], vectors[1:]
    cosine = [_cosine(qv, pv) for pv in pvs]
    graph = graph_scores(question, passages)
    bm25 = bm25_scores(question, passages, bm25_k1, bm25_b)
    bm25_n = _minmax(bm25)
    cos_n = _minmax(cosine)
    graph_n = _minmax(graph)
    kw = keyword_weight
    keyword = [(1.0 - kw) * c + kw * b for c, b in zip(cos_n, bm25_n)]
    hybrid = [(c + g + b) / 3.0 for c, g, b in zip(cos_n, graph_n, bm25_n)]
    by_arm = {
        "semantic_search": cosine,
        "graph_first": graph,
        "keyword_boosted": keyword,
        "hybrid": hybrid,
    }
    return {arm: _rank(passages, by_arm[arm], top_k) for arm in ARMS}


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

    # The library stores the snapshot revision on the transformer.
    commit = None
    try:
        commit = model[0].auto_model.config._commit_hash
    except Exception:
        commit = None
    info = {"embedding_model": model_name, "revision": commit, "card_base_model": revision}
    return embed, info


def retrieve_dataset(
    questions: Sequence[dict[str, Any]],
    embed: EmbedFn,
    top_k: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    kw = float(cfg["keyword_weight"])
    k1 = float(cfg["bm25_k1"])
    b = float(cfg["bm25_b"])
    for q in questions:
        ranked = rank_pool(
            q["question"], q["pool"], embed, top_k,
            keyword_weight=kw, bm25_k1=k1, bm25_b=b,
        )
        for arm, passages in ranked.items():
            rows.append({
                "qid": q["qid"],
                "dataset": q["dataset"],
                "arm": arm,
                "question_type": q["question_type"],
                "question": q["question"],
                "pool_size": len(q["pool"]),
                "passages": passages,
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
