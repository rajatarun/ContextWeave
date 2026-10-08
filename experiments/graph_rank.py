"""Entity graph with inverse-document-frequency edges and personalized PageRank.

The graph is built once on a dataset's shared passage collection.

* Passage entities come from a caller-supplied extractor. The retrieval
  script passes spaCy NER. This module does not substitute another extractor
  when spaCy is missing: the caller stops.
* An entity key is the surface form, casefolded, with whitespace collapsed.
  Document frequency is the number of passages that contain the key.
* The graph is bipartite. A passage node and an entity node are joined when
  the entity occurs in the passage. The undirected edge weight is ``1/df``.
* A query seed is a spaCy entity in the question whose key is in the graph,
  or a content token of the question that equals an entity key. Seeds split
  the personalization mass evenly.
* Passage score is the personalized PageRank mass on the passage node.
  Power iteration is deterministic: damping, iteration cap, and tolerance
  come from the config, and there is no random teleport. The retrieval
  artifact records ``seed: null`` for this step.
* When the question matches no entity, or every passage mass is zero, the
  arm's raw scores are the vector cosine scores. The row records
  ``graph_score_source: vector_fallback``.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from experiments.common import ProtocolError

EntityFn = Callable[[str], Sequence[str]]
TokenFn = Callable[[str], Sequence[str]]


def entity_key(surface: str) -> str:
    return " ".join((surface or "").casefold().split())


def build_graph(passages: Sequence[dict[str, str]], entity_fn: EntityFn) -> dict[str, Any]:
    """Bipartite passage/entity graph. Edge weight is ``1/df(entity)``."""
    if not passages:
        raise ProtocolError("shared passage collection is empty")
    passage_keys: dict[str, list[str]] = {}
    df: dict[str, int] = {}
    for passage in passages:
        pid = passage.get("id")
        if not pid:
            raise ProtocolError("a collection passage has an empty id")
        if pid in passage_keys:
            raise ProtocolError(f"duplicate passage id in the shared collection: {pid}")
        keys: list[str] = []
        seen: set[str] = set()
        for surface in entity_fn(passage.get("text") or ""):
            key = entity_key(surface)
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        passage_keys[pid] = keys
        for key in keys:
            df[key] = df.get(key, 0) + 1
    adj: dict[str, dict[str, float]] = {}
    for pid, keys in passage_keys.items():
        pnode = f"p:{pid}"
        for key in keys:
            weight = 1.0 / float(df[key])
            enode = f"e:{key}"
            adj.setdefault(pnode, {})[enode] = weight
            adj.setdefault(enode, {})[pnode] = weight
    return {
        "passage_ids": [p["id"] for p in passages],
        "df": df,
        "adj": adj,
        "edge_weight": "1/df",
    }


def query_seed_keys(question: str, entity_fn: EntityFn, known: set[str], tokenize: TokenFn) -> list[str]:
    """Entity keys in ``known``, in extractor order, then content-token order."""
    found: list[str] = []
    for surface in entity_fn(question or ""):
        key = entity_key(surface)
        if key in known and key not in found:
            found.append(key)
    for token in tokenize(question or ""):
        if token in known and token not in found:
            found.append(token)
    return found


def personalized_pagerank(
    adj: dict[str, dict[str, float]],
    seed_nodes: Sequence[str],
    *,
    damping: float,
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    """Deterministic power iteration. Dangling mass returns through the seeds."""
    if not seed_nodes:
        return {}
    nodes = sorted(set(adj) | set(seed_nodes))
    index = {node: i for i, node in enumerate(nodes)}
    missing = [node for node in seed_nodes if node not in index]
    if missing:
        raise ProtocolError(f"PageRank seed is not in the graph: {missing[0]}")
    n = len(nodes)
    out_sum = [0.0] * n
    edges: list[tuple[int, int, float]] = []
    for src, nbrs in adj.items():
        si = index[src]
        for dst, weight in nbrs.items():
            if weight <= 0:
                continue
            edges.append((si, index[dst], float(weight)))
            out_sum[si] += float(weight)
    personal = [0.0] * n
    share = 1.0 / len(seed_nodes)
    for node in seed_nodes:
        personal[index[node]] += share
    mass = personal[:]
    for _ in range(int(max_iter)):
        nxt = [(1.0 - damping) * p for p in personal]
        dangling = 0.0
        for i, total in enumerate(out_sum):
            if total == 0.0:
                dangling += mass[i]
        if dangling:
            for i in range(n):
                nxt[i] += damping * dangling * personal[i]
        for src, dst, weight in edges:
            nxt[dst] += damping * (weight / out_sum[src]) * mass[src]
        delta = sum(abs(nxt[i] - mass[i]) for i in range(n))
        mass = nxt
        if delta < tol:
            break
    return {node: mass[i] for node, i in index.items()}


def graph_or_fallback(
    question: str,
    passages: Sequence[dict[str, str]],
    entity_fn: EntityFn,
    vector_scores: Sequence[float],
    graph: dict[str, Any],
    tokenize: TokenFn,
    *,
    damping: float,
    max_iter: int,
    tol: float,
) -> tuple[list[float], str]:
    """PageRank passage scores, or the vector scores when the graph cannot rank."""
    if len(vector_scores) != len(passages):
        raise ProtocolError("vector score list does not match the passage collection")
    seeds = query_seed_keys(question, entity_fn, set(graph["df"]), tokenize)
    if not seeds or not graph["adj"]:
        return [float(s) for s in vector_scores], "vector_fallback"
    ranks = personalized_pagerank(
        graph["adj"], [f"e:{key}" for key in seeds],
        damping=damping, max_iter=max_iter, tol=tol,
    )
    scores = [float(ranks.get(f"p:{passage['id']}", 0.0)) for passage in passages]
    if all(score == 0.0 for score in scores):
        return [float(s) for s in vector_scores], "vector_fallback"
    return scores, "pagerank"


def load_spacy_entities(model_name: str) -> tuple[EntityFn, dict[str, Any]]:
    """Load a spaCy NER model. A missing install or model stops the run."""
    if not model_name:
        raise ProtocolError("spacy_model is empty. Refusing to guess an entity extractor.")
    try:
        import spacy
    except ImportError as exc:
        raise ProtocolError(
            "spacy is not installed; pip install -r experiments/requirements.txt"
        ) from exc
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:
        raise ProtocolError(
            f"spaCy model {model_name!r} is not installed. "
            f"Install it with: python -m spacy download {model_name}. "
            "Refusing to fall back to another entity extractor."
        ) from exc

    def extract(text: str) -> list[str]:
        if not text:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for ent in nlp(text).ents:
            surface = " ".join(ent.text.split())
            if len(surface) < 2 or surface in seen:
                continue
            seen.add(surface)
            out.append(surface)
        return out

    meta = getattr(nlp, "meta", {}) or {}
    info = {
        "extractor": "spacy",
        "spacy_model": model_name,
        "spacy_version": getattr(spacy, "__version__", None),
        "model_version": meta.get("version"),
        "lang": meta.get("lang"),
    }
    return extract, info
