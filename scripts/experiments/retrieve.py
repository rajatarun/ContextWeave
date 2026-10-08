#!/usr/bin/env python3
"""Run all four retrieval arms over each dataset's shared passage collection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (
    DATASETS, RESULTS, SAMPLE_FIELDS, ProtocolError, append_jsonl, artifact_meta,
    done_keys, load_config, read_json, read_jsonl, stage_jsonl, validate_rows, write_json,
)
from experiments.graph_rank import load_spacy_entities
from experiments.retrieve_stage import load_embedder, retrieval_stats, retrieve_dataset

load_entities = load_spacy_entities


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    top_k = cfg["top_k"] if args.top_k is None else args.top_k
    names = [p.strip() for p in args.datasets.split(",") if p.strip()]
    try:
        raw_embed, model_info = load_embedder(cfg["embedding_model"])
        entity_fn, entity_info = load_entities(cfg["spacy_model"])
        cache: dict[str, list[float]] = {}

        def embed(texts):
            missing = [t for t in texts if t not in cache]
            if missing:
                for text, vec in zip(missing, raw_embed(missing)):
                    cache[text] = vec
            return [cache[t] for t in texts]
        stats_path = args.results / "retrieval" / "retrieval_stats.json"
        per = {}
        if stats_path.is_file() and set(names) != set(DATASETS):
            previous = read_json(stats_path)
            if isinstance(previous.get("datasets"), dict):
                per.update({k: v for k, v in previous["datasets"].items() if k not in names})
        for name in names:
            src = args.results / "samples" / f"{name}.jsonl"
            collection_path = args.results / "samples" / f"{name}.passages.jsonl"
            questions = read_jsonl(src)
            validate_rows(questions, SAMPLE_FIELDS, src)
            passages = read_jsonl(collection_path)
            validate_rows(passages, ("id", "text"), collection_path)
            out = stage_jsonl(args.results / "retrieval", name)
            done = done_keys(out, ("qid", "arm"))
            pending = [q for q in questions if any((q["qid"], arm) not in done for arm in (
                "semantic_search", "graph_first", "keyword_boosted", "hybrid"))]
            print(f"{name}: {len(pending)} questions to retrieve ({len(done)} arm-rows already written)", flush=True)
            if pending:
                rows = retrieve_dataset(pending, passages, embed, entity_fn, top_k, cfg)
                for row in rows:
                    if (row["qid"], row["arm"]) in done:
                        continue
                    append_jsonl(out, row)
                    done.add((row["qid"], row["arm"]))
            allowed = {q["qid"] for q in questions}
            written_all = read_jsonl(out)
            written = [row for row in written_all if row.get("qid") in allowed]
            outside = len({row.get("qid") for row in written_all} - allowed)
            if outside:
                print(
                    f"  {outside} question ids outside the sample stay in the file and are left out of the stats",
                    flush=True,
                )
            validate_rows(written, (
                "qid", "dataset", "arm", "question_type", "passages", "pool_size",
                "gold_in_top_k", "source_retrieved",
            ), out)
            per[name] = retrieval_stats(written)
            print(f"  pool mean {per[name]['pool_size_mean']:.2f}  "
                  f"top1 differs {per[name]['fraction_top1_differs_across_arms']:.3f}", flush=True)
        stats = artifact_meta(
            cfg, seed, stage="retrieval", top_k=top_k, embedding=model_info,
            entities=entity_info, collection="shared_dev_split",
            score_normalization="per_query_minmax",
            pagerank={
                "damping": cfg["graph_damping"],
                "max_iter": cfg["graph_max_iter"],
                "tol": cfg["graph_tol"],
                "deterministic": True,
                "seed": None,
            },
            datasets=per,
        )
        write_json(args.results / "retrieval" / "retrieval_stats.json", stats)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"wrote {args.results / 'retrieval' / 'retrieval_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
