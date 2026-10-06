#!/usr/bin/env python3
"""Run all four retrieval arms over each sampled question's candidate pool."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (
    DATASETS, RESULTS, SAMPLE_FIELDS, ProtocolError, artifact_meta, done_keys,
    load_config, read_jsonl, validate_rows, write_json,
)
from experiments.retrieve_stage import load_embedder, retrieval_stats, retrieve_dataset


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
        cache: dict[str, list[float]] = {}

        def embed(texts):
            missing = [t for t in texts if t not in cache]
            if missing:
                for text, vec in zip(missing, raw_embed(missing)):
                    cache[text] = vec
            return [cache[t] for t in texts]
        per = {}
        for name in names:
            src = args.results / "samples" / f"{name}.jsonl"
            questions = read_jsonl(src)
            validate_rows(questions, SAMPLE_FIELDS, src)
            out = args.results / "retrieval" / f"{name}.jsonl"
            done = done_keys(out, ("qid", "arm"))
            pending = [q for q in questions if any((q["qid"], arm) not in done for arm in (
                "semantic_search", "graph_first", "keyword_boosted", "hybrid"))]
            print(f"{name}: {len(pending)} questions to retrieve ({len(done)} arm-rows already written)", flush=True)
            if pending:
                rows = retrieve_dataset(pending, embed, top_k, cfg)
                with out.open("a") as fh:
                    for row in rows:
                        if (row["qid"], row["arm"]) in done:
                            continue
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done.add((row["qid"], row["arm"]))
            written = read_jsonl(out)
            validate_rows(written, ("qid", "dataset", "arm", "question_type", "passages", "pool_size"), out)
            per[name] = retrieval_stats(written)
            print(f"  pool mean {per[name]['pool_size_mean']:.2f}  "
                  f"top1 differs {per[name]['fraction_top1_differs_across_arms']:.3f}", flush=True)
        stats = artifact_meta(
            cfg, seed, stage="retrieval", top_k=top_k, embedding=model_info, datasets=per,
        )
        write_json(args.results / "retrieval" / "retrieval_stats.json", stats)
    except ProtocolError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"wrote {args.results / 'retrieval' / 'retrieval_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
