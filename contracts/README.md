# Vendored contracts

These files are copies, not originals:

- `observatory_metrics_item.json`
- `conformance.py`

**Canonical home:** [`mcp-observatory/contracts/`](https://github.com/rajatarun/mcp-observatory/tree/main/contracts).
The shared `OBSERVATORY_METRICS` DynamoDB table has many writers and readers
across repositories that cannot see each other's code, so the item shape for
that table is defined once, in `mcp-observatory`, and vendored unmodified into
every repository that writes or reads it — this one included.

`conformance.py` is deliberately dependency-free (no `jsonschema`, no
`boto3`) precisely so it can be copied around like this and run the same way
in every consumer.

**Do not edit these files here.** If the contract changes, it changes in
`mcp-observatory` first, and every vendored copy — including this one — is
updated together to the new version. A copy that drifts from the canonical
file defeats the point of having one.

`contextweave_http_api.json` in this same directory is a different kind of
file: it is ContextWeave's own canonical contract (the HTTP surface
TeamWeave consumes), not a vendored copy, and lives here permanently.

## The score contract — canonical here

`score_envelope.json` and `score_contract.py` define what a score in `[0, 1]`
means across the weave systems (kinds, required fields, the R1–R5 combination
rules from `docs/confidence-semantics.md`) and how to check one. **This
directory is their canonical home**; DeviceWeave, CipherWeave and
mcp-observatory vendor them byte-identical, and every copy's
`tests/test_score_contract.py` pins the same sha256 so an edit in one place
fails there instead of drifting.

`scores.json` is *not* shared: each repository declares its own scores in its
own `contracts/scores.json` — name, kind, range, source, calibrated, meaning
and the producing code — and its test resolves every producer, so a
declaration cannot outlive the code it describes.
