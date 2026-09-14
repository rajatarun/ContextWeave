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
