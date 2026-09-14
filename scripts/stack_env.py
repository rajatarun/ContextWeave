#!/usr/bin/env python3
"""Resolve every endpoint and database coordinate a test harness needs from the
CloudFormation stack, so nothing has to be hardcoded or pasted from a console.

The problem this solves: an automation run needs the API base URL, the Postgres
host/port/database, the secret holding its credentials, and -- because the
instance is stopped between runs to save money -- the RDS instance identifier to
start it with. Before this script the API URL lived in a CI variable, the
Postgres host was read out of a secret whose ARN was itself hardcoded, and the
instance identifier was not published at all: the deploy workflow dug it out
with ``describe-stack-resource --logical-resource-id PostgresRDS``, which works
only if you already know the logical id. Each of those is a copy of something
CloudFormation already knows, and each drifts silently when the stack changes.

Every value below is read from a stack Output. ``tests/test_openapi_contract.py``
asserts that each output named here actually exists in ``template.yaml``, so
deleting an output breaks the test rather than the next automation run.

Usage
-----
    python scripts/stack_env.py --stack contextweave-rag-prod            # JSON
    eval "$(python scripts/stack_env.py --stack contextweave-rag-prod --format sh)"

    # then, in a harness:
    curl -sS "$CONTEXTWEAVE_API_BASE/health"

Credentials are NOT resolved here. ``CONTEXTWEAVE_PG_SECRET_ARN`` names the
Secrets Manager secret; fetch it at the point of use so the password never
lands in a shell environment, a process list, or a CI log.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys

# env var -> stack Output key. Absence of a REQUIRED output is an error: a
# harness that silently proceeds without the API base URL will fail later, in a
# place that does not say "the stack does not export this".
REQUIRED_OUTPUTS = {
    "CONTEXTWEAVE_API_BASE": "APIEndpoint",
    "CONTEXTWEAVE_PG_HOST": "PostgresEndpoint",
    "CONTEXTWEAVE_PG_PORT": "PostgresPort",
    "CONTEXTWEAVE_PG_DATABASE": "PostgresDbName",
    "CONTEXTWEAVE_PG_SECRET_ARN": "PostgresSecretArn",
    "CONTEXTWEAVE_PG_INSTANCE_ID": "PostgresInstanceIdentifier",
}

# Useful but not fatal to omit -- these are conditional on the deployed
# configuration (Step Functions are behind a condition) or only needed by a
# subset of harnesses.
OPTIONAL_OUTPUTS = {
    "CONTEXTWEAVE_QUERY_URL": "QueryExpertiseURL",
    "CONTEXTWEAVE_ARTIFACTS_BUCKET": "ArtifactsBucketName",
    "CONTEXTWEAVE_MEMGRAPH_SECRET_ARN": "MemgraphSecretArn",
    "CONTEXTWEAVE_MEMGRAPH_INSTANCE_ID": "MemgraphInstanceId",
    "CONTEXTWEAVE_OPENAPI_SPEC": "OpenApiSpecPath",
    "CONTEXTWEAVE_INGESTION_STATE_MACHINE_ARN": "IngestionStateMachineArn",
}


class MissingOutputs(RuntimeError):
    """The stack exists but does not publish something a harness needs."""


def build_env(outputs: dict) -> dict:
    """Map stack outputs onto environment variable names.

    Pure: takes the already-fetched ``{OutputKey: OutputValue}`` mapping so the
    naming contract can be tested without an AWS account.
    """
    env = {}
    missing = []
    for var, key in REQUIRED_OUTPUTS.items():
        if key in outputs and outputs[key] != "":
            env[var] = outputs[key]
        else:
            missing.append(key)
    if missing:
        raise MissingOutputs(
            "stack publishes no value for: "
            + ", ".join(sorted(missing))
            + ". Deploy a template that exports them (see the Outputs section "
            "of template.yaml) -- do not hardcode them in the harness."
        )
    for var, key in OPTIONAL_OUTPUTS.items():
        if outputs.get(key):
            env[var] = outputs[key]
    return env


def fetch_outputs(stack: str, region: str | None) -> dict:
    """Read a stack's outputs, preferring boto3 and falling back to the CLI."""
    try:
        import boto3  # noqa: PLC0415 -- optional; the CLI path exists for CI images without it
    except ImportError:
        boto3 = None

    if boto3 is not None:
        cfn = boto3.client("cloudformation", region_name=region) if region else boto3.client("cloudformation")
        stacks = cfn.describe_stacks(StackName=stack)["Stacks"]
    else:
        cmd = ["aws", "cloudformation", "describe-stacks", "--stack-name", stack, "--output", "json"]
        if region:
            cmd += ["--region", region]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        stacks = json.loads(proc.stdout)["Stacks"]

    return {o["OutputKey"]: o.get("OutputValue", "") for o in stacks[0].get("Outputs", [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", default="contextweave-rag-dev",
                    help="CloudFormation stack name (default: contextweave-rag-dev)")
    ap.add_argument("--region", default=None, help="AWS region (default: the caller's configured region)")
    ap.add_argument("--format", choices=("json", "sh"), default="json",
                    help="json (default) or sh, for `eval \"$(...)\"`")
    args = ap.parse_args(argv)

    try:
        outputs = fetch_outputs(args.stack, args.region)
        env = build_env(outputs)
    except (MissingOutputs, RuntimeError) as exc:
        print(f"stack_env: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        for var in sorted(env):
            print(f"export {var}={shlex.quote(env[var])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
