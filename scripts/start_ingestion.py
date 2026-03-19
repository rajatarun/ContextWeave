#!/usr/bin/env python3
"""
ExpertiseRAG – Ingest all raw/ repos into pgvector + Memgraph

Discovers every immediate sub-prefix under raw/ in the artifacts S3 bucket
and triggers the preprocessor for each one, either via the Step Functions
state machine (preferred) or by directly invoking the preprocessor Lambda.

Usage:
    python scripts/start_ingestion.py [options]

Options:
    --stack-name    CloudFormation stack name (default: expertise-rag-dev)
    --env           Environment suffix override (default: derived from stack name)
    --repo          Only ingest this specific repo prefix, e.g. contextweave
    --region        AWS region (default: us-east-1)
    --profile       AWS profile name
    --wait          Poll each execution until complete
    --parallel      Start all executions simultaneously then wait (implies --wait)

Prerequisites:
    pip install boto3
    aws configure  # or set AWS_PROFILE / AWS_ACCESS_KEY_ID etc.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError


# ─────────────────────────────────────────────────────────────────────────────
# Stack helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_stack_outputs(cf_client, stack_name: str) -> dict[str, str]:
    """Return all CloudFormation stack outputs as a flat dict."""
    resp = cf_client.describe_stacks(StackName=stack_name)
    outputs = resp["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}


def resolve_config(stack_name: str, region: str, profile: Optional[str]) -> dict:
    """
    Resolve bucket name, state machine ARN, and preprocessor Lambda name
    from CloudFormation stack outputs.
    """
    session = boto3.Session(region_name=region, profile_name=profile)
    cf = session.client("cloudformation")

    try:
        outputs = get_stack_outputs(cf, stack_name)
    except ClientError as exc:
        print(f"ERROR: Could not describe stack '{stack_name}': {exc.response['Error']['Message']}")
        sys.exit(1)

    bucket = outputs.get("ArtifactsBucketName")
    if not bucket:
        print("ERROR: Stack output 'ArtifactsBucketName' not found. Has the stack deployed?")
        sys.exit(1)

    sf_arn = outputs.get("IngestionStateMachineArn")

    # Derive the preprocessor function name from the stack name
    # Stack: expertise-rag-dev → function: expertise-rag-preprocessor-dev
    env = stack_name.rsplit("-", 1)[-1]  # dev / staging / prod
    lambda_name = f"expertise-rag-preprocessor-{env}"

    return {
        "session": session,
        "bucket": bucket,
        "sf_arn": sf_arn,
        "lambda_name": lambda_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# S3 prefix discovery
# ─────────────────────────────────────────────────────────────────────────────

def list_raw_repos(s3_client, bucket: str) -> list[str]:
    """
    Return all immediate sub-prefixes of raw/ (one per repo folder).
    e.g. ['raw/contextweave/', 'raw/otherrepo/']
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    repos: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix="raw/", Delimiter="/"):
        for prefix_obj in page.get("CommonPrefixes", []):
            repos.append(prefix_obj["Prefix"])
    return repos


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion dispatch
# ─────────────────────────────────────────────────────────────────────────────

def start_via_step_functions(
    sf_client,
    sf_arn: str,
    bucket: str,
    repo_prefix: str,
) -> str:
    """Start a Step Functions execution and return the execution ARN."""
    import uuid
    payload = {
        "repo_prefix": repo_prefix,
        "bucket": bucket,
        "trigger_source": "manual-ingest-script",
    }
    resp = sf_client.start_execution(
        stateMachineArn=sf_arn,
        name=f"ingest-{repo_prefix.strip('/').replace('/', '-')}-{uuid.uuid4().hex[:8]}",
        input=json.dumps(payload),
    )
    return resp["executionArn"]


def start_via_lambda(
    lambda_client,
    lambda_name: str,
    bucket: str,
    repo_prefix: str,
) -> dict:
    """Directly invoke the preprocessor Lambda and return the parsed response."""
    payload = {
        "repo_prefix": repo_prefix,
        "bucket": bucket,
        "trigger_source": "manual-ingest-script",
    }
    resp = lambda_client.invoke(
        FunctionName=lambda_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode(),
    )
    body = json.loads(resp["Payload"].read())
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Polling
# ─────────────────────────────────────────────────────────────────────────────

def poll_execution(sf_client, exec_arn: str, repo_prefix: str, poll_interval: int = 15) -> str:
    """
    Poll a Step Functions execution until it reaches a terminal state.
    Returns the final status string.
    """
    elapsed = 0
    while True:
        resp = sf_client.describe_execution(executionArn=exec_arn)
        status = resp["status"]
        print(f"  [{elapsed:4d}s] {repo_prefix:<35} {status}", flush=True)
        if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"):
            return status
        time.sleep(poll_interval)
        elapsed += poll_interval


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest all raw/ repos into pgvector + Memgraph"
    )
    parser.add_argument("--stack-name", default="expertise-rag-dev",
                        help="CloudFormation stack name (default: expertise-rag-dev)")
    parser.add_argument("--repo", default=None,
                        help="Only ingest this specific repo, e.g. contextweave")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--wait", action="store_true",
                        help="Poll each execution until complete")
    parser.add_argument("--parallel", action="store_true",
                        help="Start all executions then wait in parallel (implies --wait)")
    parser.add_argument("--poll-interval", type=int, default=15)
    args = parser.parse_args()

    if args.parallel:
        args.wait = True

    cfg = resolve_config(args.stack_name, args.region, args.profile)
    session: boto3.Session = cfg["session"]
    bucket: str = cfg["bucket"]
    sf_arn: Optional[str] = cfg["sf_arn"]
    lambda_name: str = cfg["lambda_name"]

    s3 = session.client("s3")
    sf = session.client("stepfunctions") if sf_arn else None
    lam = session.client("lambda")

    # Discover repos
    if args.repo:
        repo_prefix = f"raw/{args.repo.strip('/')}/"
        repos = [repo_prefix]
    else:
        repos = list_raw_repos(s3, bucket)

    if not repos:
        print(f"No sub-prefixes found under s3://{bucket}/raw/ — nothing to ingest.")
        sys.exit(0)

    print(f"Stack  : {args.stack_name}")
    print(f"Bucket : {bucket}")
    print(f"Mode   : {'Step Functions' if sf_arn else 'Direct Lambda invocation'}")
    print(f"Repos  : {len(repos)}")
    for r in repos:
        print(f"  {r}")
    print()

    executions: list[tuple[str, str]] = []  # (repo_prefix, exec_arn)
    results: list[tuple[str, str]] = []     # (repo_prefix, status)

    for repo_prefix in repos:
        if sf_arn and sf:
            exec_arn = start_via_step_functions(sf, sf_arn, bucket, repo_prefix)
            print(f"Started execution for {repo_prefix}")
            print(f"  {exec_arn}")
            executions.append((repo_prefix, exec_arn))

            if args.wait and not args.parallel:
                status = poll_execution(sf, exec_arn, repo_prefix, args.poll_interval)
                results.append((repo_prefix, status))
        else:
            # Fallback: invoke Lambda directly (synchronous, no parallel support)
            print(f"Invoking preprocessor for {repo_prefix} ... ", end="", flush=True)
            resp = start_via_lambda(lam, lambda_name, bucket, repo_prefix)
            status = resp.get("status", "unknown")
            chunks = resp.get("chunks_written", 0)
            files = resp.get("files_processed", 0)
            errors = resp.get("errors", [])
            print(f"{status}  ({files} files, {chunks} chunks)")
            if errors:
                for e in errors:
                    print(f"  ERROR: {e}")
            results.append((repo_prefix, status))

    # Parallel wait: all executions started, now poll
    if args.parallel and sf and executions:
        print("\nPolling all executions...")
        pending = list(executions)
        while pending:
            still_pending = []
            for repo_prefix, exec_arn in pending:
                resp = sf.describe_execution(executionArn=exec_arn)
                status = resp["status"]
                if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"):
                    print(f"  DONE  {repo_prefix:<35} {status}")
                    results.append((repo_prefix, status))
                else:
                    still_pending.append((repo_prefix, exec_arn))
            if still_pending:
                pending = still_pending
                time.sleep(args.poll_interval)
            else:
                break

    # Summary
    print(f"\n{'─' * 60}")
    print(f"{'Repo':<40} {'Result'}")
    print(f"{'─' * 60}")
    failed = 0
    for repo_prefix, status in results:
        ok = status in ("SUCCEEDED", "success", "partial")
        icon = "✓" if status in ("SUCCEEDED", "success") else ("~" if status == "partial" else "✗")
        print(f"  {icon}  {repo_prefix:<38} {status}")
        if not ok:
            failed += 1

    if executions and not results:
        print("  Executions started (use --wait to poll for completion)")

    print(f"{'─' * 60}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
