#!/usr/bin/env python3
"""
ExpertiseRAG – Start Bedrock Ingestion Job (standalone boto3 script)

Usage:
    python scripts/start_ingestion.py \\
        --knowledge-base-id <KB_ID> \\
        --data-source-id <DS_ID> \\
        [--wait] \\
        [--region us-east-1]

This script demonstrates the Bedrock Agent `start_ingestion_job` and
`get_ingestion_job` API calls used inside the IngestionTriggerFunction Lambda.

Prerequisites:
    pip install boto3
    aws configure  # or set AWS_PROFILE / AWS_ACCESS_KEY_ID etc.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime

import boto3
from botocore.exceptions import ClientError


def start_ingestion_job(
    client,
    knowledge_base_id: str,
    data_source_id: str,
    description: str = "",
) -> dict:
    """
    Start a Bedrock Knowledge Base ingestion job.

    Idempotent: if a job is already running (ConflictException),
    lists the most recent job instead.
    """
    client_token = str(uuid.uuid4())
    try:
        response = client.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            clientToken=client_token,
            description=description or "ExpertiseRAG manual ingestion",
        )
        return response["ingestionJob"]
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ConflictException":
            print("⚠ Ingestion job already running – retrieving latest job status...")
            return get_latest_ingestion_job(client, knowledge_base_id, data_source_id)
        raise


def get_ingestion_job(
    client,
    knowledge_base_id: str,
    data_source_id: str,
    ingestion_job_id: str,
) -> dict:
    """Retrieve current status of a specific ingestion job."""
    response = client.get_ingestion_job(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=data_source_id,
        ingestionJobId=ingestion_job_id,
    )
    return response["ingestionJob"]


def get_latest_ingestion_job(
    client,
    knowledge_base_id: str,
    data_source_id: str,
) -> dict:
    """Return the most recently started ingestion job."""
    response = client.list_ingestion_jobs(
        knowledgeBaseId=knowledge_base_id,
        dataSourceId=data_source_id,
        sortBy={"attribute": "STARTED_AT", "order": "DESCENDING"},
        maxResults=1,
    )
    jobs = response.get("ingestionJobSummaries", [])
    if not jobs:
        raise RuntimeError("No ingestion jobs found for this knowledge base / data source")
    job_id = jobs[0]["ingestionJobId"]
    return get_ingestion_job(client, knowledge_base_id, data_source_id, job_id)


def wait_for_completion(
    client,
    knowledge_base_id: str,
    data_source_id: str,
    ingestion_job_id: str,
    poll_interval: int = 15,
    timeout: int = 1800,  # 30 minutes
) -> dict:
    """
    Poll ingestion job until COMPLETE, FAILED, or STOPPED.
    Prints status updates to stdout.
    """
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        if elapsed >= timeout:
            print(f"\n✗ Timeout after {timeout}s – last known status below")
            break

        job = get_ingestion_job(client, knowledge_base_id, data_source_id, ingestion_job_id)
        status = job.get("status", "UNKNOWN")
        stats = job.get("statistics", {})
        updated = job.get("updatedAt", "")
        updated_str = str(updated)[:19] if updated else ""

        print(
            f"\r[{elapsed:4d}s] Status: {status:<15} "
            f"Scanned: {stats.get('numberOfDocumentsScanned', 0):>6} "
            f"Indexed: {stats.get('numberOfNewDocumentsIndexed', 0):>6} "
            f"Failed: {stats.get('numberOfDocumentsFailed', 0):>4}  "
            f"Updated: {updated_str}",
            end="",
            flush=True,
        )

        if status in ("COMPLETE", "FAILED", "STOPPED"):
            print()  # newline after progress
            return job

        time.sleep(poll_interval)

    print()
    return get_ingestion_job(client, knowledge_base_id, data_source_id, ingestion_job_id)


def print_job_summary(job: dict) -> None:
    """Print a formatted summary of the ingestion job result."""
    status = job.get("status", "UNKNOWN")
    stats = job.get("statistics", {})
    failure_reasons = job.get("failureReasons", [])

    icon = "✓" if status == "COMPLETE" else "✗"
    print(f"\n{icon} Ingestion job: {job.get('ingestionJobId', 'unknown')}")
    print(f"  Status    : {status}")
    print(f"  Started   : {job.get('startedAt', '')}")
    print(f"  Updated   : {job.get('updatedAt', '')}")
    print("\n  Statistics:")
    for key, label in [
        ("numberOfDocumentsScanned", "Documents scanned"),
        ("numberOfNewDocumentsIndexed", "New docs indexed"),
        ("numberOfModifiedDocumentsIndexed", "Modified docs indexed"),
        ("numberOfDocumentsDeleted", "Documents deleted"),
        ("numberOfDocumentsFailed", "Documents failed"),
    ]:
        print(f"    {label:<30}: {stats.get(key, 0)}")

    if failure_reasons:
        print("\n  Failure reasons:")
        for reason in failure_reasons:
            print(f"    - {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start and optionally wait for a Bedrock Knowledge Base ingestion job"
    )
    parser.add_argument(
        "--knowledge-base-id", "-k",
        required=True,
        help="Bedrock Knowledge Base ID (e.g. KBID1234567890)"
    )
    parser.add_argument(
        "--data-source-id", "-d",
        required=True,
        help="Bedrock Data Source ID (e.g. DSID1234567890)"
    )
    parser.add_argument(
        "--wait", "-w",
        action="store_true",
        help="Poll until the ingestion job completes"
    )
    parser.add_argument(
        "--region", "-r",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--profile", "-p",
        default=None,
        help="AWS profile name"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Poll interval in seconds when --wait is set (default: 15)"
    )
    args = parser.parse_args()

    # Create boto3 session
    session = boto3.Session(
        region_name=args.region,
        profile_name=args.profile,
    )
    client = session.client("bedrock-agent")

    print(f"Starting ingestion job...")
    print(f"  Knowledge Base: {args.knowledge_base_id}")
    print(f"  Data Source   : {args.data_source_id}")
    print(f"  Region        : {args.region}")

    try:
        job = start_ingestion_job(
            client,
            knowledge_base_id=args.knowledge_base_id,
            data_source_id=args.data_source_id,
        )
        job_id = job.get("ingestionJobId", "")
        print(f"\n✓ Job started: {job_id} | initial status: {job.get('status')}")

        if args.wait and job_id:
            print(f"\nPolling every {args.poll_interval}s (Ctrl+C to stop)...")
            job = wait_for_completion(
                client,
                knowledge_base_id=args.knowledge_base_id,
                data_source_id=args.data_source_id,
                ingestion_job_id=job_id,
                poll_interval=args.poll_interval,
            )

        print_job_summary(job)

        if job.get("status") != "COMPLETE":
            sys.exit(1)

    except ClientError as exc:
        print(f"\n✗ AWS API error: {exc.response['Error']['Code']}: {exc.response['Error']['Message']}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted – job may still be running in the background")
        sys.exit(130)


if __name__ == "__main__":
    main()
