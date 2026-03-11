#!/usr/bin/env python3
"""
ExpertiseRAG – Upload Repository Files to S3

Walks a local directory and uploads all supported files to the
raw/<repo-name>/ prefix in the ExpertiseRAG S3 bucket.

Supported files:
  *.md, *.markdown, *.yaml, *.yml, *.puml, *.plantuml, *.pu, *.txt, *.rst

Usage:
    python scripts/upload_repo.py \\
        --bucket expertise-rag-artifacts-123456789012-dev \\
        --repo-name my-saas-platform \\
        --source-dir /path/to/local/repo \\
        [--include-hidden] \\
        [--dry-run]

    # Or use SAM output directly:
    BUCKET=$(aws cloudformation describe-stacks \\
        --stack-name expertise-rag-dev \\
        --query 'Stacks[0].Outputs[?OutputKey==`ArtifactsBucketName`].OutputValue' \\
        --output text)

    python scripts/upload_repo.py --bucket $BUCKET --repo-name my-repo --source-dir .
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

SUPPORTED_EXTENSIONS = {
    ".md", ".markdown", ".mdx",
    ".yaml", ".yml",
    ".puml", ".plantuml", ".pu", ".wsd",
    ".txt", ".rst",
}

SKIP_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".tox",
}


def upload_repo(
    bucket: str,
    repo_name: str,
    source_dir: str,
    include_hidden: bool = False,
    dry_run: bool = False,
    region: str = "us-east-1",
    profile: str | None = None,
) -> int:
    """Upload supported files to S3. Returns count of uploaded files."""
    source_path = Path(source_dir).resolve()
    if not source_path.exists():
        print(f"✗ Source directory not found: {source_path}")
        return 0

    session = boto3.Session(region_name=region, profile_name=profile)
    s3 = session.client("s3")

    uploaded = 0
    skipped = 0
    errors = 0

    print(f"Scanning {source_path}...")
    print(f"Target: s3://{bucket}/raw/{repo_name}/\n")

    for path in sorted(source_path.rglob("*")):
        if not path.is_file():
            continue

        # Skip hidden files/dirs unless requested
        parts = path.relative_to(source_path).parts
        if not include_hidden and any(p.startswith(".") for p in parts):
            continue

        # Skip known build / dependency directories
        if any(p in SKIP_DIRS for p in parts):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue

        relative_key = str(path.relative_to(source_path))
        s3_key = f"raw/{repo_name}/{relative_key}"

        size = path.stat().st_size
        size_str = f"{size:>8,} bytes"

        if dry_run:
            print(f"  [DRY RUN] Would upload {relative_key} ({size_str}) → {s3_key}")
            uploaded += 1
            continue

        try:
            s3.upload_file(
                str(path),
                bucket,
                s3_key,
                ExtraArgs={"ServerSideEncryption": "aws:kms"},
            )
            print(f"  ✓ {relative_key} ({size_str})")
            uploaded += 1
        except ClientError as exc:
            print(f"  ✗ {relative_key}: {exc.response['Error']['Message']}")
            errors += 1

    print(f"\n{'DRY RUN – ' if dry_run else ''}Done:")
    print(f"  Uploaded : {uploaded}")
    print(f"  Skipped  : {skipped} (unsupported extension)")
    print(f"  Errors   : {errors}")

    if uploaded > 0 and not dry_run:
        print(
            f"\n💡 Preprocessing Lambda will run automatically via S3 event trigger.\n"
            f"   To manually trigger ingestion afterward:\n"
            f"   python scripts/start_ingestion.py --knowledge-base-id <KB_ID> --data-source-id <DS_ID> --wait"
        )

    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload repository files to ExpertiseRAG S3 bucket")
    parser.add_argument("--bucket", "-b", required=True, help="S3 bucket name")
    parser.add_argument("--repo-name", "-n", required=True, help="Repository slug (used as S3 prefix)")
    parser.add_argument("--source-dir", "-s", default=".", help="Local directory to upload (default: .)")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files/directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    parser.add_argument("--region", "-r", default="us-east-1")
    parser.add_argument("--profile", "-p", default=None)
    args = parser.parse_args()

    count = upload_repo(
        bucket=args.bucket,
        repo_name=args.repo_name,
        source_dir=args.source_dir,
        include_hidden=args.include_hidden,
        dry_run=args.dry_run,
        region=args.region,
        profile=args.profile,
    )
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
