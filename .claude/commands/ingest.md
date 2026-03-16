# Trigger document ingestion workflow

Upload repo signal files to S3 and trigger the Bedrock Knowledge Base ingestion pipeline.

**Step 1 — Get the S3 artifacts bucket name**:
```bash
aws cloudformation describe-stacks --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ArtifactsBucket`].OutputValue' \
  --output text
```

**Step 2 — Upload signal files** using the upload script:
```bash
python scripts/upload_repo.py
```

**Step 3 — Trigger ingestion** using the ingestion script:
```bash
python scripts/start_ingestion.py
```

**Step 4 — Monitor the Step Functions execution**:
```bash
aws stepfunctions list-executions \
  --state-machine-arn $(aws cloudformation describe-stacks --stack-name expertise-rag-dev \
    --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' --output text) \
  --status-filter RUNNING
```

After ingestion completes, the Neptune graph will contain updated DocumentType, ChunkingStrategy, and expertise nodes. Confirm the ingestion job status and report how many documents were indexed.

**Signal files ingested** (from `raw/contextweave/`):
- `CLAUDE.md` (weight 1.0)
- `docs/architecture.md` (weight 1.0)
- `repo-signals.yaml` (weight 0.7)
- `docs/c4.puml` (weight 0.8)
- `docs/aws-infrastructure.puml` (weight 0.8)
