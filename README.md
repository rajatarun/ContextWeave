# ContextWeave / ExpertiseRAG

**AWS-native GraphRAG platform for deep expertise queries**

ExpertiseRAG ingests Git repositories, architecture documents, research notes,
and resume content from S3, then uses Amazon Bedrock Knowledge Bases with
Neptune Analytics GraphRAG to answer deep, evidence-backed questions about a
developer's professional expertise.

```
POST /query-expertise
{ "question": "What AWS services has this developer built production systems with?" }

→ {
    "answer": "Based on architecture.md and CLAUDE.md (authoritative sources)...",
    "inferredSkills": ["Amazon Bedrock", "Neptune Analytics", "AWS Lambda"],
    "repeatedPatterns": ["serverless", "event-driven", "infrastructure-as-code"],
    "confidence": 0.94
  }
```

---

## Architecture Overview

```
S3 (raw/ uploads)
    ↓ S3 event
Preprocessor Lambda (Python 3.12)
  • Markdown / YAML / PlantUML extraction
  • Graph entity + edge construction
    ↓ writes derived/ artifacts
S3 (derived/ artifacts)
    ↓ Step Functions → Bedrock KB ingestion
Bedrock Knowledge Base
  • Titan Embeddings V2 (1024-dim)
  • Hierarchical chunking (1500 / 300 tokens)
  • Neptune Analytics vector + graph store
    ↑ query time
API Gateway → Query API Lambda
  • Question classification
  • Bedrock Retrieve (HYBRID)
  • Neptune Analytics graph expansion
  • Bedrock Converse synthesis
  • Evidence-weighted answer
```

| Layer | Service | Role |
|-------|---------|------|
| Storage | Amazon S3 | Raw uploads (`raw/`) and derived artifacts (`derived/`) |
| Preprocessing | AWS Lambda (Python 3.12) | Text extraction, graph entity construction |
| Knowledge Graph | Amazon Neptune Analytics | Vector + graph hybrid store (openCypher, 1024-dim) |
| Embeddings | Amazon Titan Text Embeddings V2 | 1024-dimensional semantic embeddings |
| Knowledge Base | Amazon Bedrock Knowledge Base | Hierarchical chunking, hybrid search |
| Query API | AWS Lambda + API Gateway HTTP v2 | POST /query-expertise |
| Orchestration | AWS Step Functions | Preprocess → StartIngestionJob → Poll |
| Encryption | AWS KMS | SSE-KMS on S3; key rotates annually |
| Observability | AWS X-Ray + CloudWatch | Tracing on all Lambdas and Step Functions |
| IaC | AWS SAM (`template.yaml`) | Full infrastructure as code |
| CI/CD | GitHub Actions | OIDC → SAM validate → build → deploy |

Full architecture documentation: [`docs/architecture.md`](docs/architecture.md)

---

## Quick Start

### Prerequisites

- AWS CLI configured with appropriate permissions
- AWS SAM CLI >= 1.100
- Python >= 3.12
- Bedrock model access enabled (see below)

### 1. Enable Bedrock Models

In the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess), enable:
- **Amazon Titan Text Embeddings V2** (required for embeddings)
- **Anthropic Claude 3 Haiku** (used for document parsing during ingestion)
- Your chosen **generation model** (Claude 3.5 Sonnet recommended)

### 2. Deploy

```bash
# Clone the repo
git clone https://github.com/rajatarun/ContextWeave.git && cd ContextWeave

# Build Lambda packages
sam build --template template.yaml --parallel --cached

# Deploy interactively (first time)
sam deploy --guided --template template.yaml

# Or use samconfig.toml after first deploy
sam deploy
```

**SAM deploy parameters**:

| Parameter | Recommended value |
|-----------|------------------|
| `Stack Name` | `expertise-rag-dev` |
| `AWS Region` | `us-east-1` |
| `Environment` | `dev` |
| `NeptuneVectorDimension` | `1024` (do not change) |
| `NeptuneProvisionedMemory` | `16` (minimum) |
| `LogRetentionDays` | `30` |
| `EnableStepFunctions` | `true` |

### 3. Get Stack Outputs

```bash
aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs' \
  --output table
```

| Output Key | Description |
|------------|-------------|
| `ArtifactsBucketName` | S3 bucket for uploads |
| `QueryExpertiseURL` | Full POST /query-expertise URL |
| `KnowledgeBaseId` | Bedrock Knowledge Base ID |
| `DataSourceId` | Bedrock Data Source ID |
| `NeptuneGraphIdentifier` | Neptune Analytics graph ID |

---

## Uploading Repository Content

### Option A: Upload script

```bash
python scripts/upload_repo.py \
  --bucket expertise-rag-artifacts-$(aws sts get-caller-identity --query Account --output text)-dev \
  --repo-name my-project \
  --source-dir /path/to/project

# Dry run first
python scripts/upload_repo.py \
  --bucket $BUCKET --repo-name my-project --source-dir . --dry-run
```

### Option B: Manual S3 upload

```bash
BUCKET=expertise-rag-artifacts-<account-id>-dev
REPO=my-project

aws s3 cp architecture.md     s3://$BUCKET/raw/$REPO/architecture.md
aws s3 cp CLAUDE.md            s3://$BUCKET/raw/$REPO/CLAUDE.md
aws s3 cp repo-signals.yaml   s3://$BUCKET/raw/$REPO/repo-signals.yaml
aws s3 cp docs/c4.puml        s3://$BUCKET/raw/$REPO/docs/c4.puml
```

**Evidence weighting by file type**:

| File | Purpose | Evidence Weight |
|------|---------|----------------|
| `architecture.md`, `CLAUDE.md` | Authoritative architecture docs | 1.0 (highest) |
| `*.puml` / `*.plantuml` | Architecture diagrams | 0.8 |
| `repo-signals.yaml` | Structured expertise manifest | 0.7 |
| `README.md`, code | Implementation evidence | 0.6 |
| Resume PDF/MD | Supporting context | 0.3 |

See `examples/repo_manifest_schema.json` for the `repo-signals.yaml` schema.

---

## Triggering Ingestion

Preprocessing runs automatically on S3 upload. To trigger a Bedrock KB ingestion job:

```bash
KB_ID=$(aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`KnowledgeBaseId`].OutputValue' \
  --output text)

DS_ID=$(aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`DataSourceId`].OutputValue' \
  --output text)

python scripts/start_ingestion.py \
  --knowledge-base-id $KB_ID \
  --data-source-id $DS_ID \
  --wait
```

### Via Step Functions (recommended for large repos)

```bash
SFN_ARN=$(aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`IngestionStateMachineArn`].OutputValue' \
  --output text)

aws stepfunctions start-execution \
  --state-machine-arn $SFN_ARN \
  --input file://events/step_functions_input.json
```

---

## Querying

```bash
API_URL=$(aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`QueryExpertiseURL`].OutputValue' \
  --output text)

curl -s -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{"question": "What AWS services has this developer built production systems with?", "topK": 10}' \
  | jq .
```

**Sample questions**:

```
"What is this developer's level of expertise with Amazon Bedrock?"
"What architecture patterns has this developer applied repeatedly across projects?"
"What production AWS systems has this developer built from scratch?"
"Does this developer prefer serverless or container-based architectures?"
```

---

## Local Testing

```bash
# Test Preprocessor Lambda
sam local invoke PreprocessorFunction \
  --event events/preprocessor_direct_event.json \
  --env-vars <(echo '{"PreprocessorFunction": {
    "ARTIFACTS_BUCKET": "local-test-bucket",
    "AWS_REGION": "us-east-1"
  }}')

# Test Query API Lambda
sam local invoke QueryAPIFunction \
  --event events/api_query_event.json \
  --env-vars <(echo '{"QueryAPIFunction": {
    "KNOWLEDGE_BASE_ID": "KBID1234567890",
    "NEPTUNE_GRAPH_ID": "g-ABCDEF123456",
    "GENERATION_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "ARTIFACTS_BUCKET": "expertise-rag-artifacts-123456789012-dev"
  }}')
```

---

## Project Structure

```
.
├── template.yaml                  # SAM/CloudFormation template
├── samconfig.toml                 # SAM deployment configuration
├── src/
│   ├── preprocessor/
│   │   ├── handler.py            # Lambda entry point (S3 trigger)
│   │   ├── extractors.py         # Markdown/YAML/PlantUML extraction
│   │   ├── graph_builder.py      # Graph entity/edge construction
│   │   └── requirements.txt
│   ├── query_api/
│   │   ├── handler.py            # Lambda entry point (POST /query-expertise)
│   │   ├── retriever.py          # Bedrock KB Retrieve with source weighting
│   │   ├── graph_expander.py     # Neptune Analytics graph expansion
│   │   ├── synthesizer.py        # Bedrock Converse answer synthesis
│   │   └── requirements.txt
│   ├── ingestion_trigger/
│   │   ├── handler.py            # CloudFormation custom resource + Step Functions
│   │   └── requirements.txt
│   └── shared/
│       └── models.py             # Shared domain models and constants
├── events/                        # Sample Lambda event payloads
├── examples/                      # Sample graph/API inputs and outputs
├── scripts/
│   ├── upload_repo.py            # Upload local repo to S3
│   ├── start_ingestion.py        # Start + monitor Bedrock ingestion job
│   └── query_expertise.py        # Query via Bedrock or HTTP API
└── docs/
    ├── architecture.md           # System architecture and design decisions
    ├── graph_schema.md           # Graph node/edge schema and example queries
    ├── api_reference.md          # Full API reference
    ├── c4.puml                   # C4 system context + container diagram
    └── aws-infrastructure.puml   # AWS service topology diagram
```

---

## CI/CD

Push to `main`/`master` triggers an automatic deploy to the `dev` environment via GitHub Actions. The workflow uses OIDC — no long-lived AWS credentials are stored in GitHub secrets.

```
Push → sam validate → sam build → sam deploy → S3 upload (signal files)
```

Manual deploys to `staging` or `prod` are triggered via `workflow_dispatch`.

---

## Cost Considerations

| Service | Cost driver | Notes |
|---------|-------------|-------|
| Neptune Analytics | Provisioned memory (min 16 GB) | Largest ongoing cost; ~$0.50/GB-hr |
| Amazon Bedrock KB | Per ingestion job + storage | Pay per use |
| Titan Embeddings V2 | Per 1000 tokens | ~$0.0002 / 1K tokens |
| Bedrock Converse | Per input/output token | Depends on model |
| AWS Lambda | Per invocation + duration | Typically cents |
| Amazon S3 | Storage + requests | Minimal |

**Estimated minimum monthly cost** (dev, light usage): ~$200–350/month
(dominated by Neptune Analytics provisioned memory)

To reduce costs in development:
- Set `NeptuneProvisionedMemory` to `16` (minimum)
- Set `EnableStepFunctions` to `false`
- Use smaller generation models (Claude Haiku)

---

## Security

- All S3 data encrypted with a customer-managed KMS key (auto-rotates annually)
- S3 bucket policy enforces SSL-only access; no public S3 access
- All IAM roles are least-privilege
- Neptune Analytics uses IAM authentication
- GitHub Actions deploys via OIDC — no long-lived access keys
- CloudWatch logs retained per `LogRetentionDays` parameter

---

## License

See [LICENSE](LICENSE).
