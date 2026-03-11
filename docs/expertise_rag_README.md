# ExpertiseRAG

**AWS-native GraphRAG platform for deep expertise queries**

ExpertiseRAG ingests Git repositories, architecture documents, research notes,
and resume content from S3, then uses Amazon Bedrock Knowledge Bases with
Neptune Analytics GraphRAG to answer deep, evidence-backed questions about a
developer's expertise.

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

## Architecture

```
S3 (raw/ uploads)
    ↓ S3 event
Preprocessor Lambda
  • Markdown / YAML / PlantUML extraction
  • Graph entity + edge construction
    ↓ writes derived/ artifacts
S3 (derived/ artifacts)
    ↓ Bedrock KB ingestion
Bedrock Knowledge Base
  • Titan Embeddings V2 (1024-dim)
  • Hierarchical chunking
  • Neptune Analytics vector/graph store
    ↑ query time
API Gateway → Query API Lambda
  • Question classification
  • Bedrock Retrieve (HYBRID)
  • Neptune Analytics graph expansion
  • Bedrock Converse synthesis
  • Evidence-weighted answer
```

Full architecture documentation: `docs/architecture.md`

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
- **Anthropic Claude 3 Haiku** (used for parsing)
- Your chosen **generation model** (Claude 3.5 Sonnet recommended)

### 2. Deploy

```bash
# Clone the repo (if not already)
git clone <repo-url> && cd ContextWeave

# Build Lambda packages
sam build --template template.yaml

# Deploy interactively (first time)
sam deploy --guided --template template.yaml

# Or use samconfig.toml after first deploy
sam deploy
```

**SAM deploy prompts**:

| Parameter | Recommended value |
|-----------|------------------|
| `Stack Name` | `expertise-rag-dev` |
| `AWS Region` | `us-east-1` |
| `Environment` | `dev` |
| `NeptuneVectorDimension` | `1024` (do not change) |
| `NeptuneProvisionedMemory` | `16` (minimum) |
| `LogRetentionDays` | `30` |
| `EnableStepFunctions` | `true` |

### 3. Configure Generation Model

After deployment, set the generation model on the Query API Lambda:

```bash
# Option A: Update via AWS CLI
aws lambda update-function-configuration \
  --function-name expertise-rag-query-api-dev \
  --environment "Variables={
    GENERATION_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0,
    KNOWLEDGE_BASE_ID=$(aws cloudformation describe-stacks \
      --stack-name expertise-rag-dev \
      --query 'Stacks[0].Outputs[?OutputKey==`KnowledgeBaseId`].OutputValue' \
      --output text),
    NEPTUNE_GRAPH_ID=$(aws cloudformation describe-stacks \
      --stack-name expertise-rag-dev \
      --query 'Stacks[0].Outputs[?OutputKey==`NeptuneGraphIdentifier`].OutputValue' \
      --output text),
    ARTIFACTS_BUCKET=$(aws cloudformation describe-stacks \
      --stack-name expertise-rag-dev \
      --query 'Stacks[0].Outputs[?OutputKey==`ArtifactsBucketName`].OutputValue' \
      --output text),
    ENVIRONMENT=dev
  }"

# Option B: Add to samconfig.toml parameter_overrides
# GenerationModelId=anthropic.claude-3-5-sonnet-20241022-v2:0
```

> See `src/query_api/synthesizer.py` for the `TODO_GENERATION_MODEL_ID` marker.

### 4. Get Stack Outputs

```bash
aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs' \
  --output table
```

Key outputs:

| Output Key | Description |
|------------|-------------|
| `ArtifactsBucketName` | S3 bucket for uploads |
| `QueryExpertiseURL` | Full POST /query-expertise URL |
| `KnowledgeBaseId` | Bedrock Knowledge Base ID |
| `DataSourceId` | Bedrock Data Source ID |
| `NeptuneGraphIdentifier` | Neptune Analytics graph ID |

---

## Uploading Repository Content

### Option A: Use the upload script

```bash
# Upload this repo's docs
python scripts/upload_repo.py \
  --bucket expertise-rag-artifacts-$(aws sts get-caller-identity --query Account --output text)-dev \
  --repo-name expertise-rag-platform \
  --source-dir .

# Dry run first
python scripts/upload_repo.py \
  --bucket $BUCKET \
  --repo-name my-project \
  --source-dir /path/to/project \
  --dry-run
```

### Option B: Manual S3 upload

```bash
BUCKET=expertise-rag-artifacts-<account-id>-dev
REPO=my-saas-platform

aws s3 cp architecture.md s3://$BUCKET/raw/$REPO/architecture.md
aws s3 cp CLAUDE.md        s3://$BUCKET/raw/$REPO/CLAUDE.md
aws s3 cp repo-signals.yaml s3://$BUCKET/raw/$REPO/repo-signals.yaml
aws s3 cp docs/c4_diagram.puml s3://$BUCKET/raw/$REPO/docs/c4_diagram.puml
```

**Recommended files per repo**:

| File | Purpose | Evidence Weight |
|------|---------|----------------|
| `architecture.md` | Architecture decisions | 1.0 (highest) |
| `CLAUDE.md` | Project context and tech stack | 1.0 (highest) |
| `repo-signals.yaml` | Structured expertise manifest | 0.7 |
| `*.puml` / `*.plantuml` | Architecture diagrams | 0.8 |
| `README.md` | Project overview | 0.6 |
| Articles / blog posts | Published thinking | 0.5 |
| Resume PDF/MD | Supporting context | 0.3 |

See `examples/repo_manifest_schema.json` for the `repo-signals.yaml` schema.

---

## Triggering Ingestion

The preprocessing Lambda runs automatically on S3 upload. After preprocessing
completes, trigger a Bedrock KB ingestion job:

```bash
# Get IDs from stack outputs
KB_ID=$(aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`KnowledgeBaseId`].OutputValue' \
  --output text)

DS_ID=$(aws cloudformation describe-stacks \
  --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`DataSourceId`].OutputValue' \
  --output text)

# Start ingestion and wait for completion
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

### Using the API endpoint

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

### Using the Python script

```bash
# Full API call (requires deployed stack)
python scripts/query_expertise.py api \
  --endpoint https://<api-id>.execute-api.us-east-1.amazonaws.com/dev \
  --question "What architecture patterns does this developer use?"

# Direct Bedrock retrieve (debug – no synthesis)
python scripts/query_expertise.py bedrock \
  --knowledge-base-id $KB_ID \
  --question "What AWS services has this developer used?" \
  --top-k 10
```

### Sample questions

```bash
# Skill depth
"What is this developer's level of expertise with Amazon Bedrock?"

# Architecture
"What architecture patterns has this developer applied repeatedly across projects?"

# Project evidence
"What production AWS systems has this developer built from scratch?"

# Comparison
"Does this developer prefer serverless or container-based architectures?"

# Technology
"What is this developer's primary backend language and why?"
```

---

## Local Testing

### Test Preprocessor Lambda

```bash
sam local invoke PreprocessorFunction \
  --event events/preprocessor_direct_event.json \
  --env-vars <(echo '{"PreprocessorFunction": {
    "ARTIFACTS_BUCKET": "local-test-bucket",
    "AWS_REGION": "us-east-1"
  }}')
```

### Test Query API Lambda

```bash
sam local invoke QueryAPIFunction \
  --event events/api_query_event.json \
  --env-vars <(echo '{"QueryAPIFunction": {
    "KNOWLEDGE_BASE_ID": "KBID1234567890",
    "NEPTUNE_GRAPH_ID": "g-ABCDEF123456",
    "GENERATION_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "ARTIFACTS_BUCKET": "expertise-rag-artifacts-123456789012-dev"
  }}')
```

### Unit test preprocessor extractors

```bash
cd src
python -c "
from preprocessor.extractors import extract_markdown, extract_plantuml, extract_yaml

# Test markdown
result = extract_markdown('# Architecture\nUses AWS Lambda and Amazon Bedrock.', 'architecture.md', 1.0)
print('Signals:', [s['value'] for s in result['expertise_signals']])

# Test PlantUML
puml = '@startuml\ncomponent Lambda\ncomponent Bedrock\nLambda --> Bedrock : invoke\n@enduml'
result = extract_plantuml(puml, 'arch.puml', 0.8)
print('PlantUML summary:', result['summary'])

# Test YAML
yaml_content = 'aws_services:\n  - name: Amazon Bedrock\n  - name: AWS Lambda\n'
result = extract_yaml(yaml_content, 'repo-signals.yaml', 0.7)
print('YAML signals:', [s['value'] for s in result['expertise_signals']])
"
```

---

## Project Structure

```
.
├── template.yaml                  # SAM/CloudFormation template
├── samconfig.toml                 # SAM deployment configuration
├── src/
│   ├── preprocessor/
│   │   ├── __init__.py
│   │   ├── handler.py            # Lambda entry point
│   │   ├── extractors.py         # Markdown/YAML/PlantUML extraction
│   │   ├── graph_builder.py      # Graph entity/edge construction
│   │   └── requirements.txt
│   ├── query_api/
│   │   ├── __init__.py
│   │   ├── handler.py            # Lambda entry point (POST /query-expertise)
│   │   ├── retriever.py          # Bedrock KB Retrieve with source weighting
│   │   ├── graph_expander.py     # Neptune Analytics expansion queries
│   │   ├── synthesizer.py        # Bedrock Converse answer synthesis
│   │   └── requirements.txt
│   ├── ingestion_trigger/
│   │   ├── __init__.py
│   │   ├── handler.py            # Custom resource + Step Functions task
│   │   └── requirements.txt
│   └── shared/
│       ├── __init__.py
│       └── models.py             # Shared domain models and constants
├── events/
│   ├── s3_upload_event.json      # S3 notification event
│   ├── api_query_event.json      # API Gateway HTTP event
│   ├── step_functions_input.json # Step Functions execution input
│   └── preprocessor_direct_event.json
├── examples/
│   ├── repo_manifest_schema.json  # repo-signals.yaml JSON Schema
│   ├── graph_entities.json        # Sample graph nodes
│   ├── graph_edges.json           # Sample graph edges
│   ├── api_request.json           # Sample API request
│   └── api_response.json          # Sample API response
├── scripts/
│   ├── upload_repo.py            # Upload local repo to S3
│   ├── start_ingestion.py        # Start + monitor Bedrock ingestion job
│   └── query_expertise.py        # Query via Bedrock or HTTP API
└── docs/
    ├── architecture.md           # System architecture and component details
    ├── graph_schema.md           # Graph node/edge schema and example queries
    └── api_reference.md          # Full API reference
```

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
| API Gateway | Per request | Minimal |
| Step Functions | Per state transition | Minimal |

**Estimated minimum monthly cost** (dev, light usage): ~$200–350/month
(dominated by Neptune Analytics provisioned memory)

To reduce costs in development:
- Set `NeptuneProvisionedMemory` to `16` (minimum)
- Set `EnableStepFunctions` to `false`
- Use smaller generation models (Haiku)

---

## TODOs

- [ ] **Set `GENERATION_MODEL_ID`** in `src/query_api/synthesizer.py` and Lambda env vars
- [ ] **Enable Bedrock model access** for Titan Embeddings V2 + chosen generation model
- [ ] **Populate `PERSON_NAME`** environment variable on PreprocessorFunction
- [ ] **Add NER** to `graph_expander.py` for better entity extraction from retrieved chunks
- [ ] **Add streaming** to Query API for lower perceived latency
- [ ] **Add authentication** (Cognito or API key) to API Gateway if public-facing
- [ ] **Add CloudWatch alarms** for Lambda errors and API latency
- [ ] **Tune chunking** parameters once you observe retrieval quality

---

## Security

- All S3 data encrypted with customer-managed KMS key
- S3 bucket policy enforces SSL
- All IAM roles are least-privilege
- Neptune Analytics uses IAM authentication
- CloudWatch logs retained per `LogRetentionDays` parameter
- No public S3 access

---

## License

See [LICENSE](LICENSE).
