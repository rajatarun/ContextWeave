# ContextWeave / ExpertiseRAG

**AWS-native GraphRAG + CAG platform with adaptive RAG routing**

ExpertiseRAG ingests Git repositories, architecture documents, research notes,
and resume content from S3, then uses a knowledge graph (Memgraph) and a
PostgreSQL/pgvector chunk store to answer deep, evidence-backed questions about
a developer's professional expertise.

A **semantic response cache (CAG)** short-circuits the full RAG pipeline for
repeated or near-identical questions. An **agentic routing layer** classifies
every document at ingestion time, recommends the best chunking strategy, and at
query time consults a self-improving Memgraph routing graph to pick the optimal retrieval
strategy.

```
POST /query-expertise
{ "question": "What AWS services has this developer built production systems with?" }

→ {
    "answer": "Based on architecture.md and CLAUDE.md (authoritative sources)...",
    "inferredSkills": ["Amazon Bedrock", "Memgraph", "AWS Lambda"],
    "repeatedPatterns": ["serverless", "event-driven", "infrastructure-as-code"],
    "confidence": 0.94,
    "routingDecision": {
      "strategy": "graph_first",
      "questionType": "architecture",
      "strategyConfidence": 0.80,
      "graphExpansionForced": true
    },
    "cacheHit": false,
    "latencyMs": 1840
  }
```

---

## What's New — CAG + Adaptive RAG Routing

### Cache-Augmented Generation (CAG)

Before the RAG pipeline runs, the system checks a **semantic response cache**
backed by PostgreSQL/pgvector:

1. The question is embedded with Titan Text Embeddings V2 (1024-dim).
2. The `query_cache` table is searched for an unexpired entry with cosine
   similarity ≥ 0.95.
3. **Cache hit** → cached response returned immediately (`cacheHit: true`);
   the entire classify → route → retrieve → synthesize pipeline is skipped.
4. **Cache miss** → full pipeline runs; if confidence ≥ 0.5, the response
   is written to cache with a 7-day TTL.

Time-sensitive questions (`today`, `currently`, `latest`, etc.) bypass the cache.

### Adaptive RAG Routing

Every document is automatically classified into a **DocumentType** and assigned
the best **ChunkingStrategy** based on structural analysis:

| DocumentType | Detection heuristics | ChunkingStrategy |
|---|---|---|
| `technical_spec` | ≥3 headings + ≥2 AWS service signals | `hierarchical` (1500/300 tokens) |
| `narrative` | Long sentences, few headings, low code ratio | `sentence` |
| `structured_data` | `.yaml`/`.json` / key-value density | `fixed_256` |
| `code` | Source file extension / code ratio > 35% | `fixed_512` |
| `diagram_derived` | `.puml` extension / `@startuml` marker | `fixed_256` |

DocumentType and ChunkingStrategy nodes — and their relationships to each
Document node — are written directly into Memgraph.

### Self-improving routing feedback loop

A **RAGRouter** agent reads `EFFECTIVE_FOR` edge weights from Memgraph to select
the optimal retrieval strategy:

| Strategy | When chosen | What it does |
|---|---|---|
| `graph_first` | architecture, skill_depth | pgvector retrieval + forced Memgraph graph expansion |
| `hybrid` | comparison | pgvector + Memgraph vectors + keyword boost |
| `keyword_boosted` | project, credential | pgvector + keyword-overlap reranking |
| `semantic_search` | general | pgvector semantic search (baseline) |

After every query, the router updates edge weights in Memgraph:
- `confidence ≥ 0.70` → reinforce winning strategy (`+0.05`, cap 1.0)
- `confidence < 0.40` → penalise winning strategy (`−0.02`, floor 0.1)

No retraining. No manual tuning. The graph learns from every answered question.

### Test results (32 questions)
- `skill_depth` routing accuracy: **100%**
- Average answer confidence: **0.83**
- `graph_first` selected 16/32 times (highest prior weight: 0.80)

---

## Architecture Overview

```
S3 (raw/ uploads)
    ↓ S3 event
Preprocessor Lambda (Python 3.12)
  • Markdown / YAML / PlantUML extraction
  • routing_analyzer → DocumentType + ChunkingStrategy
  • Graph entity + edge construction (incl. routing metadata)
    ↓ writes derived/ artifacts
S3 (derived/ artifacts)
    ↓ Step Functions → chunk embedding → pgvector ingestion
PostgreSQL + pgvector
  • chunks table (Titan V2 1024-dim embeddings, ivfflat ANN)
  • query_cache table (CAG semantic response cache)
    ↑ query time
API Gateway → Query API Lambda
  0. embed question → check CAG cache (skip steps 1-8 on hit)
  1. classify_question()
  2. RAGRouter.select_strategy()   ← reads Memgraph EFFECTIVE_FOR edges
  3. retrieve_with_strategy()      ← pgvector ± Memgraph ± keyword boost
  4. deduplicate_chunks()
  5. expand_graph_context()        ← Memgraph openCypher traversal
  6. synthesize_answer()           ← Bedrock Converse
  7. write_cache()                 ← writes to query_cache if conf ≥ 0.5
  8. RAGRouter.update_feedback()   ← writes confidence back to Memgraph
```

| Layer | Service | Role |
|-------|---------|------|
| Storage | Amazon S3 | Raw uploads (`raw/`) and derived artifacts (`derived/`) |
| Preprocessing | AWS Lambda (Python 3.12) | Text extraction, document classification, graph construction |
| Expertise Graph | Memgraph | openCypher graph database — skills, patterns, AWS services |
| Vector Store | PostgreSQL + pgvector | Chunk embeddings (1024-dim) + CAG semantic response cache |
| Embeddings | Amazon Titan Text Embeddings V2 | 1024-dimensional semantic embeddings |
| Routing Graph | Memgraph (EC2) | EFFECTIVE_FOR routing weights + graph expansion |
| Query API | AWS Lambda + API Gateway HTTP v2 | POST /query-expertise with CAG + adaptive routing |
| MCP | API Gateway `POST /mcp` and `GET /mcp` | Streamable HTTP MCP server (protocol 2025-06-18) |
| Orchestration | AWS Step Functions | Preprocess → embed chunks → poll |
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
- Memgraph instance accessible (bolt://host:7687)
- PostgreSQL instance with pgvector extension

### 1. Enable Bedrock Models

In the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess), enable:
- **Amazon Titan Text Embeddings V2** (required for embeddings and CAG cache)
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
| `LogRetentionDays` | `30` |
| `EnableStepFunctions` | `true` |

On first deploy the `DbInitFunction`:
1. **Seeds the routing graph** in Memgraph with initial prior weights
2. **Initialises the pgvector schema** (`chunks` and `query_cache` tables)

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
| `McpEndpoint` | Streamable HTTP MCP URL (`POST /mcp`) |
| `MemgraphInstanceId` | Memgraph EC2 instance ID |

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

**Document type classification** (automatic at ingestion):

Each uploaded file is classified by the `routing_analyzer` module. The chosen
`ChunkingStrategy` is stored in Memgraph and surfaced in `processing_manifest.json`
under `routing_summary`.

See `examples/repo_manifest_schema.json` for the `repo-signals.yaml` schema.

---

## Triggering Ingestion

Preprocessing runs automatically on S3 upload. To trigger a full ingestion
(embed chunks into pgvector):

```bash
python scripts/start_ingestion.py \
  --stack-name expertise-rag-dev \
  --wait
```

### Seed routing graph manually

If you need to (re)seed the routing graph with initial prior weights:

```bash
aws lambda invoke \
  --function-name contextweave-rag-db-init-dev \
  --payload '{"action":"seed_routing"}' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
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

**Sample questions by type**:

```
# skill_depth → routed to graph_first (prior 0.80)
"How proficient is this developer with Amazon Bedrock?"
"What is the depth of their AWS Lambda experience?"

# architecture → routed to graph_first (prior 0.80)
"What architectural patterns does this developer prefer?"
"How is the knowledge graph structured in this system?"

# project → routed to keyword_boosted (prior 0.70)
"What projects has this developer built and deployed?"
"How was the Step Functions orchestration implemented?"

# comparison → routed to hybrid (prior 0.70)
"Why was Memgraph chosen for the expertise graph?"
"What are the trade-offs between CAG and RAG?"

# credential → routed to keyword_boosted (prior 0.80)
"What formal training or courses has this developer completed?"

# general → routed to semantic_search (prior 0.70)
"What problem does ExpertiseRAG solve?"
```

The response includes:
- `routingDecision` — strategy, question type, prior confidence, graph expansion flags
- `cacheHit` — `true` if served from CAG cache
- `latencyMs` — total wall-clock time

---

## MCP connector

The same API serves a remote MCP server on **Streamable HTTP**, protocol
**2025-06-18**. Clients POST JSON-RPC to `/mcp`. `GET /mcp` returns **405**
with `Allow: POST` — this server does not open a server-initiated SSE stream,
which the transport permits.

The endpoint is the `McpEndpoint` stack output:

```bash
aws cloudformation describe-stacks \
  --stack-name contextweave-rag-prod \
  --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='McpEndpoint'].OutputValue" \
  --output text
```

It looks like `https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/mcp`.
The prod API is the HTTP API in `template.yaml` (stack `contextweave-rag-prod`).

Every request must send `Authorization: Bearer <token>`. A missing or wrong
token is **401**. The check does not say which of those it was.

### Store the token

The template parameter `McpAuthTokenSecretName` is the **name** of the secret,
never the token. Do not commit the token and do not pass it as a parameter
override.

Secrets Manager (name does not start with `/`):

```bash
aws secretsmanager create-secret \
  --region us-east-1 \
  --name contextweave/prod/mcp-bearer-token \
  --description "Bearer token for the ContextWeave MCP endpoint" \
  --secret-string "$(openssl rand -base64 32)"
```

A JSON object is also accepted, with the token under `token`, `bearerToken`,
`bearer_token`, or `value`.

SSM SecureString (name starts with `/`):

```bash
aws ssm put-parameter \
  --region us-east-1 \
  --name /contextweave/prod/mcp-bearer-token \
  --type SecureString \
  --value "$(openssl rand -base64 32)"
```

Read the value back only to put it in the connector config, not into git:

```bash
aws secretsmanager get-secret-value \
  --region us-east-1 \
  --secret-id contextweave/prod/mcp-bearer-token \
  --query SecretString --output text
```

The function caches a successful read for 60 seconds. After you rotate the
secret, wait for that window or update the function so a new environment
starts.

### Deploy

Merging to `main` runs `.github/workflows/deploy.yaml`, which deploys
`contextweave-rag-prod`. That deploy passes `McpAuthTokenSecretName`. The
value is the repository variable `MCP_AUTH_TOKEN_SECRET_NAME` when it is set,
and `contextweave/prod/mcp-bearer-token` otherwise. Pull requests only
validate and build; they do not deploy.

**Create the secret before merging.** The workflow does not create it and
does not contain the token. If `contextweave/prod/mcp-bearer-token` (or the
name in `MCP_AUTH_TOKEN_SECRET_NAME`) is missing when prod deploys, the
parameter is still set and `/mcp` returns 401 for every caller. An empty
parameter does the same thing, which is how the first deploy of this endpoint
behaved.

1. Create the secret or parameter above, in the prod account, before the merge.
2. Build, when deploying by hand:

```bash
sam build --template template.yaml --parallel --cached
```

3. A hand deploy has to pass the same overrides as job `deploy-prod`, including
   `McpAuthTokenSecretName`. Copy the current stack parameters so the other
   values stay as they are:

```bash
aws cloudformation describe-stacks \
  --stack-name contextweave-rag-prod \
  --region us-east-1 \
  --query "Stacks[0].Parameters[].[ParameterKey,ParameterValue]" \
  --output text
```

```bash
sam deploy \
  --stack-name contextweave-rag-prod \
  --region us-east-1 \
  --config-env prod \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --resolve-s3 \
  --parameter-overrides \
    Environment=prod \
    LogRetentionDays=365 \
    GenerationModelId=us.amazon.nova-pro-v1:0 \
    SharedVpcId=<current SharedVpcId> \
    SharedLambdaSubnetAz1Id=<current SharedLambdaSubnetAz1Id> \
    SharedLambdaSubnetAz2Id=<current SharedLambdaSubnetAz2Id> \
    SharedPublicSubnetAz1Id=<current SharedPublicSubnetAz1Id> \
    SharedObservatoryMetricsTableName=<current SharedObservatoryMetricsTableName> \
    SharedObservatoryMetricsTableArn=<current SharedObservatoryMetricsTableArn> \
    SiweAuthorizerFunctionArn=<current SiweAuthorizerFunctionArn> \
    McpAuthTokenSecretName=contextweave/prod/mcp-bearer-token
```

For the SSM parameter, set `McpAuthTokenSecretName=/contextweave/prod/mcp-bearer-token`
instead. Do not drop the existing overrides: a partial list can reset other
parameters. An empty `SiweAuthorizerFunctionArn` leaves the health-record
routes and the health-record MCP tools off, which is the existing gate.

4. Confirm `McpEndpoint` with the `describe-stacks` command above.

### Add the connector

In Cursor, add a remote MCP server (Streamable HTTP) whose URL is `McpEndpoint`
and whose header is the bearer token. The same config works in an MCP client
that speaks Streamable HTTP:

```json
{
  "mcpServers": {
    "contextweave": {
      "url": "https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/mcp",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

The client sends `initialize`, then `tools/list` and `tools/call`. Supported
methods also include `ping` and the `notifications/initialized` notification.

| Tool | Wraps | Notes |
|---|---|---|
| `query_expertise` | `POST /query-expertise` | `question` required |
| `submit_feedback` | `POST /feedback` | `queryId` and `rating` |
| `list_routing_decisions` | `GET /routing-decisions` | `mode` `list` or `summary` |
| `health` | `GET /health` | Liveness and routing-graph health |
| `get_agent_card` | `GET /.well-known/agent-card.json` | |
| `health_query` | `POST /health/query` | Listed only when the health store is enabled |
| `list_health_documents` | `GET /health/documents` | Listed only when the health store is enabled |
| `delete_health_document` | `DELETE /health/documents/{docId}` | `destructiveHint: true` |

Direct HTTP calls to `/health/query`, `/health/documents`, and
`/health/documents/{docId}` still use the API's custom authorizer. Those routes
are unchanged. `/mcp` does not use that authorizer; it checks the bearer token
above. The health-record tools are registered only when that authorizer is
configured, which is the same condition that creates the health routes.

---

## Observing Routing Behaviour

### Check current strategy weights

```bash
# GET /health includes document type distribution from the routing graph
curl -s $API_URL/../health | jq .routingGraph
```

### Read EFFECTIVE_FOR weights directly from Memgraph

```cypher
MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
RETURN r.label AS strategy, d.question_type AS question_type,
       e.weight AS weight, e.feedback_count AS queries
ORDER BY d.question_type, e.weight DESC
```

### Inspect the CAG cache

```sql
SELECT question_type, hit_count, created_at, expires_at
FROM query_cache
WHERE expires_at > NOW()
ORDER BY hit_count DESC
LIMIT 20;
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
    "GENERATION_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "ARTIFACTS_BUCKET": "expertise-rag-artifacts-123456789012-dev",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_DB": "expertiserag",
    "MEMGRAPH_HOST": "localhost"
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
│   │   ├── graph_builder.py      # Graph entity/edge + routing metadata construction
│   │   ├── routing_analyzer.py   # Document type classifier + chunking recommender
│   │   └── requirements.txt
│   ├── query_api/
│   │   ├── handler.py            # Lambda entry point (POST /query-expertise)
│   │   ├── retriever.py          # Multi-strategy retrieval (pgvector/Memgraph/keyword)
│   │   ├── rag_router.py         # Routing agent: strategy selection + feedback
│   │   ├── cache.py              # CAG: semantic response cache (pgvector)  ★ NEW
│   │   ├── graph_expander.py     # Memgraph graph expansion + routing queries
│   │   ├── synthesizer.py        # Bedrock Converse answer synthesis
│   │   └── requirements.txt
│   ├── mcp_server/
│   │   ├── handler.py            # POST /mcp and GET /mcp (Streamable HTTP MCP)
│   │   ├── tools.py              # tools/list and tools/call wrappers
│   │   └── auth.py               # Bearer token check (Secrets Manager or SSM)
│   ├── ingestion_trigger/
│   │   ├── handler.py            # CloudFormation custom resource + routing graph seed
│   │   └── requirements.txt
│   └── shared/                                                              ★ NEW
│       ├── db_clients.py         # Memgraph + PostgreSQL singleton factories ★ NEW
│       ├── embedder.py           # Titan Text Embeddings V2 wrapper          ★ NEW
│       ├── chunker.py            # Text chunking utilities                   ★ NEW
│       └── models.py             # Domain models, routing enums, constants
├── events/                        # Sample Lambda event payloads
├── examples/                      # Sample graph/API inputs and outputs
├── scripts/
│   ├── upload_repo.py            # Upload local repo to S3
│   ├── start_ingestion.py        # Start + monitor repo ingestion runs
│   └── query_expertise.py        # Query via HTTP API
└── docs/
    ├── architecture.md           # System architecture and design decisions
    ├── graph_schema.md           # Graph node/edge schema including routing graph
    ├── api_reference.md          # Full API reference including routingDecision field
    ├── ingestion-and-retrieval-walkthrough.md  # End-to-end trace with routing steps
    ├── c4.puml                   # C4 system context + container diagram
    └── aws-infrastructure.puml   # AWS service topology diagram
```

---

## CI/CD

Push to `main`/`master` triggers an automatic deploy to the `dev` environment via GitHub Actions. The workflow uses OIDC — no long-lived AWS credentials are stored in GitHub secrets.

```
Push → sam validate → sam build → sam deploy → S3 upload (signal files)
                                              → routing graph seeded on CloudFormation Create/Update
                                              → pgvector schema initialised on CloudFormation Create/Update
```

Manual deploys to `staging` or `prod` are triggered via `workflow_dispatch`.

---

## Cost Considerations

| Service | Cost driver | Notes |
|---------|-------------|-------|
| Memgraph (EC2) | EC2 instance type + EBS | Expertise + routing graph; Community Edition is free |
| PostgreSQL | Instance type + storage | Chunk vectors + CAG cache; RDS or self-managed |
| Amazon Bedrock | Per input/output token (Converse) + per 1K tokens (Titan V2) | ~$0.0002/1K embed tokens |
| AWS Lambda | Per invocation + duration | Typically cents |
| Amazon S3 | Storage + requests | Minimal |

**Estimated minimum monthly cost** (dev, light usage): ~$100–200/month
(Memgraph EC2 + PostgreSQL instance)

To reduce costs in development:
- Use a smaller Memgraph EC2 type in dev (for example `t4g.nano`)
- Set `EnableStepFunctions` to `false`
- Use smaller generation models (Claude Haiku)
- Use a single PostgreSQL instance for both pgvector and the CAG cache

---

## Security

- All S3 data encrypted with a customer-managed KMS key (auto-rotates annually)
- S3 bucket policy enforces SSL-only access; no public S3 access
- All IAM roles are least-privilege
- Access to Memgraph and PostgreSQL is mediated by VPC security groups and Secrets Manager
- Memgraph and PostgreSQL credentials stored in AWS Secrets Manager (never in env vars or code)
- GitHub Actions deploys via OIDC — no long-lived access keys
- `POST /mcp` and `GET /mcp` require a bearer token stored in Secrets Manager or SSM SecureString. The template holds the secret name (`McpAuthTokenSecretName`), not the token. A failed check returns 401
- CloudWatch logs retained per `LogRetentionDays` parameter

---

## License

See [LICENSE](LICENSE).
