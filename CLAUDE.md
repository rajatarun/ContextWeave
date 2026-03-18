# CLAUDE.md – ContextWeave / ExpertiseRAG

> Authoritative project context for Claude AI and other LLM-based tools.
> Evidence weight: **1.0** (highest authority signal in the RAG system).
> Last updated: 2026-03-18

---

## Project Identity

**Name**: ContextWeave / ExpertiseRAG
**Type**: AWS-native GraphRAG platform
**Purpose**: Answers deep, evidence-backed questions about a developer's professional expertise using retrieval-augmented generation with a knowledge graph.
**Owner**: Rajat Arun (rajatarun)
**Repository**: https://github.com/rajatarun/ContextWeave
**Account**: AWS account `239571291755` (teamweave)

---

## Architecture Summary

ExpertiseRAG is a **serverless, event-driven GraphRAG system** combining AWS managed services with self-hosted graph and vector databases inside a private VPC:

| Layer | Service | Role |
|-------|---------|------|
| Storage | Amazon S3 | Raw uploads (`raw/`) and derived artifacts (`derived/`) |
| Networking | VPC + NAT Gateway + S3 Endpoint | Private subnets for Lambda, Memgraph, RDS; NAT for outbound; S3 endpoint bypasses NAT |
| Preprocessing | AWS Lambda (Python 3.12) | Extracts text, classifies document type, builds knowledge graph entities, generates embeddings |
| Routing Analyzer | `src/preprocessor/routing_analyzer.py` | Classifies DocumentType + recommends ChunkingStrategy per document |
| Knowledge Graph | Memgraph (EC2/EBS, `t3.medium`) | Graph store for expertise nodes/edges + routing intelligence graph |
| Vector Store | PostgreSQL pgvector (RDS) | Stores text chunks with 1024-dimensional Titan V2 embeddings |
| Embeddings | Amazon Titan Text Embeddings V2 | 1024-dimensional semantic embeddings via Bedrock |
| Query API | AWS Lambda + API Gateway HTTP v2 | POST /query-expertise – classify → route → retrieve → graph-expand → synthesize → feedback |
| RAG Router | `src/query_api/rag_router.py` | Selects optimal retrieval strategy from Memgraph EFFECTIVE_FOR weights; updates weights after each query |
| Answer Synthesis | Amazon Bedrock Converse API | Amazon Nova Pro v1:0 (`us.amazon.nova-pro-v1:0`) for answer generation |
| DB Initializer | AWS Lambda (CloudFormation custom resource) | Initializes Memgraph schema + seeds routing graph; runs once on deploy |
| Orchestration | AWS Step Functions | Preprocess → DB init → Poll loop |
| Encryption | AWS KMS | SSE-KMS on S3; key rotates annually |
| Observability | AWS X-Ray + CloudWatch | All Lambdas and Step Functions traced |
| IaC | AWS SAM (`template.yaml`) | Full infrastructure as code |
| CI/CD | GitHub Actions | OIDC → SAM validate → build → deploy → S3 upload |

### Adaptive RAG Routing

ContextWeave includes an **agentic routing layer** that automatically selects the best retrieval strategy for each question type and improves continuously through a feedback loop.

**At ingestion time**, every document is classified into a `DocumentType` and assigned a `ChunkingStrategy`:

| DocumentType | Detection heuristics | ChunkingStrategy |
|---|---|---|
| `technical_spec` | ≥3 headings + ≥2 AWS service signals | `hierarchical` (1500/300 tokens) |
| `narrative` | Long sentences, few headings, low code ratio | `sentence` |
| `structured_data` | `.yaml`/`.json` / key-value density | `fixed_256` |
| `code` | Source file extension / code ratio > 35% | `fixed_512` |
| `diagram_derived` | `.puml` extension / `@startuml` marker | `fixed_256` |

DocumentType and ChunkingStrategy nodes — and their relationships to each Document node — are written to Memgraph alongside expertise nodes.

**At query time**, the `RAGRouter` reads `EFFECTIVE_FOR` edge weights from Memgraph to select the optimal strategy:

| Strategy | When chosen | Behaviour |
|---|---|---|
| `graph_first` | skill_depth, architecture | Graph traversal in Memgraph + pgvector retrieval |
| `hybrid` | comparison | pgvector + Memgraph vector search + keyword boost |
| `keyword_boosted` | project, credential | pgvector + keyword-overlap reranking (25/75 blend) |
| `semantic_search` | general | pgvector semantic search only |

**After every query**, the winning strategy's `EFFECTIVE_FOR` edge weight is updated:
- `confidence ≥ 0.70` → `weight += 0.05` (cap 1.00)
- `confidence < 0.40` → `weight -= 0.02` (floor 0.10)
- `0.40 ≤ confidence < 0.70` → no change

The graph learns from every answered question. No retraining. No manual tuning.

---

## Repository Structure

```
/
├── template.yaml                           # SAM CloudFormation template (1200+ lines)
├── samconfig.toml                          # SAM deployment config (dev/staging/prod)
├── CLAUDE.md                               # This file — authoritative AI context
├── repo-signals.yaml                       # Structured expertise signals (weight 0.7)
├── README.md                               # User guide and quick start
├── .github/
│   └── workflows/
│       └── deploy.yaml                     # GitHub Actions CI/CD pipeline
├── src/
│   ├── shared/                             # Shared library across all Lambdas
│   │   ├── models.py                       # Canonical domain models (nodes, edges, enums)
│   │   ├── chunker.py                      # Text chunking strategies
│   │   ├── embedder.py                     # Titan V2 embedding generation
│   │   └── db_clients.py                   # Memgraph + PostgreSQL connection clients
│   ├── preprocessor/
│   │   ├── handler.py                      # S3-triggered Lambda entrypoint
│   │   ├── extractors.py                   # Text extraction (markdown, yaml, puml)
│   │   ├── graph_builder.py                # Graph entity + edge construction
│   │   ├── routing_analyzer.py             # Document type classification
│   │   ├── models.py                       # Preprocessor-local models
│   │   └── requirements.txt
│   ├── query_api/
│   │   ├── handler.py                      # API Gateway Lambda entrypoint
│   │   ├── rag_router.py                   # Adaptive retrieval strategy selection
│   │   ├── retriever.py                    # Multi-strategy chunk retrieval
│   │   ├── graph_expander.py               # Memgraph context traversal
│   │   ├── synthesizer.py                  # Question classification + answer generation
│   │   ├── models.py                       # Query API-local models
│   │   └── requirements.txt
│   └── ingestion_trigger/
│       ├── handler.py                      # CloudFormation custom resource + DB seed
│       └── requirements.txt
├── docs/
│   ├── architecture.md                     # System architecture deep-dive (weight 1.0)
│   ├── graph_schema.md                     # Memgraph node/edge schema reference
│   ├── api_reference.md                    # Full API reference
│   ├── ingestion-and-retrieval-walkthrough.md
│   ├── c4.puml                             # C4 Level 1 & 2 diagrams (weight 0.8)
│   └── aws-infrastructure.puml            # AWS service topology diagram (weight 0.8)
├── scripts/
│   ├── upload_repo.py                      # Upload files to S3 raw/
│   ├── start_ingestion.py                  # Trigger Step Functions ingestion
│   └── query_expertise.py                  # Test query via API Gateway
└── events/                                 # Sample Lambda test events
```

---

## Core Design Decisions

### 1. Memgraph (EC2) + PostgreSQL pgvector (not Neptune Analytics)
Memgraph provides **full Cypher query support** for complex graph traversal at lower cost than Neptune Analytics. PostgreSQL pgvector handles chunk vector storage with native `<=>` cosine similarity operators. Both run inside private VPC subnets accessible only to Lambda.

### 2. Hierarchical Chunking (1500 / 300 tokens)
Parent chunks (1500 tokens) preserve full section context for synthesis; child chunks (300 tokens) are the retrieval units for precision. Overlap (60 tokens) ensures continuity across chunk boundaries.

### 3. Amazon Nova Pro for Answer Synthesis
The query API uses Amazon Nova Pro v1:0 (`us.amazon.nova-pro-v1:0`) via Bedrock Converse API for answer generation. Cross-region inference profile (`us.` prefix) ensures on-demand throughput without provisioning.

### 4. Evidence Weighting at Query Time
Source credibility is encoded in the retrieval pipeline via `SOURCE_WEIGHTS` in `src/shared/models.py`:
- `architecture.md`, `CLAUDE.md` → weight **1.0** (self-authored authoritative docs)
- PlantUML-derived summaries → weight **0.8** (diagram relationships)
- `repo-signals.yaml` → weight **0.7** (structured signals)
- Code / README → weight **0.6** (implementation evidence)
- Articles / blog posts → weight **0.5**
- Resume → weight **0.3** (self-reported, not verified)

`RetrievedChunk.effective_score = score × source_weight` governs final ranking.

### 5. OIDC-based GitHub Actions Deployment
No long-lived AWS credentials are stored in GitHub secrets. The workflow assumes IAM role `arn:aws:iam::239571291755:role/teamweave-github-actions-sam-deployer` via OIDC token exchange.

### 6. VPC-Private Databases
Memgraph EC2 and RDS PostgreSQL are in private subnets with no public IP. Lambdas connect via VPC. A NAT Gateway enables outbound internet (Bedrock API calls). An S3 VPC Gateway Endpoint routes S3 traffic without hitting NAT.

---

## Lambda Functions

### `expertise-rag-preprocessor-{env}`
- **Trigger**: S3 `ObjectCreated` on `raw/` prefix
- **Input**: Raw files (`.md`, `.yaml`, `.puml`, `.txt`)
- **Pipeline**:
  1. Parse S3 event → identify repo prefix and file keys
  2. Download files from S3
  3. Extract content via `extractors.extract()`
  4. Classify document type via `routing_analyzer.analyze_document()`
  5. Build graph entities + edges via `GraphBuilder`
  6. Chunk text + generate Titan V2 embeddings → write to PostgreSQL pgvector
  7. Write expertise graph to Memgraph
  8. Write derived artifacts to `derived/<repo>/` in S3
- **Output**: `*.derived.json`, `*.extracted.txt`, `graph_entities.json`, `graph_edges.json`, `expertise_signals.json`, `processing_manifest.json`
- **Code**: `src/preprocessor/handler.py`

### `expertise-rag-query-api-{env}`
- **Trigger**: API Gateway HTTP v2 — POST `/query-expertise`, GET `/health`
- **Pipeline**:
  1. Parse and validate `QueryRequest` (question, topK, includeGraphExpansion, minConfidence)
  2. Classify question type via `synthesizer.classify_question()`
  3. Select strategy via `RAGRouter.select_strategy()` (reads Memgraph EFFECTIVE_FOR weights)
  4. Retrieve chunks via `retriever.retrieve_with_strategy()` (pgvector ± Memgraph ± keyword)
  5. Deduplicate chunks by content hash
  6. Expand graph context via Memgraph traversal (when `include_graph=True`)
  7. Synthesize answer via Bedrock Converse (Nova Pro)
  8. Update routing feedback in Memgraph via `RAGRouter.update_feedback()`
- **Response**: `{ answer, sources, inferredSkills, repeatedPatterns, confidence, questionType, graphEntitiesUsed, retrievalCount, modelId, routingDecision }`
- **Code**: `src/query_api/handler.py`

### `expertise-rag-db-initializer-{env}` (DBInitFunction)
- **Trigger**: CloudFormation custom resource (post-deploy)
- **Actions**: Creates Memgraph schema constraints + indexes; seeds `RAGStrategy`, `DocumentType`, and `EFFECTIVE_FOR` edges with `ROUTING_PRIORS`; initializes PostgreSQL pgvector extension and `chunks` table
- **On Deploy**: Idempotent — safe to re-run
- **Code**: `src/ingestion_trigger/handler.py`

---

## Shared Library (`src/shared/`)

| Module | Purpose |
|--------|---------|
| `models.py` | All canonical dataclasses: `GraphNode`, `GraphEdge`, `ExpertiseSignal`, `DerivedArtifact`, `QueryRequest`, `RetrievedChunk`, `QueryResponse`, `RetrievalConfig`, `DocumentTypeAnalysis`; all enums (`NodeType`, `EdgeType`, `DocumentTypeLabel`, `ChunkingStrategyLabel`, `RAGStrategyLabel`); `ROUTING_PRIORS` dict; `SOURCE_WEIGHTS` dict |
| `chunker.py` | Text chunking strategies (hierarchical 1500/300, sentence, fixed-512, fixed-256) |
| `embedder.py` | Titan Text Embeddings V2 via Bedrock; returns 1024-dim float arrays |
| `db_clients.py` | Connection factories for Memgraph (Bolt protocol) and PostgreSQL (psycopg2); secrets fetched from Secrets Manager via `MEMGRAPH_SECRET_ARN` / `POSTGRES_SECRET_ARN` env vars |

---

## Key Domain Models (`src/shared/models.py`)

```python
# Node types in Memgraph
NodeType: PERSON, REPOSITORY, DOCUMENT, SKILL, PATTERN, TECHNOLOGY,
          AWS_SERVICE, ARCHITECTURE_STYLE, EVIDENCE, CLAIM,
          DOCUMENT_TYPE, CHUNKING_STRATEGY, RAG_STRATEGY

# Edge types in Memgraph
EdgeType: BUILT, CONTAINS, USES_TECH, USES_AWS_SERVICE,
          DEMONSTRATES_PATTERN, SUPPORTS_CLAIM, INDICATES_SKILL,
          DEMONSTRATES_SKILL, STRENGTHENS,
          CHUNKED_WITH,    # Document → ChunkingStrategy
          HAS_TYPE,        # Document → DocumentType
          EFFECTIVE_FOR    # RAGStrategy → DocumentType (learned weight)
```

---

## Deployment

```bash
# Validate and build
sam validate --lint
sam build --parallel --cached

# Deploy to dev (default)
sam deploy --config-env default       # → expertise-rag-dev stack

# Deploy to staging / prod
sam deploy --config-env staging       # → expertise-rag-staging stack
sam deploy --config-env prod          # → expertise-rag-prod stack
```

**Key SAM parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `Environment` | `dev` | `dev`, `staging`, `prod` |
| `GenerationModelId` | `us.amazon.nova-pro-v1:0` | Bedrock cross-region inference profile |
| `MemgraphInstanceType` | `t3.medium` | Options: `t3.medium/large/xlarge`, `m6i.large/xlarge` |
| `MemgraphEBSSizeGB` | `100` | 20–1000 GB |
| `PostgresInstanceClass` | `db.t3.medium` | Options: `db.t3.micro/medium/large`, `db.r6g.large` |
| `LogRetentionDays` | `30` (dev) / `365` (prod) | |
| `EnableStepFunctions` | `true` | Toggle Step Functions state machine |
| `VpcCidr` | `10.0.0.0/16` | VPC CIDR |
| `AdminCidr` | `10.0.0.0/8` | SSH access CIDR for Memgraph EC2 |

**IAM role used by GitHub Actions:**
```
arn:aws:iam::239571291755:role/teamweave-github-actions-sam-deployer
```

**Stack names:**
- `expertise-rag-dev`
- `expertise-rag-staging`
- `expertise-rag-prod`

**Artifacts bucket:**
```
expertise-rag-artifacts-239571291755-{env}
```

---

## CI/CD Pipeline (`.github/workflows/deploy.yaml`)

The GitHub Actions workflow runs on push to `main`/`master` or `workflow_dispatch`:

1. **validate**: `sam validate --lint`
2. **build**: `sam build --parallel --cached`
3. **deploy-prod**: `sam deploy --config-env prod` → captures stack outputs
4. **Post-deploy**: Uploads signal files to `s3://<bucket>/raw/contextweave/`
5. **Ingestion**: Triggers Step Functions ingestion + polls until complete (max 10 min)

---

## Repository Signal Files (uploaded to S3 post-deploy)

| File | S3 Key | Purpose | Weight |
|------|--------|---------|--------|
| `CLAUDE.md` | `raw/contextweave/CLAUDE.md` | Authoritative AI context (this file) | 1.0 |
| `docs/architecture.md` | `raw/contextweave/architecture.md` | System architecture deep-dive | 1.0 |
| `repo-signals.yaml` | `raw/contextweave/repo-signals.yaml` | Structured expertise signals | 0.7 |
| `docs/c4.puml` | `raw/contextweave/c4.puml` | C4 context + container diagram | 0.8 |
| `docs/aws-infrastructure.puml` | `raw/contextweave/aws-infrastructure.puml` | AWS topology diagram | 0.8 |

---

## Key Technologies Demonstrated

- **AWS SAM** – Infrastructure as code for serverless + EC2 + RDS hybrid deployments
- **Amazon Bedrock** – Embeddings (Titan V2), LLM inference (Nova Pro via Converse API)
- **Memgraph** – Self-hosted graph database (EC2/EBS) with Cypher query language; routing intelligence graph
- **PostgreSQL pgvector** – Managed RDS with vector similarity search (`<=>` cosine operator)
- **AWS Step Functions** – Long-running workflow orchestration with polling loops
- **AWS Lambda** – Python 3.12, AWS Lambda Powertools, VPC networking
- **Amazon API Gateway v2** – HTTP API with throttling and CORS
- **Amazon VPC** – Private subnets, NAT Gateway, Security Groups, S3 VPC Endpoint
- **AWS KMS** – Customer-managed key with automatic rotation
- **AWS Secrets Manager** – Memgraph + PostgreSQL credentials via `MEMGRAPH_SECRET_ARN` / `POSTGRES_SECRET_ARN`
- **GitHub Actions OIDC** – Keyless AWS authentication from CI/CD
- **GraphRAG** – Graph-augmented retrieval augmented generation pattern
- **Adaptive RAG Routing** – Self-improving Cypher-based routing graph that learns which strategy works best per question type
- **Hierarchical RAG chunking** – Parent/child chunk strategy for precision + context
- **Multi-strategy retrieval** – graph_first, hybrid, keyword_boosted, semantic_search

---

## Development Conventions

### Python
- Runtime: **Python 3.12**
- All Lambda handlers follow the `handler(event, context)` signature
- Shared code lives in `src/shared/` and is packaged into each Lambda layer or ZIP
- Use `from __future__ import annotations` for deferred type evaluation
- Domain models use `@dataclass` with `field(default_factory=...)` for mutable defaults
- Enums inherit `(str, Enum)` for JSON serialisation compatibility

### Infrastructure (SAM / CloudFormation)
- All resource names use `!Sub` with `${Environment}` suffix for multi-env isolation
- Lambda environment variables are set in the `Globals` section and per-function
- Secrets are passed via ARN env vars (`MEMGRAPH_SECRET_ARN`, `POSTGRES_SECRET_ARN`) — never plaintext
- IAM policies follow least-privilege; no `*` actions except on the KMS root key policy
- `EnableStepFunctions` condition gates the state machine to avoid cost when not needed

### Routing Graph
- `ROUTING_PRIORS` in `src/shared/models.py` are the single source of truth for initial weights
- `seed_routing_graph()` in the DB initializer is idempotent — uses `MERGE` in Cypher
- Weight clamping: `[0.10, 1.00]`; update only on `confidence ≥ 0.70` (reward) or `< 0.40` (penalty)

### Testing
- Sample Lambda events in `events/` directory
- Scripts in `scripts/` for manual upload, ingestion trigger, and query testing
- Local queries: `python scripts/query_expertise.py`

---

## Anti-patterns Explicitly Avoided

- ❌ No long-lived IAM access keys in GitHub secrets (OIDC used instead)
- ❌ No hardcoded bucket names (dynamic via CloudFormation `!Sub` + account ID)
- ❌ No public S3 access (all buckets private, SSL-only policy enforced)
- ❌ No public database endpoints (Memgraph and RDS in private subnets only)
- ❌ No monolithic Lambda (three separate functions with single-responsibility)
- ❌ No polling in Lambda (Step Functions handles the wait loop)
- ❌ No plaintext at rest (SSE-KMS on all S3 objects)
- ❌ No hardcoded credentials (Secrets Manager via ARN env vars)
- ❌ No cross-Lambda coupling (shared models only via `src/shared/`)
