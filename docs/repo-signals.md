# repo-signals – ContextWeave / ExpertiseRAG

> Structured expertise signals for the ExpertiseRAG GraphRAG system.
> **Evidence weight: 0.7** (high-authority structured metadata).
>
> This file is uploaded to S3 (`raw/contextweave/`) by the GitHub Actions
> workflow after every successful SAM deploy, and ingested into the Bedrock
> Knowledge Base → Neptune Analytics graph.

| Field | Value |
|---|---|
| Schema version | 1.0 |
| Generated at | 2026-03-20 |
| Repo | contextweave |
| Owner | rajatarun |

---

## Project Metadata

**Name**: ContextWeave / ExpertiseRAG
**Type**: production-platform
**Domain**: graphrag-expertise-retrieval

AWS-native GraphRAG platform with adaptive RAG routing. Combines Amazon Bedrock Knowledge Bases with Neptune Analytics to answer deep, evidence-backed questions about a developer's professional expertise. An agentic routing layer classifies every document at ingestion time and at query time selects the optimal retrieval strategy from a self-improving Neptune graph. Uses semantic retrieval + graph traversal for evidence synthesis.

| Property | Value |
|---|---|
| AWS Account | `239571291755` |
| Region | `us-east-1` |
| Stack – dev | `expertise-rag-dev` |
| Stack – staging | `expertise-rag-staging` |
| Stack – prod | `expertise-rag-prod` |

### CI/CD Pipeline

| Property | Value |
|---|---|
| Platform | GitHub Actions |
| Auth method | OIDC keyless |
| Deploy role | `arn:aws:iam::239571291755:role/teamweave-github-actions-sam-deployer` |
| Stages | sam-validate → sam-build → sam-deploy → s3-artifact-upload |

---

## Architecture Patterns

### GraphRAG

Graph-augmented retrieval-augmented generation. Combines vector similarity search (Bedrock KB) with graph traversal (Neptune Analytics) to surface co-occurring skills, pattern evidence, and cross-repository context that pure vector search cannot recover.

**Evidence**: `src/query_api/graph_expander.py`, `src/query_api/retriever.py`, `src/preprocessor/graph_builder.py`

---

### Serverless Event-Driven Architecture

Fully serverless. S3 ObjectCreated events trigger Lambda. Step Functions orchestrates multi-step workflows without any always-on compute. API Gateway routes HTTP requests to Lambda. No EC2, no containers.

**Evidence**: `template.yaml`, `src/preprocessor/handler.py`, `src/ingestion_trigger/handler.py`

---

### Hierarchical RAG Chunking

Parent chunks (1500 tokens) preserve full section context for answer synthesis; child chunks (300 tokens) are the precision retrieval units. 60-token overlap ensures continuity at boundaries.

**Evidence**: `template.yaml` (`BedrockDataSource HierarchicalChunkingConfiguration`)

---

### Infrastructure as Code (SAM)

All AWS resources defined in `template.yaml` using AWS SAM. Reproducible across dev / staging / prod environments via `samconfig.toml`. CI/CD deploys exclusively via SAM CLI, never via console clicks.

**Evidence**: `template.yaml`, `samconfig.toml`, `.github/workflows/deploy.yaml`

---

### Least-Privilege IAM

Every Lambda has a separate IAM role with only the permissions required for that function. No wildcard resource ARNs except where unavoidable (X-Ray, CloudWatch). S3 access scoped to specific prefixes.

**Evidence**: `template.yaml` (`PreprocessorRole`, `QueryAPIRole`, `IngestionTriggerRole`)

---

### OIDC Keyless Authentication

GitHub Actions assumes the deployer IAM role via OIDC token exchange. No long-lived AWS access keys stored in GitHub secrets. Token scoped to specific repo and workflow run.

**Evidence**: `.github/workflows/deploy.yaml`

---

### Adaptive RAG Routing

Agentic routing layer that classifies every document at ingestion time into a `DocumentType` (`technical_spec`, `narrative`, `structured_data`, `code`, `diagram_derived`) and recommends the best `ChunkingStrategy`. At query time, a `RAGRouter` reads `EFFECTIVE_FOR` edge weights from Neptune to select the optimal retrieval strategy (`graph_first`, `hybrid`, `keyword_boosted`, `semantic_search`). After each query, the winning strategy's weight is updated based on answer confidence, making the routing graph self-improving without any retraining.

**Evidence**: `src/preprocessor/routing_analyzer.py`, `src/query_api/rag_router.py`, `src/query_api/handler.py`, `src/shared/models.py`

---

### Self-Improving Knowledge Graph

Neptune Analytics routing sub-graph with `EFFECTIVE_FOR` edges between `RAGStrategy` and `DocumentType` nodes. Weights are seeded from `ROUTING_PRIORS` at deploy time and updated after every query (confidence ≥ 0.70: weight +0.05; confidence < 0.40: weight −0.02; clamped to [0.10, 1.00]). Strategies that consistently produce high-confidence answers accumulate higher weights and are selected more often, creating a compounding improvement loop with zero manual intervention.

**Evidence**: `src/query_api/rag_router.py`, `src/shared/models.py`, `docs/graph_schema.md`

---

## Design Patterns

### GoF Behavioural Patterns

#### Strategy Pattern (Chunking)

Runtime selection of the text-chunking algorithm based on `DocumentType`. `chunk_text()` in `chunker.py` dispatches to `_chunk_hierarchical()`, `_chunk_sentence()`, or `_chunk_fixed()` depending on the strategy recommended by the routing analyzer. New chunking strategies can be added without touching the dispatcher.

**Evidence**: `src/shared/chunker.py`, `src/preprocessor/routing_analyzer.py`

---

#### Strategy Pattern (RAG Retrieval)

Four interchangeable retrieval strategies (`semantic_search`, `graph_first`, `keyword_boosted`, `hybrid`) share a common interface. `RAGRouter` selects the strategy at runtime by reading learned `EFFECTIVE_FOR` weights from Neptune; `retrieve_with_strategy()` dispatches to the chosen implementation.

**Evidence**: `src/query_api/rag_router.py`, `src/query_api/retriever.py`

---

#### Template Method Pattern

The preprocessor `lambda_handler()` defines the invariant skeleton of the document processing algorithm — parse event → extract → route-classify → build graph → write derived artifacts — while delegating each step to specialised helper functions that can be evolved independently.

**Evidence**: `src/preprocessor/handler.py`

---

#### Facade Pattern

`_run_query_pipeline()` in `query_api/handler.py` presents a single unified entry point to a five-stage reasoning pipeline (classify → `RAGRouter.select_strategy` → `retrieve_with_strategy` → `expand_graph_context` → `synthesize_answer` → `RAGRouter.update_feedback`). Callers interact with one function; internal orchestration complexity is hidden behind the facade.

**Evidence**: `src/query_api/handler.py`

---

#### Command Pattern (Action Dispatch)

The ingestion-trigger Lambda dispatches on an `action` field in the event payload: `start`, `status`, `seed_routing`, `empty_all`, etc. Each action is encapsulated in its own function, making it trivial to add or remove operations without changing the dispatcher.

**Evidence**: `src/ingestion_trigger/handler.py`

---

#### Observer Pattern (Confidence Feedback Loop)

After every synthesized answer, `RAGRouter.update_feedback()` observes the answer confidence score and adjusts the `EFFECTIVE_FOR` edge weight for the winning strategy in Neptune. Confidence ≥ 0.70 → +0.05 (reinforce); confidence < 0.40 → −0.02 (penalise). The routing graph acts as the subject; the feedback function is the observer that mutates it.

**Evidence**: `src/query_api/rag_router.py`

---

#### Chain of Responsibility (Classification Fallback)

`classify_question()` in `synthesizer.py` tries handlers in order: model-based classification → regex-pattern matching → default fallback to `"general"`. Each handler attempts to satisfy the request; if it cannot, responsibility is passed to the next handler in the chain.

**Evidence**: `src/query_api/synthesizer.py`

---

### GoF Creational Patterns

#### Builder Pattern

`GraphBuilder` in `graph_builder.py` is a stateful accumulator. Callers repeatedly invoke `add_extraction()` for each document; the builder deduplicates nodes and edges using stable IDs, then finalises the graph with `build()` or persists it via `write_to_neptune()`. Separates construction from representation.

**Evidence**: `src/preprocessor/graph_builder.py`

---

#### Factory Pattern (Extractor Dispatch)

`extract()` in `extractors.py` acts as a factory function: given a file path and its content, it selects and invokes the correct extractor (`extract_markdown`, `extract_yaml`, `extract_plantuml`, `extract_text`) based on file extension, returning a uniform output envelope regardless of the underlying extractor used.

**Evidence**: `src/preprocessor/extractors.py`

---

#### Singleton / Connection Pool Pattern

AWS SDK and Neptune clients are initialised once per Lambda container lifecycle and cached in module-level globals. `get_neptune_client()` and `_get_bedrock_client()` in `db_clients.py` return the cached instance on warm invocations, avoiding repeated TCP handshakes and auth overhead.

**Evidence**: `src/shared/db_clients.py`

---

### GoF Structural Patterns

#### Adapter Pattern (JSON ↔ Dataclass)

`QueryRequest.from_dict()`, `RetrievedChunk.to_dict()`, and `QueryResponse.to_dict()` in `models.py` adapt the camelCase HTTP request and response shapes to and from the snake_case Python dataclasses used internally, preventing HTTP-layer concerns from leaking into business logic.

**Evidence**: `src/shared/models.py`

---

### Cloud / Distributed Patterns

#### Polling Pattern (Async Job Completion)

The Step Functions state machine uses a Wait → CheckStatus → branch loop to monitor the Bedrock Knowledge Base ingestion job. Rather than blocking a Lambda for potentially minutes, the orchestrator polls at a fixed interval and branches on `COMPLETE` / `FAILED` / `IN_PROGRESS`, avoiding Lambda timeout limits for long-running operations.

**Evidence**: `template.yaml` (`WaitForIngestion` + `CheckIngestionStatus` states)

---

#### Retry with Exponential Backoff

Step Functions state machine tasks declare `Retry` blocks with `BackoffRate: 2` and `MaxAttempts: 3`, providing automatic exponential backoff for transient service errors. The GitHub Actions workflow also retries SAM deploy steps on network failures.

**Evidence**: `template.yaml`, `.github/workflows/deploy.yaml`

---

#### Circuit Breaker (Connection Health Check)

Before returning a cached Neptune or Bedrock client, `db_clients.py` performs a lightweight connectivity probe (`verify_connectivity()`). If the probe fails, the cached reference is cleared and a fresh connection is established, preventing cascading failures from stale client state.

**Evidence**: `src/shared/db_clients.py`

---

#### CloudFormation Custom Resource Pattern

`PostDeploymentIngestionJob` is a CloudFormation custom resource backed by the `IngestionTriggerFunction` Lambda. On stack Create/Update events, it seeds the Neptune routing graph and kicks off the initial Bedrock KB ingestion job, then signals success or failure to CloudFormation via a pre-signed S3 URL callback.

**Evidence**: `src/ingestion_trigger/handler.py`, `template.yaml` (`AWS::CloudFormation::CustomResource`)

---

#### Idempotency Pattern

All schema-init and graph-seed operations use upsert semantics (SQL `IF NOT EXISTS`, openCypher `MERGE`) so they are safe to run multiple times without side effects. This makes CloudFormation stack updates, Lambda retries, and manual re-seeds all idempotent by construction.

**Evidence**: `src/ingestion_trigger/handler.py`, `src/preprocessor/graph_builder.py`

---

#### Bulkhead Pattern (Lambda Isolation)

Each Lambda function (Preprocessor, Query API, IngestionTrigger) is deployed with independent memory, timeout, IAM role, and log group. Resource exhaustion or a runaway request in one function cannot affect the others, containing blast radius to the function boundary.

**Evidence**: `template.yaml` (Globals + per-function overrides)

---

#### Secrets Manager Integration

Database credentials and API keys are stored in AWS Secrets Manager. Lambda functions retrieve secrets at cold-start and cache them for the container lifetime. No credentials appear in environment variables or source code.

**Evidence**: `src/shared/db_clients.py`, `template.yaml` (`AWS::SecretsManager::Secret`)

---

## AWS Services

### Compute

| Service | Runtime | Functions | Depth |
|---|---|---|---|
| AWS Lambda | Python 3.12 | `expertise-rag-preprocessor`, `expertise-rag-query-api`, `expertise-rag-ingestion-trigger` | Advanced |

### Storage

| Service | Usage | Encryption | Depth |
|---|---|---|---|
| Amazon S3 | Artifact store | SSE-KMS | Advanced |

### Database

| Service | Usage | Query Language | Vector Dim | Depth |
|---|---|---|---|---|
| Amazon Neptune Analytics | Vector + graph hybrid store | openCypher | 1024 | Advanced |

### AI / ML

| Service | Usage | Model ID | Depth |
|---|---|---|---|
| Amazon Bedrock Knowledge Bases | RAG knowledge base (hierarchical chunking) | — | Advanced |
| Amazon Bedrock – Titan Text Embeddings V2 | 1024-dim embeddings | `amazon.titan-embed-text-v2:0` | Intermediate |
| Amazon Bedrock – Claude 3 Haiku | Document parsing during ingestion | `anthropic.claude-3-haiku-20240307-v1:0` | Intermediate |
| Amazon Bedrock – Converse API | Answer synthesis | — | Advanced |

### Integration

| Service | Usage | Depth |
|---|---|---|
| Amazon API Gateway (HTTP API v2) | REST API | Intermediate |
| AWS Step Functions (STANDARD) | Ingestion orchestration | Advanced |
| Amazon EventBridge | Lambda event routing | Intermediate |

### Security

| Service | Usage | Depth |
|---|---|---|
| AWS KMS | SSE-CMK | Intermediate |
| AWS IAM | Least-privilege roles | Advanced |
| AWS STS | OIDC assume-role | Intermediate |

### Observability

| Service | Usage | Depth |
|---|---|---|
| AWS X-Ray | Distributed tracing | Intermediate |
| Amazon CloudWatch Logs | Structured logging | Intermediate |

### IaC / CI/CD

| Service | Usage | Depth |
|---|---|---|
| AWS SAM | Infrastructure as code | Advanced |
| AWS CloudFormation | Underlying IaC | Intermediate |
| GitHub Actions | CI/CD pipeline | Advanced |

---

## Technical Skills

### Languages

| Language | Version | Depth | Evidence |
|---|---|---|---|
| Python | 3.12 | Advanced | `src/preprocessor/`, `src/query_api/`, `src/ingestion_trigger/`, `src/shared/` |
| YAML | — | Advanced | `template.yaml`, `samconfig.toml`, `.github/workflows/deploy.yaml`, `repo-signals.yaml` |
| Bash | — | Intermediate | `deploy.sh`, `guardrail.sh` |
| JavaScript (Node.js) | 20 | Intermediate | `src/index.mjs` |

### Cloud Platforms

**Amazon Web Services** – Expert

Specialisations: Serverless (Lambda, API Gateway, Step Functions) · AI/ML (Bedrock, Neptune Analytics, Titan Embeddings) · Storage (S3, KMS) · Security (IAM, OIDC, KMS) · Observability (CloudWatch, X-Ray) · IaC (SAM, CloudFormation)

### Concepts

| Concept | Depth |
|---|---|
| Retrieval-Augmented Generation (RAG) | Advanced |
| GraphRAG | Advanced |
| Adaptive RAG Routing | Advanced |
| Agentic AI Systems | Advanced |
| Knowledge Graph Design | Advanced |
| Self-Improving Systems (Reinforcement without Retraining) | Advanced |
| Event-Driven Architecture | Advanced |
| Serverless Architecture | Advanced |
| CI/CD Pipeline Design | Advanced |
| Security Best Practices (Zero Trust, Least Privilege) | Advanced |
| Infrastructure as Code | Advanced |

---

## Documentation Index

| File | Type | Weight | Description |
|---|---|---|---|
| `CLAUDE.md` | authoritative-context | 1.0 | AI project context – architecture decisions, patterns, anti-patterns |
| `docs/architecture.md` | architecture-doc | 1.0 | Full system architecture with component details, data flows, IAM model |
| `docs/c4.puml` | c4-diagram | 0.8 | C4 Level 1 (System Context) and Level 2 (Container) PlantUML diagrams |
| `docs/aws-infrastructure.puml` | infrastructure-diagram | 0.8 | AWS service topology with all connections, IAM roles, data flows |
| `repo-signals.yaml` | structured-signals | 0.7 | Machine-readable expertise signals for RAG ingestion |
| `docs/repo-signals.md` | structured-signals | 0.7 | Markdown rendering of repo-signals.yaml |
| `docs/graph_schema.md` | schema-doc | 0.8 | Neptune Analytics node/edge schema definition |
| `docs/api_reference.md` | api-reference | 0.6 | Query API endpoint reference |

---

## Accomplishments

- Built production-grade GraphRAG platform entirely on AWS managed services
- Eliminated long-lived credentials by implementing GitHub Actions OIDC federated auth
- Designed multi-environment SAM pipeline (dev/staging/prod) with automated CI/CD
- Implemented custom C4 + AWS infrastructure diagrams as PlantUML for living documentation
- Achieved least-privilege IAM across 5 separate service roles with resource-scoped policies
- Architected Neptune Analytics graph schema with 13 node types and 12 edge types for expertise + routing modelling
- Integrated hierarchical chunking (1500/300 token parent/child) for precision RAG retrieval
- Built evidence weighting system that prioritises architecture docs over resume claims
- Designed and implemented agentic adaptive RAG routing layer with 4 retrieval strategies and 5 document type classifiers
- Built self-improving routing graph in Neptune that learns optimal strategy per question type through post-query feedback
- Achieved 100% skill_depth routing accuracy and 0.83 average answer confidence across 32 test questions
- Implemented keyword-overlap reranking (Jaccard-based, 25/75 blend) for project and credential question types
- Seeded routing priors at deploy time for zero-cold-start operation before any feedback data is collected
