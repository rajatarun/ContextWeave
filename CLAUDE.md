# CLAUDE.md – ContextWeave / ExpertiseRAG

> Authoritative project context for Claude AI and other LLM-based tools.
> Evidence weight: **1.0** (highest authority signal in the RAG system).

---

## Project Identity

**Name**: ContextWeave / ExpertiseRAG
**Type**: AWS-native GraphRAG + CAG platform
**Purpose**: Answers deep, evidence-backed questions about a developer's professional expertise using retrieval-augmented generation with a knowledge graph and a semantic response cache.
**Owner**: Rajat Arun (rajatarun)
**Repository**: https://github.com/rajatarun/ContextWeave
**Account**: AWS account `239571291755` (teamweave)

---

## Architecture Summary

ExpertiseRAG is a **serverless, event-driven GraphRAG + CAG system** built on AWS managed services and open-source databases:

| Layer | Service | Role |
|-------|---------|------|
| Storage | Amazon S3 | Raw uploads (`raw/`) and derived artifacts (`derived/`) |
| Preprocessing | AWS Lambda (Python 3.12) | Extracts text, classifies document type, builds knowledge graph entities |
| Routing Analyzer | `src/preprocessor/routing_analyzer.py` | Classifies DocumentType + recommends ChunkingStrategy per document |
| Expertise Graph | Memgraph | Graph database for skills, patterns, AWS services, relationships (Neo4j bolt / openCypher) |
| Vector Store | PostgreSQL + pgvector | Chunk embeddings (1024-dim) for semantic retrieval via ivfflat ANN |
| Semantic Cache | PostgreSQL + pgvector | CAG `query_cache` table — cosine-similarity response cache (TTL 7 days) |
| Embeddings | Amazon Titan Text Embeddings V2 | 1024-dimensional semantic embeddings (shared module `embedder.py`) |
| Routing Graph | Amazon Neptune Analytics | Routing intelligence graph — EFFECTIVE_FOR edge weights + graph expansion |
| Query API | AWS Lambda + API Gateway HTTP v2 | POST /query-expertise — CAG check → classify → route → retrieve → graph-expand → synthesize → feedback |
| RAG Router | `src/query_api/rag_router.py` | Selects optimal retrieval strategy from Neptune EFFECTIVE_FOR weights; updates weights after each query |
| Orchestration | AWS Step Functions | Preprocess → StartIngestionJob → Poll loop |
| Encryption | AWS KMS | SSE-KMS on S3; key rotates annually |
| Observability | AWS X-Ray + CloudWatch | All Lambdas and Step Functions traced |
| IaC | AWS SAM (template.yaml) | Full infrastructure as code |
| CI/CD | GitHub Actions | OIDC → SAM validate → build → deploy → S3 upload |

### Cache-Augmented Generation (CAG)

ContextWeave includes a **semantic response cache** that short-circuits the entire RAG pipeline for repeated or near-identical questions.

**How it works:**

1. At query time, the question is embedded with Titan Text Embeddings V2 (1024-dim).
2. The `query_cache` table (PostgreSQL/pgvector) is searched for an unexpired entry with cosine similarity ≥ 0.95.
3. **Cache hit** → cached response returned immediately, skipping all downstream steps. Response includes `cacheHit: true`.
4. **Cache miss** → full 8-step RAG pipeline runs; if answer confidence ≥ 0.5, response is written to cache with a 7-day TTL.

**Time-sensitive bypass:** Questions containing keywords like `today`, `currently`, `latest`, `recent`, etc. always bypass the cache.

**`query_cache` table schema:**
- `question_embedding vector(1024)` — Titan V2 embedding of the question
- `response_json jsonb` — full serialised response (excluding `latencyMs`)
- `question_type text` — classified type for analytics
- `hit_count int` — incremented on every cache hit
- `expires_at timestamptz` — TTL-based expiry (default 7 days)

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

DocumentType and ChunkingStrategy nodes — and their relationships to each Document node — are written to Neptune Analytics alongside expertise nodes.

**At query time**, the `RAGRouter` reads `EFFECTIVE_FOR` edge weights from Neptune to select the optimal strategy:

| Strategy | When chosen | Behaviour |
|---|---|---|
| `graph_first` | skill_depth, architecture | pgvector retrieval + forced Memgraph graph expansion |
| `hybrid` | comparison | pgvector + Memgraph vectors + keyword boost |
| `keyword_boosted` | project, credential | pgvector + keyword-overlap reranking (25/75 blend) |
| `semantic_search` | general | pgvector semantic search only |

**After every query**, the winning strategy's `EFFECTIVE_FOR` edge weight is updated in Neptune:
- `confidence ≥ 0.70` → `weight += 0.05` (cap 1.00)
- `confidence < 0.40` → `weight -= 0.02` (floor 0.10)
- `0.40 ≤ confidence < 0.70` → no change

The graph learns from every answered question. No retraining. No manual tuning.

---

## Core Design Decisions

### 1. CAG + RAG Hybrid (semantic cache layer)
High-frequency or near-duplicate questions are served from a PostgreSQL/pgvector semantic cache (CAG) without touching the RAG pipeline. This eliminates redundant Bedrock inference calls and reduces p50 latency for warm questions to sub-100 ms. Only novel or time-sensitive questions pay the full RAG cost.

### 2. Memgraph for Expertise Graph + pgvector for Chunks
Memgraph (openCypher, Neo4j bolt protocol) stores the expertise graph — skills, patterns, AWS services, relationships. PostgreSQL/pgvector stores chunk embeddings for fast ivfflat ANN search. This separates graph traversal from vector retrieval and avoids Neptune Analytics costs for the hot read path. Neptune Analytics is retained for the routing intelligence graph (small, write-light).

### 3. Neptune Analytics for Routing Graph Only
Neptune Analytics provides EFFECTIVE_FOR edge weight queries and updates (the learning feedback loop). This is a small, write-light graph (strategy × question-type nodes only). Using Neptune keeps routing state durable and globally consistent without a separate cache-coherence layer.

### 4. Hierarchical Chunking (1500 / 300 tokens)
Parent chunks (1500 tokens) preserve full section context for synthesis; child chunks (300 tokens) are the retrieval units for precision. Overlap (60 tokens) ensures continuity across chunk boundaries.

### 5. Claude Haiku for Document Parsing
Bedrock's `BEDROCK_FOUNDATION_MODEL` parsing strategy uses Claude 3 Haiku to extract technical signals during ingestion — not just raw text. PlantUML component relationships, architecture patterns, and AWS service references are captured as structured evidence before entering the vector store.

### 6. Evidence Weighting at Query Time
Source credibility is encoded in the retrieval pipeline:
- `architecture.md`, `CLAUDE.md` → weight **1.0** (self-authored authoritative docs)
- PlantUML-derived summaries → weight **0.8** (diagram relationships)
- `repo-signals.yaml` → weight **0.7** (structured signals)
- Code / README → weight **0.6** (implementation evidence)
- Resume → weight **0.3** (self-reported, not verified)

### 7. OIDC-based GitHub Actions Deployment
No long-lived AWS credentials are stored in GitHub secrets. The workflow assumes IAM role `arn:aws:iam::239571291755:role/teamweave-github-actions-sam-deployer` via OIDC token exchange.

---

## Lambda Functions

### `expertise-rag-preprocessor-{env}`
- **Trigger**: S3 ObjectCreated on `raw/` prefix + EventBridge (Step Functions)
- **Input**: Raw files (`.md`, `.yaml`, `.puml`, `.txt`)
- **Output**: `derived/<repo>/*.derived.json`, `*.extracted.txt`, `graph_entities.json`, `graph_edges.json`, `expertise_signals.json`, `processing_manifest.json`
- **Routing**: Calls `routing_analyzer.analyze_document()` → writes `DocumentType`, `ChunkingStrategy`, `HAS_TYPE`, `CHUNKED_WITH` nodes/edges to Neptune
- **Code**: `src/preprocessor/handler.py`

### `expertise-rag-query-api-{env}`
- **Trigger**: API Gateway POST `/query-expertise`, GET `/health`
- **Pipeline**:
  - Step 0: Embed question → check CAG semantic cache (short-circuit on hit)
  - Step 1: classify_question()
  - Step 2: RAGRouter.select_strategy() (reads Neptune EFFECTIVE_FOR weights)
  - Step 3: retrieve_with_strategy() (pgvector ± Memgraph ± keyword boost)
  - Step 4: deduplicate_chunks()
  - Step 5: expand_graph_context() (Neptune openCypher traversal)
  - Step 6: synthesize_answer() (Bedrock Converse)
  - Step 7: write_cache() (if confidence ≥ 0.5)
  - Step 8: RAGRouter.update_feedback() (Neptune weight update)
- **Response shape**: `{ answer, sources, inferredSkills, repeatedPatterns, confidence, questionType, graphEntitiesUsed, routingDecision, cacheHit, latencyMs }`
- **Code**: `src/query_api/handler.py`

### `expertise-rag-db-init-{env}`
- **Trigger**: CloudFormation custom resource (post-deploy) + Step Functions + manual
- **Actions**: `start` (StartIngestionJob), `status` (GetIngestionJob), `seed_routing` (seed Neptune routing graph), `empty_all` (clear all Neptune data)
- **On Deploy**: Seeds Neptune with initial `EFFECTIVE_FOR` prior weights for all strategy/question-type pairs; initialises pgvector schema (`chunks` + `query_cache` tables)
- **Code**: `src/ingestion_trigger/handler.py`

---

## Shared Modules (`src/shared/`)

| Module | Purpose |
|--------|---------|
| `db_clients.py` | Singleton factories for Memgraph (Neo4j bolt) and PostgreSQL (psycopg2); credentials read from AWS Secrets Manager |
| `embedder.py` | Titan Text Embeddings V2 wrapper — `embed_text()` and `embed_texts()` |
| `chunker.py` | Text chunking utilities (hierarchical, sentence, fixed-window) |
| `models.py` | Canonical domain models, routing enums, ROUTING_PRIORS, SOURCE_WEIGHTS |

---

## Deployment

```bash
# Local development
sam validate --lint
sam build --parallel --cached
sam deploy --config-env default   # → expertise-rag-dev stack

# CI/CD (GitHub Actions)
# Push to main/master → auto-deploys to dev
# workflow_dispatch → choose dev / staging / prod
```

IAM role used by GitHub Actions:
```
arn:aws:iam::239571291755:role/teamweave-github-actions-sam-deployer
```

Stack names:
- `expertise-rag-dev`
- `expertise-rag-staging`
- `expertise-rag-prod`

Artifacts bucket (naming convention):
```
expertise-rag-artifacts-239571291755-{env}
```

---

## Repository Signal Files (uploaded to S3 post-deploy)

These files are uploaded to `s3://<bucket>/raw/contextweave/` by the GitHub Actions workflow after every successful SAM deploy:

| File | Purpose | Weight |
|------|---------|--------|
| `CLAUDE.md` | Authoritative AI context (this file) | 1.0 |
| `docs/architecture.md` | System architecture deep-dive | 1.0 |
| `repo-signals.yaml` | Structured expertise signals | 0.7 |
| `docs/c4.puml` | C4 system context + container diagram | 0.8 |
| `docs/aws-infrastructure.puml` | AWS service topology diagram | 0.8 |

---

## Key Technologies Demonstrated

- **AWS SAM** – Infrastructure as code for serverless applications
- **Amazon Bedrock** – Embeddings (Titan V2), LLM inference (Claude Converse API)
- **Amazon Neptune Analytics** – Routing intelligence graph; openCypher EFFECTIVE_FOR weight queries and graph expansion
- **Memgraph** – Expertise knowledge graph; openCypher graph traversal via Neo4j bolt driver
- **PostgreSQL + pgvector** – Chunk vector store (ivfflat ANN) + CAG semantic query cache
- **Cache-Augmented Generation (CAG)** – Embedding-keyed semantic response cache; cosine-similarity hit detection (threshold 0.95)
- **AWS Step Functions** – Long-running workflow orchestration with polling
- **AWS Lambda** – Python 3.12, AWS Lambda Powertools
- **Amazon API Gateway v2** – HTTP API with throttling and CORS
- **AWS KMS** – Customer-managed key with automatic rotation
- **GitHub Actions OIDC** – Keyless AWS authentication from CI/CD
- **GraphRAG** – Graph-augmented retrieval augmented generation pattern
- **Adaptive RAG Routing** – Self-improving routing graph that learns which strategy works best per question type
- **Hierarchical RAG chunking** – Parent/child chunk strategy for precision + context
- **Multi-strategy retrieval** – graph_first, hybrid, keyword_boosted, semantic_search

---

## Anti-patterns Explicitly Avoided

- ❌ No long-lived IAM access keys in GitHub secrets (OIDC used instead)
- ❌ No hardcoded bucket names (dynamic via CloudFormation outputs)
- ❌ No public S3 access (all buckets private, SSL-only policy enforced)
- ❌ No monolithic Lambda (three separate functions with single-responsibility)
- ❌ No polling in Lambda (Step Functions handles the wait loop)
- ❌ No plaintext at rest (SSE-KMS on all S3 objects)
- ❌ No full RAG pipeline for repeated questions (CAG cache short-circuits at step 0)
