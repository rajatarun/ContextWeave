# CLAUDE.md – ContextWeave / ExpertiseRAG

> Authoritative project context for Claude AI and other LLM-based tools.
> Evidence weight: **1.0** (highest authority signal in the RAG system).

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

ExpertiseRAG is a **serverless, event-driven GraphRAG system** built entirely on AWS managed services:

| Layer | Service | Role |
|-------|---------|------|
| Storage | Amazon S3 | Raw uploads (`raw/`) and derived artifacts (`derived/`) |
| Preprocessing | AWS Lambda (Python 3.12) | Extracts text, classifies document type, builds knowledge graph entities |
| Routing Analyzer | `src/preprocessor/routing_analyzer.py` | Classifies DocumentType + recommends ChunkingStrategy per document |
| Knowledge Graph | Amazon Neptune Analytics | Vector + graph hybrid store + routing intelligence graph |
| Embeddings | Amazon Titan Text Embeddings V2 | 1024-dimensional semantic embeddings |
| Knowledge Base | Amazon Bedrock Knowledge Base | Hierarchical chunking, hybrid search |
| Query API | AWS Lambda + API Gateway HTTP v2 | POST /query-expertise – classify → route → retrieve → graph-expand → synthesize → feedback |
| RAG Router | `src/query_api/rag_router.py` | Selects optimal retrieval strategy from Neptune EFFECTIVE_FOR weights; updates weights after each query |
| Orchestration | AWS Step Functions | Preprocess → StartIngestionJob → Poll loop |
| Encryption | AWS KMS | SSE-KMS on S3; key rotates annually |
| Observability | AWS X-Ray + CloudWatch | All Lambdas and Step Functions traced |
| IaC | AWS SAM (template.yaml) | Full infrastructure as code |
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

DocumentType and ChunkingStrategy nodes — and their relationships to each Document node — are written to Neptune Analytics alongside expertise nodes.

**At query time**, the `RAGRouter` reads `EFFECTIVE_FOR` edge weights from Neptune to select the optimal strategy:

| Strategy | When chosen | Behaviour |
|---|---|---|
| `graph_first` | skill_depth, architecture | Bedrock KB + forced graph expansion |
| `hybrid` | comparison | Bedrock KB + Neptune vector search + keyword boost |
| `keyword_boosted` | project, credential | Bedrock KB + keyword-overlap reranking (25/75 blend) |
| `semantic_search` | general | Bedrock KB semantic search only |

**After every query**, the winning strategy's `EFFECTIVE_FOR` edge weight is updated:
- `confidence ≥ 0.70` → `weight += 0.05` (cap 1.00)
- `confidence < 0.40` → `weight -= 0.02` (floor 0.10)
- `0.40 ≤ confidence < 0.70` → no change

The graph learns from every answered question. No retraining. No manual tuning.

---

## Core Design Decisions

### 1. Neptune Analytics (not RDS pgvector)
Neptune Analytics provides **vector + graph traversal in a single query**. This enables GraphRAG: after semantic retrieval, the system expands context by walking graph edges (co-occurring skills, pattern evidence, AWS service co-usage). A plain vector store cannot do this without separate graph database calls.

### 2. Hierarchical Chunking (1500 / 300 tokens)
Parent chunks (1500 tokens) preserve full section context for synthesis; child chunks (300 tokens) are the retrieval units for precision. Overlap (60 tokens) ensures continuity across chunk boundaries.

### 3. Claude Haiku for Document Parsing
Bedrock's `BEDROCK_FOUNDATION_MODEL` parsing strategy uses Claude 3 Haiku to extract technical signals during ingestion – not just raw text. This means PlantUML component relationships, architecture patterns, and AWS service references are captured as structured evidence before entering the vector store.

### 4. Evidence Weighting at Query Time
Source credibility is encoded in the retrieval pipeline:
- `architecture.md`, `CLAUDE.md` → weight **1.0** (self-authored authoritative docs)
- PlantUML-derived summaries → weight **0.8** (diagram relationships)
- `repo-signals.yaml` → weight **0.7** (structured signals)
- Code / README → weight **0.6** (implementation evidence)
- Resume → weight **0.3** (self-reported, not verified)

### 5. OIDC-based GitHub Actions Deployment
No long-lived AWS credentials are stored in GitHub secrets. The workflow assumes IAM role `arn:aws:iam::239571291755:role/teamweave-github-actions-sam-deployer` via OIDC token exchange. This follows AWS security best practices.

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
- **Pipeline**: classify_question → **RAGRouter.select_strategy** → retrieve_with_strategy (Bedrock ± Neptune) → expand_graph_context (Neptune) → synthesize_answer (Bedrock Converse) → **RAGRouter.update_feedback**
- **Response shape**: `{ answer, sources, inferredSkills, repeatedPatterns, confidence, questionType, graphEntitiesUsed, routingDecision }`
- **Code**: `src/query_api/handler.py`

### `expertise-rag-db-init-{env}`
- **Trigger**: CloudFormation custom resource (post-deploy) + Step Functions + manual
- **Actions**: `start` (StartIngestionJob), `status` (GetIngestionJob), `seed_routing` (seed Neptune routing graph), `empty_all` (clear all Neptune data)
- **On Deploy**: Seeds Neptune with initial `EFFECTIVE_FOR` prior weights for all strategy/question-type pairs
- **Code**: `src/ingestion_trigger/handler.py`

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
- **Amazon Bedrock** – Knowledge Bases, embeddings (Titan V2), LLM inference (Claude)
- **Amazon Neptune Analytics** – Graph + vector hybrid store, openCypher, routing intelligence graph
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
