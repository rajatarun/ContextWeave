# CLAUDE.md – ContextWeave / ExpertiseRAG

> Authoritative project context for Claude AI and other LLM-based tools.
> Evidence weight: **1.0** (highest authority signal in the RAG system).

---

## Project Identity

**Name**: ContextWeave / ExpertiseRAG
**Type**: AWS-native GraphRAG + CAG platform
**Purpose**: Answers deep, evidence-backed questions about a developer's professional expertise using retrieval-augmented generation with a knowledge graph and a semantic response cache.
**Owner**: Tarun Raja (rajatarun)
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

**At query time**, the `RAGRouter` reads each strategy's `EFFECTIVE_FOR` posterior — `Beta(alpha, beta)` per (strategy, question type) — and selects by **Thompson sampling**: one draw per strategy, highest draw wins. A strategy is therefore selected roughly in proportion to the probability that it is the best one, so an untried strategy still gets tried. The scalar `weight` on the edge is maintained as the posterior mean so the policy stays readable as a table.

| Strategy | When chosen | Behaviour |
|---|---|---|
| `graph_first` | skill_depth, architecture | pgvector retrieval + forced Memgraph graph expansion |
| `hybrid` | comparison | pgvector + Memgraph vectors + keyword boost |
| `keyword_boosted` | project, credential | pgvector + keyword-overlap reranking (25/75 blend) |
| `semantic_search` | general | pgvector semantic search only |

**After every query**, the synthesis confidence `c ∈ [0,1]` is folded into the selected strategy's posterior as a fractional Bernoulli reward:
- `alpha += c`, `beta += 1 − c`
- `weight = alpha / (alpha + beta)` (posterior mean; kept for dashboards and legacy queries)

Every observation moves the posterior — there is no confidence band in which feedback is discarded and no ceiling at which it saturates.

**Only reported confidences are rewards.** When the model omits the `confidence` field, returns non-JSON, or the call fails, the response carries a fallback constant (0.7 / 0.5 / 0.0) and `confidenceReported: false`; the router is **not** updated in that case and the decision is recorded with a NULL confidence. A constant the code chose is not an observation. `docs/confidence-semantics.md` states what every score in this and the sibling systems means and when they may be combined.

**Second reward source.** The self-confidence above is the model's opinion of its own answer; a confidently wrong answer reinforces the strategy that produced it. Every response therefore carries a `queryId`, and `POST /feedback {"queryId", "rating"}` (rating: `up`/`down`/`neutral`, `true`/`false`, or a number in `[0,1]`) folds an independent rating into the same posterior with weight `ROUTER_HUMAN_FEEDBACK_WEIGHT` (default 2.0 — one rating counts as two self-assessments), once per `queryId`. Human and self observations are counted separately on the edge (`human_feedback_count`, `feedback_count`). Decisions are recorded in the `routing_decisions` table (query id, question type, strategy, propensity, self-confidence, rating), which is also the data an operator needs to check whether self-confidence predicts ratings at all.

`GET /routing-decisions` reads that table back out (`src/query_api/routing_decisions_api.py`). `?mode=list` pages the raw decisions (`questionType`, `strategy`, `since`, `limit`, `offset` filters, newest first); `?mode=summary` groups by (question type, strategy) and reports `meanAbsDiff` — `AVG(ABS(confidence - rating))` over the decisions that carry **both** a reported self-confidence and a rating. That number is the calibration check: near 0 the self-assessment tracks what people think and is worth using as the cheap always-available reward; near 0.5 the router has been learning from a proxy that measures nothing. It is null, never 0.0, for a group nobody has rated. Read-only and free of PII — the table holds query ids, strategy labels and two scores, never the question, the answer, or anything about the caller. Consumed by the platform `/observability` console (weave-platform E10).

Each decision also records its **selection propensity** (`routingDecision.selectionPropensity`), the probability Thompson sampling had of choosing that strategy, so a different routing policy can later be evaluated from the decision log by inverse-propensity weighting without being deployed.

> **History.** Until September 2026 the router selected by `argmax` over the scalar weight and applied fixed steps (+0.05 above 0.70, −0.02 below 0.40, nothing in between). That is a bandit with no exploration: only the selected strategy was ever updated, so a challenger was never tried and its weight never moved. With an incumbent answering inside the [0.40, 0.70) dead band, a strategy that would have answered at 0.90 was selected 0 times in 2000 simulated queries. The failure was silent — a frozen router and a converged one log identically. `GET /health` now reports `routingGraph.health` with a per-question-type verdict of `learning` / `converged` / `starved`; `starved` (one strategy heavily observed while siblings have none) is the signature of that defect and should alert. `scripts/routing_regret_sim.py` reproduces the comparison offline.

The graph learns from every answered question. No retraining. No manual tuning.

---

## The health store — a separate database, behind the only authorizer

A medical record does not belong in the expertise corpus, and four properties
of that corpus say why. Each is answered by a layer of this store rather than
by a convention:

| The expertise corpus | The health store |
|---|---|
| `POST /query-expertise` has **no authorizer** — nor does any route on that API | every `/health/*` route requires the SIWE bearer token |
| retrieval searches one undifferentiated `chunks` table (its only filters are the query's own and `embedding IS NOT NULL`) | its own **database** and role; the expertise role has no `CONNECT` |
| no per-document delete: re-ingestion replaces a file's own chunks, and the graph's only removal is `MATCH (n) DETACH DELETE n` | `DELETE /health/documents/{docId}`, and deleting the S3 object withdraws the record |
| every answer cached in `query_cache` for 7 days | nothing is cached; `cacheHit` is always false |

Without the first, anyone with the URL could read the record. Without the
second, `linkedin_quick_post` asking about work under pressure could retrieve a
chunk of a discharge summary and put it in a draft — cosine similarity does not
know what it has found. Without the third, ingesting would be one-way. Without
the fourth, an answer would outlive the delete of the record it came from.

**Health documents never enter the knowledge graph.** Memgraph and Neptune are
shared with the expertise path and have no per-document delete, so a graph write
is the one thing `delete_document` could not take back. The cost is real: no
entity expansion for health questions, only vector retrieval over the record.

**The authorizer is declared but is not the default.** Every expertise route is
open deliberately — the A2A card has to be readable without a token or
discovery cannot work, and TeamWeave calls `/query-expertise` with none. A
`DefaultAuthorizer` would break both, which is exactly what TeamWeave's own API
did when its card answered `401` to the clients it exists to inform. The health
routes opt in one at a time, and `HealthEnabled` means no authorizer configured
→ no health resources at all, so the surface cannot ship open by accident.

Three smaller decisions worth knowing:

- **An empty retrieval is not an answer.** `found: false` and an empty string,
  never a model answering from general knowledge — that would present its
  recollection as the person's own chart.
- **A failure never echoes the database error.** A psycopg2 message can carry a
  row, and a row here is a medical record.
- **`DELETE` answers 404 when nothing was removed.** "It is gone" and "it was
  never here" have to be different answers, or you cannot confirm a record is
  deleted.

The embedder is injected rather than imported at module scope, so listing and
deleting work when the Bedrock and observability stack does not — a withdrawal
must not depend on the thing that ingested it still being healthy.

`tests/test_health_store.py` holds all of it, and ten mutations — dropping the
route's authorizer, adding a default one, falling back to the expertise
database, pointing the SQL at `chunks`, answering an empty retrieval, losing the
404, echoing the error, dropping removal events, granting `s3:PutObject`,
reading unsupported file types — each fail it.

## A2A Agent Card — how siblings find this service

`GET /.well-known/agent-card.json` publishes ContextWeave as an
[A2A](https://github.com/a2aproject/A2A) agent (v1.0, Linux Foundation).
`src/query_api/agent_card.py` builds it; `handler.py` serves it before every
other route, because it is how a caller learns the other routes exist.

**Why it exists.** TeamWeave reached this service by assuming
`{CONTEXTWEAVE_URL}/query-expertise` — a path constant in *TeamWeave's*
repository. Move a route here and the break surfaces there as a 404 its RAG
layer degrades past in silence: the run loses its grounding and nobody is
told. The card replaces that guess with a document.

Each route a caller uses is a **skill** (`query-expertise`, `feedback`,
`routing-decisions`) and each description names its path. The interface URL is
derived from the request, not configured, so the card cannot advertise a URL
this deployment does not serve. `capabilities.streaming` is false and no A2A
transport binding is claimed beyond `HTTP+JSON`, because this service serves
its own HTTP API and does not implement `message:send` — advertising a
protocol that is not there is a lie a machine acts on.

`tests/test_agent_card.py` drives the real handler rather than grepping it: a
source-level assertion passed happily while the branch that serves the card
was disabled.

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
- **Trigger**: API Gateway POST `/query-expertise`, POST `/feedback`, GET `/health`, GET `/routing-decisions`
- **Pipeline**:
  - Step 0: Embed question → check CAG semantic cache (short-circuit on hit)
  - Step 1: classify_question()
  - Step 2: RAGRouter.select_strategy() (reads Neptune EFFECTIVE_FOR weights)
  - Step 3: retrieve_with_strategy() (pgvector ± Memgraph ± keyword boost)
  - Step 4: deduplicate_chunks()
  - Step 5: expand_graph_context() (Neptune openCypher traversal)
  - Step 6: synthesize_answer() (Bedrock Converse)
  - Step 7: write_cache() (if confidence ≥ 0.5)
  - Step 8: RAGRouter.update_feedback() (posterior update from self-confidence)
  - Step 9: feedback.record_decision() (so POST /feedback can rate this answer later)
- **Response shape**: `{ queryId, answer, sources, inferredSkills, repeatedPatterns, confidence, questionType, graphEntitiesUsed, routingDecision, cacheHit, latencyMs }`
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
| `docs/weave-platform.md` | Portfolio map and integration plan for all weave repositories | 1.0 |

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
