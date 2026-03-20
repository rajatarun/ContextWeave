# ExpertiseRAG – Architecture Overview

## System Purpose

ExpertiseRAG is an AWS-native GraphRAG + CAG platform designed to answer deep,
evidence-backed questions about a developer's professional expertise. It combines:

1. **Cache-Augmented Generation (CAG)** – PostgreSQL/pgvector semantic response
   cache; near-identical questions are served instantly without touching the RAG
   pipeline
2. **Semantic retrieval** – Amazon Titan Text Embeddings V2 (1024-dim) stored in
   PostgreSQL/pgvector, queried via ivfflat ANN
3. **Graph traversal** – Memgraph GraphRAG to surface co-occurring skills,
   repeated patterns, and cross-repository evidence
4. **Adaptive routing** – an agentic layer that classifies documents and queries,
   then selects the best RAG strategy from a self-improving Neptune routing graph
5. **Source weighting** – architecture docs and CLAUDE.md treated as
   authoritative; resume as supporting evidence only
6. **Evidence synthesis** – Bedrock Converse API generates grounded,
   citation-backed answers

---

## High-Level Architecture

```
                        ┌─────────────────────────────┐
  Developer uploads     │          S3 Bucket           │
  raw repo files ──────▶│  raw/<repo>/architecture.md  │
  (architecture.md,     │  raw/<repo>/CLAUDE.md        │
   CLAUDE.md,           │  raw/<repo>/*.puml            │
   *.yaml, …)           │  raw/<repo>/repo-signals.yaml│
                        └──────────────┬──────────────┘
                                       │ S3 ObjectCreated
                                       ▼
                        ┌─────────────────────────────┐
                        │   Preprocessor Lambda        │
                        │   (Python 3.12)              │
                        │  • extract_markdown()        │
                        │  • extract_yaml()            │
                        │  • extract_plantuml()        │
                        │  • routing_analyzer()   ★    │
                        │  • GraphBuilder              │
                        └──────────────┬──────────────┘
                                       │ writes derived/
                                       ▼
                        ┌─────────────────────────────┐
                        │         S3 Bucket            │
                        │  derived/<repo>/*.derived.json│
                        │  derived/<repo>/*.extracted.txt│
                        │  derived/<repo>/graph_entities.json│
                        │  derived/<repo>/graph_edges.json│
                        │  derived/<repo>/expertise_signals.json│
                        │  derived/<repo>/processing_manifest.json│
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┼────────────────────────┐
                    │                  │                        │
                    ▼                  ▼                        │
     ┌──────────────────────┐  ┌──────────────────────┐        │
     │  Step Functions       │  │  Bedrock KB Ingestion│        │
     │  (Orchestration)      │  │  Custom Resource      │        │
     │  • Preprocess         │  │  (post-deploy)        │        │
     │  • Start ingestion    │  └────────────┬─────────┘        │
     │  • Poll until done    │               │                  │
     └──────────────────────┘               │                  │
                                            ▼                  │
                        ┌─────────────────────────────┐        │
                        │  Chunk Embeddings             │        │
                        │  (Titan Text Embeddings V2)  │        │
                        │  stored in pgvector          │        │
                        └──────────────┬──────────────┘        │
                                       │                        │
                    ┌──────────────────┴────────────┐          │
                    ▼                               ▼           │
     ┌──────────────────────────┐  ┌─────────────────────────┐ │
     │  PostgreSQL + pgvector   │  │   Memgraph              │◀┘
     │  • chunks table          │  │   • Expertise graph      │
     │  • query_cache table ★   │  │   • openCypher queries  │
     │  • ivfflat ANN search    │  │   • Neo4j bolt driver   │
     └──────────────────────────┘  └─────────────────────────┘
                                                  ▲
                        ┌─────────────────────────────┐
                        │   Neptune Analytics          │
                        │   • Routing graph            │
                        │   • EFFECTIVE_FOR weights ★  │
                        │   • Graph expansion queries  │
                        └─────────────────────────────┘
                                       ▲▼
                    Query time:        │
                                       │
     ┌─────────────┐    ┌─────────────┴────────────────────────┐
     │  API Gateway │    │   Query API Lambda                    │
     │  HTTP API v2 │───▶│  POST /query-expertise               │
     └─────────────┘    │  0. embed + check CAG cache ★        │
                        │  1. classify_question()               │
                        │  2. RAGRouter.select_strategy() ★     │
                        │  3. retrieve_with_strategy() ★        │
                        │  4. deduplicate_chunks()              │
                        │  5. expand_graph_context()            │
                        │  6. synthesize_answer()               │
                        │  7. write_cache() (if conf ≥ 0.5) ★  │
                        │  8. RAGRouter.update_feedback() ★     │
                        └──────────────────────────────────────┘
```

★ = adaptive routing + CAG components

---

## Component Details

### S3 Bucket

- **Prefix layout**:
  - `raw/<repo-name>/` – original uploads (triggers preprocessor)
  - `derived/<repo-name>/` – preprocessor output (fed to Bedrock KB ingestion)
- **Security**: SSE-KMS, versioning, public access blocked, SSL-only policy
- **Lifecycle**: Non-current versions → IA after 30d, expire after 365d

### Preprocessor Lambda

Triggered by S3 ObjectCreated events on `raw/` prefix.

**Extractors**:

| File Type | Extractor | Output |
|-----------|-----------|--------|
| `.md`, `.markdown` | `extract_markdown()` | Clean text, heading structure, code blocks |
| `.yaml`, `.yml` | `extract_yaml()` | Flattened key/value pairs, skill signals from well-known keys |
| `.puml`, `.plantuml` | `extract_plantuml()` | Component list, relationship arrows, AWS sprite references, prose summary |
| All others | `extract_text()` | Raw text with signal extraction |

**Routing Analyzer** (`routing_analyzer.py`):

After extraction, each document is classified using structural text statistics:

| Metric | How computed |
|--------|-------------|
| `heading_count` | Count of `#`/`##` lines |
| `code_char_ratio` | Characters inside fenced code blocks ÷ total chars |
| `avg_sentence_len` | Mean word count per sentence (prose only) |
| `has_yaml_keys` | Regex match for `key: value` patterns |
| `has_plantuml` | `@startuml` / `@startc4` marker |

Classification rules (priority order):

```
diagram_derived  ← .puml extension or @startuml marker
code             ← .py/.ts/.js/… or code_char_ratio > 0.35
structured_data  ← .yaml/.json or has_yaml_keys
technical_spec   ← heading_count ≥ 3 AND aws_signal_count ≥ 2
narrative        ← everything else
```

Chunking strategy assigned per type:

| DocumentType | ChunkingStrategy | Rationale |
|---|---|---|
| `technical_spec` | `hierarchical` (1500/300 tokens) | Section context is critical |
| `narrative` | `sentence` | Preserve prose flow |
| `structured_data` | `fixed_256` | Short dense entries |
| `code` | `fixed_512` | Preserve function scope |
| `diagram_derived` | `fixed_256` | Already distilled prose |

**Graph Builder** (`graph_builder.py`):
- Seeds `Person` and `Repository` nodes
- For each file: creates `Document` node, `CONTAINS` edge
- For each signal: creates typed nodes (Skill, Technology, AWSService, Pattern)
- Cross-entity `STRENGTHENS` edges connect related skills/patterns
- `add_routing_metadata()` writes `DocumentType` + `ChunkingStrategy` nodes and
  `HAS_TYPE` / `CHUNKED_WITH` edges per document

**Derived artifacts** written per file and per repo:
```
derived/<repo>/
  <file>.derived.json       # full extraction envelope
  <file>.extracted.txt      # clean text for Bedrock ingestion
  graph_entities.json       # all nodes (incl. DocumentType + ChunkingStrategy)
  graph_edges.json          # all edges (incl. HAS_TYPE + CHUNKED_WITH)
  expertise_signals.json    # aggregated + deduplicated signals
  processing_manifest.json  # statistics, error log, routing_summary
```

`routing_summary` in the manifest:
```json
{
  "routing_summary": {
    "technical_spec:hierarchical": 2,
    "structured_data:fixed_256": 1,
    "diagram_derived:fixed_256": 2
  }
}
```

### PostgreSQL + pgvector

Used for two purposes:

**1. Chunk vector store** (`chunks` table):

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Primary key |
| `doc_id` | TEXT | Source document identifier |
| `source_file` | TEXT | Originating S3 key |
| `doc_type` | TEXT | DocumentType label |
| `strategy` | TEXT | ChunkingStrategy label |
| `content` | TEXT | Chunk text |
| `embedding` | vector(1024) | Titan V2 embedding |
| `parent_content` | TEXT | Parent chunk text (hierarchical) |
| `is_child` | BOOLEAN | True for child chunks |
| `metadata` | JSONB | Arbitrary metadata |

Indexed with `ivfflat` (cosine ops, 100 lists) for ANN search.

**2. CAG semantic response cache** (`query_cache` table):

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Primary key |
| `question_embedding` | vector(1024) | Titan V2 embedding of the question |
| `response_json` | JSONB | Full serialised response |
| `question_type` | TEXT | Classified question type |
| `hit_count` | INTEGER | Cache hit counter |
| `created_at` | TIMESTAMPTZ | Insert timestamp |
| `expires_at` | TIMESTAMPTZ | TTL expiry (default +7 days) |

Indexed with `ivfflat` (cosine ops, 50 lists) and a separate expiry index.

Cache lookup performs:
```sql
SELECT id, response_json FROM query_cache
WHERE expires_at > NOW()
  AND 1 - (question_embedding <=> $1::vector) >= 0.95
ORDER BY question_embedding <=> $1::vector
LIMIT 1
```

### Memgraph (Expertise Graph)

- **Protocol**: Neo4j bolt (port 7687); Community Edition runs without auth
- **Query language**: openCypher
- **Driver**: `neo4j` Python driver, cached singleton via `db_clients.get_memgraph_driver()`
- **Use at query time**: expertise graph traversal — skill neighbourhood, pattern evidence, AWS service context

**Expertise graph node types**:

| Type | Description |
|------|-------------|
| `Person` | The developer (central node) |
| `Repository` | A git repository or project |
| `Document` | A specific file within a repo |
| `Skill` | A demonstrated professional skill |
| `Pattern` | An architecture or design pattern |
| `Technology` | A programming language, framework, or tool |
| `AWSService` | An AWS service used in a project |
| `ArchitectureStyle` | A high-level style (serverless, microservices) |
| `Evidence` | A specific piece of supporting evidence |
| `Claim` | A skill/expertise claim backed by evidence |

**Expertise graph edge types**:

| Type | Direction | Meaning |
|------|-----------|---------|
| `BUILT` | Person → Repository | Developer built this repo |
| `CONTAINS` | Repository → Document | Repo contains this doc |
| `USES_TECH` | Document/Repo → Technology | Tech used in this context |
| `USES_AWS_SERVICE` | Document/Repo → AWSService | AWS service used |
| `DEMONSTRATES_PATTERN` | Document/Person → Pattern | Pattern demonstrated |
| `SUPPORTS_CLAIM` | Evidence → Claim | Evidence backs this claim |
| `INDICATES_SKILL` | Document → Skill | Document indicates skill |
| `DEMONSTRATES_SKILL` | Person → Skill | Developer demonstrates this skill |
| `STRENGTHENS` | Skill/Tech → Pattern/Service | Corroborating relationship |

### Neptune Analytics (Routing Graph)

- **Type**: `AWS::NeptuneGraph::Graph`
- **Vector dimension**: 1024 (matches Titan Text Embeddings V2)
- **Query language**: openCypher
- **Use**: Routing intelligence graph (EFFECTIVE_FOR weights) + graph expansion queries

**Routing graph node types**:

| Type | Description |
|------|-------------|
| `DocumentType` | One per doc type: `technical_spec`, `narrative`, `structured_data`, `code`, `diagram_derived`. Also used as question-type proxies (`skill_depth`, `architecture`, …) |
| `ChunkingStrategy` | One per strategy: `hierarchical`, `sentence`, `fixed_512`, `fixed_256` |
| `RAGStrategy` | One per strategy: `graph_first`, `hybrid`, `keyword_boosted`, `semantic_search` |

**Routing graph edge types**:

| Type | Direction | Meaning |
|------|-----------|---------|
| `HAS_TYPE` | Document → DocumentType | Document classified as this type |
| `CHUNKED_WITH` | Document → ChunkingStrategy | Document chunked with this strategy |
| `EFFECTIVE_FOR` | RAGStrategy → DocumentType | Learned weight: how effective this strategy is for this question type |

`EFFECTIVE_FOR` edges are the core of the learning loop. Initial weights seeded from `ROUTING_PRIORS` in `models.py`:

```
graph_first    → skill_depth:  0.80
graph_first    → architecture: 0.80
keyword_boosted→ credential:   0.80
hybrid         → comparison:   0.70
semantic_search→ general:      0.70
…
```

After every query: `weight += 0.05` (confidence ≥ 0.70) or `weight -= 0.02`
(confidence < 0.40). Weights clamped to [0.10, 1.00].

### Shared Modules (`src/shared/`)

| Module | Purpose |
|--------|---------|
| `db_clients.py` | Singleton factories for Memgraph (`get_memgraph_driver()`) and PostgreSQL (`get_pg_connection()`); credentials from AWS Secrets Manager (`MEMGRAPH_SECRET_ARN`, `POSTGRES_SECRET_ARN`) or env vars |
| `embedder.py` | Titan Text Embeddings V2 — `embed_text(text)` → `list[float]` (1024-dim) |
| `chunker.py` | Text chunking utilities — hierarchical, sentence-boundary, fixed-window |
| `models.py` | Canonical domain models, routing enums, `ROUTING_PRIORS`, `SOURCE_WEIGHTS` |

### Query API Lambda

**Full reasoning pipeline** (per-request):

0. **CAG cache check** – embed question with Titan V2; search `query_cache` for cosine similarity ≥ 0.95; time-sensitive questions bypass this step
   - **Cache HIT** → return cached response immediately (`cacheHit: true`)
   - **Cache MISS** → continue to step 1
1. **Classify** – regex patterns categorise question as: `skill_depth`, `architecture`, `project`, `comparison`, `credential`, or `general`
2. **Route** – `RAGRouter.select_strategy()` reads `EFFECTIVE_FOR` edge weights from Neptune and picks the highest-weighted `RAGStrategy` for the question type; falls back to hard-coded priors when no feedback data exists
3. **Retrieve** – `retrieve_with_strategy()` dispatches to the chosen strategy:
   - `semantic_search`: pgvector ANN search
   - `graph_first`: pgvector + forced Memgraph graph expansion
   - `keyword_boosted`: pgvector + keyword-overlap reranking (Jaccard, stopword-filtered, 25/75 blend)
   - `hybrid`: pgvector + Memgraph vectors + keyword boost + result merging
4. **Deduplicate** – Jaccard similarity prefix dedup removes near-duplicate chunks
5. **Graph expand** – Neptune Analytics openCypher queries surface skill neighbourhood, pattern evidence, AWS service context, person summary. Forced for `graph_first` and `hybrid` strategies.
6. **Synthesize** – Bedrock Converse API with grounded system prompt; model response is JSON-structured
7. **Cache write** – if confidence ≥ 0.5: write response to `query_cache` with 7-day TTL (non-fatal, skipped on error)
8. **Feedback** – `RAGRouter.update_feedback()` adjusts the `EFFECTIVE_FOR` edge weight in Neptune based on synthesis confidence

**`routingDecision` field in every response**:
```json
{
  "strategy": "graph_first",
  "questionType": "architecture",
  "strategyConfidence": 0.82,
  "graphExpansionForced": true,
  "keywordBoostApplied": false,
  "neptuneVectorsUsed": false
}
```

**Full response shape**:
```json
{
  "answer": "...",
  "sources": [...],
  "inferredSkills": [...],
  "repeatedPatterns": [...],
  "confidence": 0.92,
  "questionType": "architecture",
  "graphEntitiesUsed": [...],
  "retrievalCount": 8,
  "modelId": "anthropic.claude-3-5-sonnet-...",
  "routingDecision": { ... },
  "cacheHit": false,
  "latencyMs": 1840
}
```

### RAG Router (`rag_router.py`)

The routing agent lives in `src/query_api/rag_router.py`:

- `select_strategy(question_type, graph_id)` — queries Neptune `EFFECTIVE_FOR` edges, picks the strategy with the highest weight; returns a `RetrievalConfig`
- `update_feedback(strategy, question_type, confidence, graph_id)` — mutates the `EFFECTIVE_FOR` edge weight after each query
- `seed_routing_graph(graph_id)` — idempotent upsert of initial `RAGStrategy` + `DocumentType` nodes and `EFFECTIVE_FOR` edges; called from `IngestionTriggerFunction` on CloudFormation Create/Update

### Ingestion Trigger Lambda

On **CloudFormation Create/Update**, this Lambda:
1. Seeds the routing graph in Neptune (`seed_routing_graph()`)
2. Initialises the pgvector schema (`chunks` + `query_cache` tables)
3. Starts the Bedrock KB ingestion job
4. Polls until complete

On **direct invocation** with `action=seed_routing`, re-seeds the routing graph
without triggering ingestion (useful for resetting after graph migration).

### Step Functions State Machine

Orchestrates full ingestion workflow:

```
PreprocessRawFiles ──→ StartIngestionJob ──→ WaitForIngestion
                                                    │
                                         CheckIngestionStatus
                                                    │
                              ┌─────────────────────┴────────────────────┐
                              ▼                                           ▼
                        COMPLETE → IngestionSuccess           FAILED → IngestionFailed
                              ▼
                        (loop back to WaitForIngestion)
```

---

## Evidence Weighting

The system prioritises sources in this order:

| Source | Weight | Rationale |
|--------|--------|-----------|
| `architecture.md` | 1.0 | Authoritative architecture decisions |
| `CLAUDE.md` | 1.0 | Authoritative project context |
| PlantUML-derived summaries | 0.8 | Architecture diagrams → component relationships |
| `repo-signals.yaml` | 0.7 | Structured expert signals |
| `code`, `README.md` | 0.6 | Implementation evidence |
| Articles / blog posts | 0.5 | Published thinking |
| Resume | 0.3 | Self-reported, lower authority |

**Effective score** = raw retrieval score × source weight

**Repeated patterns** (implemented in multiple repos) are weighted higher than one-off mentions.

---

## IAM Security Model

All roles follow least-privilege:

| Role | Permissions |
|------|-------------|
| `PreprocessorRole` | S3 GetObject (raw/), S3 PutObject (derived/), KMS |
| `QueryAPIRole` | Bedrock InvokeModel (Titan embeddings + Converse), Neptune ReadDataViaQuery + ExecuteQuery (routing graph), Secrets Manager GetSecretValue (Memgraph + Postgres creds) |
| `IngestionTriggerRole` | Bedrock StartIngestionJob, GetIngestionJob, Neptune ExecuteQuery (seed routing graph), Secrets Manager GetSecretValue, S3 ListBucket |
| `BedrockKnowledgeBaseRole` | S3 ListBucket+GetObject (all), Bedrock InvokeModel (Titan) |
| `StepFunctionsRole` | Lambda InvokeFunction (Preprocessor + Trigger only), CloudWatch Logs, X-Ray |

---

## Deployment Flow

1. `sam deploy` creates all resources
2. CloudFormation custom resource (`PostDeploymentIngestionJob`) seeds the Neptune routing graph, initialises pgvector schema, then triggers an initial ingestion job
3. Upload repo files to `s3://<bucket>/raw/<repo>/`
4. S3 event triggers `PreprocessorFunction`
5. Preprocessor classifies each document, writes derived artifacts and routing metadata to Neptune
6. Trigger a new ingestion job (via `scripts/start_ingestion.py` or Step Functions) to embed chunks into pgvector
7. Query via `POST /query-expertise` — CAG cache checked first; on miss, routing agent selects strategy, answers, updates Neptune weights, writes to cache
