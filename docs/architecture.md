# ExpertiseRAG – Architecture Overview

## System Purpose

ExpertiseRAG is an AWS-native Retrieval-Augmented Generation (RAG) platform
designed to answer deep, evidence-backed questions about a developer's
professional expertise. It goes beyond simple keyword search by combining:

1. **Semantic retrieval** – Amazon Bedrock Knowledge Bases with Titan Text
   Embeddings V2 (1024-dim)
2. **Graph traversal** – Neptune Analytics GraphRAG to surface co-occurring
   skills, repeated patterns, and cross-repository evidence
3. **Adaptive routing** – an agentic layer that classifies documents and
   queries, then selects the best RAG strategy from a self-improving Neptune
   routing graph
4. **Source weighting** – architecture docs and CLAUDE.md treated as
   authoritative; resume as supporting evidence only
5. **Evidence synthesis** – Bedrock foundation model generates grounded,
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
                    ┌──────────────────┼───────────────────┐
                    │                  │                   │
                    ▼                  ▼                   │
     ┌──────────────────────┐  ┌──────────────────────┐   │
     │  Step Functions       │  │  Bedrock KB Ingestion│   │
     │  (Orchestration)      │  │  Custom Resource      │   │
     │  • Preprocess         │  │  (post-deploy)        │   │
     │  • Start ingestion    │  └────────────┬─────────┘   │
     │  • Poll until done    │               │             │
     └──────────────────────┘               │             │
                                            ▼             │
                        ┌─────────────────────────────┐   │
                        │  Bedrock Knowledge Base       │   │
                        │  • Titan Text Embeddings V2  │   │
                        │  • Hierarchical chunking      │   │
                        │  • Claude Haiku for parsing   │   │
                        └──────────────┬──────────────┘   │
                                       │ stores vectors    │
                                       ▼                   │
                        ┌─────────────────────────────┐   │
                        │   Neptune Analytics Graph    │◀──┘
                        │   • Vector search (1024-dim) │
                        │   • openCypher queries       │
                        │   • Expertise graph (skills, │
                        │     patterns, AWS services)  │
                        │   • Routing graph (strategy  │ ★
                        │     weights + feedback)      │ ★
                        └─────────────────────────────┘
                                       ▲▼
                    Query time:        │
                                       │
     ┌─────────────┐    ┌─────────────┴────────────────────┐
     │  API Gateway │    │   Query API Lambda                │
     │  HTTP API v2 │───▶│  POST /query-expertise           │
     └─────────────┘    │  1. classify_question()           │
                        │  2. RAGRouter.select_strategy() ★ │
                        │  3. retrieve_with_strategy()    ★ │
                        │  4. expand_graph_context()        │
                        │  5. synthesize_answer()           │
                        │  6. RAGRouter.update_feedback() ★ │
                        └──────────────────────────────────┘
```

★ = new adaptive routing components

---

## Component Details

### S3 Bucket

- **Prefix layout**:
  - `raw/<repo-name>/` – original uploads (triggers preprocessor)
  - `derived/<repo-name>/` – preprocessor output (fed to Bedrock KB)
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

**Routing Analyzer** (`routing_analyzer.py`) — new:

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
- **New**: `add_routing_metadata()` writes `DocumentType` + `ChunkingStrategy`
  nodes and `HAS_TYPE` / `CHUNKED_WITH` edges per document

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

`routing_summary` in the manifest shows how many files of each type were processed:
```json
{
  "routing_summary": {
    "technical_spec:hierarchical": 2,
    "structured_data:fixed_256": 1,
    "diagram_derived:fixed_256": 2
  }
}
```

### Neptune Analytics Graph

- **Type**: `AWS::NeptuneGraph::Graph`
- **Vector dimension**: 1024 (matches Titan Text Embeddings V2)
- **Query language**: openCypher
- **Use at query time**: expertise graph expansion + routing strategy selection

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

**Routing graph node types** (new):

| Type | Description |
|------|-------------|
| `DocumentType` | One per doc type: `technical_spec`, `narrative`, `structured_data`, `code`, `diagram_derived`. Also used as question-type proxies (`skill_depth`, `architecture`, …) |
| `ChunkingStrategy` | One per strategy: `hierarchical`, `sentence`, `fixed_512`, `fixed_256` |
| `RAGStrategy` | One per strategy: `graph_first`, `hybrid`, `keyword_boosted`, `semantic_search` |

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

**Routing graph edge types** (new):

| Type | Direction | Meaning |
|------|-----------|---------|
| `HAS_TYPE` | Document → DocumentType | Document classified as this type |
| `CHUNKED_WITH` | Document → ChunkingStrategy | Document chunked with this strategy |
| `EFFECTIVE_FOR` | RAGStrategy → DocumentType | Learned weight: how effective this strategy is for this question type |

`EFFECTIVE_FOR` edges are the core of the learning loop. Initial weights are
seeded from `ROUTING_PRIORS` in `models.py`:

```
graph_first    → skill_depth:  0.80
graph_first    → architecture: 0.80
keyword_boosted→ credential:   0.80
hybrid         → comparison:   0.70
semantic_search→ general:      0.70
…
```

After every query: `weight += 0.05` (confidence ≥ 0.70) or `weight -= 0.02`
(confidence < 0.40). Weights are clamped to [0.10, 1.00].

### Bedrock Knowledge Base

- **Storage**: Neptune Analytics (vector + graph hybrid)
- **Embedding**: Amazon Titan Text Embeddings V2, 1024 dimensions
- **Chunking**: Hierarchical (1500 token parents, 300 token children, 60 token overlap)
- **Parsing**: Claude 3 Haiku with domain-specific prompt extracting technical signals
- **Search**: HYBRID (semantic + keyword)

### Bedrock Data Source

- Points to both `raw/` and `derived/` S3 prefixes
- The `derived/*.extracted.txt` files are the primary ingestion target
- The `derived/*.derived.json` files provide structured metadata

### Query API Lambda

**Adaptive reasoning pipeline** (per-request):

1. **Classify** – regex patterns categorise question as: `skill_depth`, `architecture`, `project`, `comparison`, `credential`, or `general`
2. **Route** – `RAGRouter.select_strategy()` reads `EFFECTIVE_FOR` edge weights from Neptune and picks the highest-weighted `RAGStrategy` for the question type; falls back to hard-coded priors when Neptune has no feedback data yet
3. **Retrieve** – `retrieve_with_strategy()` dispatches to the chosen strategy:
   - `semantic_search`: Bedrock KB vector retrieval
   - `graph_first`: Bedrock KB + forced graph expansion
   - `keyword_boosted`: Bedrock KB + keyword-overlap reranking (Jaccard, stopword-filtered, 25/75 blend)
   - `hybrid`: Bedrock KB + Neptune vector search over `ChunkVector` nodes + keyword boost + result merging
4. **Deduplicate** – Jaccard similarity prefix dedup removes near-duplicate chunks
5. **Graph expand** – Neptune Analytics openCypher queries surface skill neighbourhood, pattern evidence, AWS service context, person summary. Forced for `graph_first` and `hybrid` strategies regardless of the `includeGraphExpansion` flag.
6. **Synthesize** – Bedrock Converse API with grounded system prompt; model response is JSON-structured
7. **Feedback** – `RAGRouter.update_feedback()` adjusts the `EFFECTIVE_FOR` edge weight in Neptune based on synthesis confidence
8. **Return** – `{ answer, sources, inferredSkills, repeatedPatterns, confidence, questionType, graphEntitiesUsed, routingDecision }`

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

### RAG Router (`rag_router.py`) — new

The routing agent lives in `src/query_api/rag_router.py`:

- `select_strategy(question_type, graph_id)` — queries Neptune `EFFECTIVE_FOR` edges, picks the strategy with the highest weight; returns a `RetrievalConfig`
- `update_feedback(strategy, question_type, confidence, graph_id)` — mutates the `EFFECTIVE_FOR` edge weight after each query
- `seed_routing_graph(graph_id)` — idempotent upsert of initial `RAGStrategy` + `DocumentType` nodes and `EFFECTIVE_FOR` edges; called from `IngestionTriggerFunction` on CloudFormation Create/Update

### Ingestion Trigger Lambda

On **CloudFormation Create/Update**, this Lambda now:
1. Seeds the routing graph (`seed_routing_graph()`)
2. Starts the Bedrock KB ingestion job
3. Polls until complete

On **direct invocation** with `action=seed_routing`, it re-seeds the routing
graph without triggering ingestion (useful for resetting after graph migration).

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
| `repo-signals.yaml` | 0.7 | Structured expert signals |
| PlantUML-derived summaries | 0.8 | Architecture diagrams → component relationships |
| `code`, `README.md` | 0.6 | Implementation evidence |
| Articles / blog posts | 0.5 | Published thinking |
| Resume | 0.3 | Self-reported, lower authority |

**Effective score** = raw retrieval score × source weight

**Repeated patterns** (implemented in multiple repos) are weighted higher than
one-off mentions.

---

## IAM Security Model

All roles follow least-privilege:

| Role | Permissions |
|------|-------------|
| `PreprocessorRole` | S3 GetObject (raw/), S3 PutObject (derived/), KMS |
| `QueryAPIRole` | Bedrock Retrieve, Bedrock InvokeModel, Neptune ReadDataViaQuery + ExecuteQuery (routing graph), S3 GetObject (derived/) |
| `IngestionTriggerRole` | Bedrock StartIngestionJob, GetIngestionJob, Neptune ExecuteQuery (seed routing graph), S3 ListBucket |
| `BedrockKnowledgeBaseRole` | S3 ListBucket+GetObject (all), Neptune Write+Read, Bedrock InvokeModel (Titan) |
| `StepFunctionsRole` | Lambda InvokeFunction (Preprocessor + Trigger only), CloudWatch Logs, X-Ray |

---

## Deployment Flow

1. `sam deploy` creates all resources
2. CloudFormation custom resource (`PostDeploymentIngestionJob`) seeds the routing graph, then triggers an initial ingestion job
3. Upload repo files to `s3://<bucket>/raw/<repo>/`
4. S3 event triggers `PreprocessorFunction`
5. Preprocessor classifies each document, writes derived artifacts and routing metadata to Neptune
6. Trigger a new ingestion job (via `scripts/start_ingestion.py` or Step Functions)
7. Query via `POST /query-expertise` — routing agent selects strategy, answers, updates Neptune weights


## System Purpose

ExpertiseRAG is an AWS-native Retrieval-Augmented Generation (RAG) platform
designed to answer deep, evidence-backed questions about a developer's
professional expertise. It goes beyond simple keyword search by combining:

1. **Semantic retrieval** – Amazon Bedrock Knowledge Bases with Titan Text
   Embeddings V2 (1024-dim)
2. **Graph traversal** – Neptune Analytics GraphRAG to surface co-occurring
   skills, repeated patterns, and cross-repository evidence
3. **Source weighting** – architecture docs and CLAUDE.md treated as
   authoritative; resume as supporting evidence only
4. **Evidence synthesis** – Bedrock foundation model generates grounded,
   citation-backed answers

---

## High-Level Architecture

```
                        ┌─────────────────────────────┐
  Developer uploads     │          S3 Bucket           │
  raw repo files ──────▶│  raw/<repo>/architecture.md  │
  (architecture.md,     │  raw/<repo>/CLAUDE.md        │
   CLAUDE.md,           │  raw/<repo>/*.puml            │
   *.puml, *.yaml, …)   │  raw/<repo>/repo-signals.yaml│
                        └──────────────┬──────────────┘
                                       │ S3 ObjectCreated
                                       ▼
                        ┌─────────────────────────────┐
                        │   Preprocessor Lambda        │
                        │   (Python 3.12)              │
                        │  • extract_markdown()        │
                        │  • extract_yaml()            │
                        │  • extract_plantuml()        │
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
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┼───────────────────┐
                    │                  │                   │
                    ▼                  ▼                   │
     ┌──────────────────────┐  ┌──────────────────────┐   │
     │  Step Functions       │  │  Bedrock KB Ingestion│   │
     │  (Orchestration)      │  │  Custom Resource      │   │
     │  • Preprocess         │  │  (post-deploy)        │   │
     │  • Start ingestion    │  └────────────┬─────────┘   │
     │  • Poll until done    │               │             │
     └──────────────────────┘               │             │
                                            ▼             │
                        ┌─────────────────────────────┐   │
                        │  Bedrock Knowledge Base       │   │
                        │  • Titan Text Embeddings V2  │   │
                        │  • Hierarchical chunking      │   │
                        │  • Claude Haiku for parsing   │   │
                        └──────────────┬──────────────┘   │
                                       │ stores vectors    │
                                       ▼                   │
                        ┌─────────────────────────────┐   │
                        │   Neptune Analytics Graph    │◀──┘
                        │   • Vector search (1024-dim) │
                        │   • openCypher queries       │
                        │   • GraphRAG traversal       │
                        └─────────────────────────────┘
                                       ▲
                    Query time:        │
                                       │
     ┌─────────────┐    ┌─────────────┴────────────┐
     │  API Gateway │    │   Query API Lambda        │
     │  HTTP API v2 │───▶│  POST /query-expertise   │
     └─────────────┘    │  1. classify_question()   │
                        │  2. retrieve_chunks()      │
                        │  3. expand_graph_context() │
                        │  4. synthesize_answer()    │
                        └──────────────────────────┘
```

---

## Component Details

### S3 Bucket

- **Prefix layout**:
  - `raw/<repo-name>/` – original uploads (triggers preprocessor)
  - `derived/<repo-name>/` – preprocessor output (fed to Bedrock KB)
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

**Graph Builder** (`GraphBuilder`):
- Seeds `Person` and `Repository` nodes
- For each file: creates `Document` node, `CONTAINS` edge
- For each signal: creates typed nodes (Skill, Technology, AWSService, Pattern)
- Cross-entity `STRENGTHENS` edges connect related skills/patterns

**Derived artifacts** written per file and per repo:
```
derived/<repo>/
  <file>.derived.json       # full extraction envelope
  <file>.extracted.txt      # clean text for Bedrock ingestion
  graph_entities.json       # all nodes
  graph_edges.json          # all edges
  expertise_signals.json    # aggregated + deduplicated signals
  processing_manifest.json  # statistics and error log
```

### Neptune Analytics Graph

- **Type**: `AWS::NeptuneGraph::Graph`
- **Vector dimension**: 1024 (matches Titan Text Embeddings V2)
- **Query language**: openCypher
- **Use at query time**: graph expansion, skill neighbourhood, pattern evidence

**Node types**:

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

**Edge types**:

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

### Bedrock Knowledge Base

- **Storage**: Neptune Analytics (vector + graph hybrid)
- **Embedding**: Amazon Titan Text Embeddings V2, 1024 dimensions
- **Chunking**: Hierarchical (1500 token parents, 300 token children, 60 token overlap)
- **Parsing**: Claude 3 Haiku with domain-specific prompt extracting technical signals
- **Search**: HYBRID (semantic + keyword)

### Bedrock Data Source

- Points to both `raw/` and `derived/` S3 prefixes
- The `derived/*.extracted.txt` files are the primary ingestion target
- The `derived/*.derived.json` files provide structured metadata

### Query API Lambda

**Reasoning pipeline** (per-request):

1. **Classify** – regex patterns categorise question as: `skill_depth`, `architecture`, `project`, `comparison`, `credential`, or `general`
2. **Retrieve** – `bedrock-agent-runtime.retrieve()` with HYBRID search, top-K chunks
3. **Deduplicate** – Jaccard similarity prefix dedup to remove near-duplicate chunks
4. **Graph expand** – Neptune Analytics openCypher queries surface:
   - Skill neighbourhood (co-occurring skills, technologies)
   - Pattern evidence (which repos demonstrate each pattern)
   - AWS service context (co-services and co-technologies)
   - Person summary (top skills, patterns, AWS services)
5. **Synthesize** – Bedrock Converse API with grounded system prompt; model response is JSON-structured
6. **Return** – `{ answer, sources, inferredSkills, repeatedPatterns, confidence, questionType, graphEntitiesUsed }`

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
| `repo-signals.yaml` | 0.7 | Structured expert signals |
| PlantUML-derived summaries | 0.8 | Architecture diagrams → component relationships |
| `code`, `README.md` | 0.6 | Implementation evidence |
| Articles / blog posts | 0.5 | Published thinking |
| Resume | 0.3 | Self-reported, lower authority |

**Effective score** = raw retrieval score × source weight

**Repeated patterns** (implemented in multiple repos) are weighted higher than
one-off mentions.

---

## IAM Security Model

All roles follow least-privilege:

| Role | Permissions |
|------|-------------|
| `PreprocessorRole` | S3 GetObject (raw/), S3 PutObject (derived/), KMS |
| `QueryAPIRole` | Bedrock Retrieve, Bedrock InvokeModel, Neptune ReadDataViaQuery, S3 GetObject (derived/) |
| `IngestionTriggerRole` | Bedrock StartIngestionJob, GetIngestionJob, S3 ListBucket |
| `BedrockKnowledgeBaseRole` | S3 ListBucket+GetObject (all), Neptune Write+Read, Bedrock InvokeModel (Titan) |
| `StepFunctionsRole` | Lambda InvokeFunction (Preprocessor + Trigger only), CloudWatch Logs, X-Ray |

---

## Deployment Flow

1. `sam deploy` creates all resources
2. CloudFormation custom resource (`PostDeploymentIngestionJob`) triggers an initial ingestion job
3. Upload repo files to `s3://<bucket>/raw/<repo>/`
4. S3 event triggers `PreprocessorFunction`
5. Preprocessor writes derived artifacts to `s3://<bucket>/derived/<repo>/`
6. Trigger a new ingestion job (via `scripts/start_ingestion.py` or Step Functions)
7. Query via `POST /query-expertise`
