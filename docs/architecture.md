# ExpertiseRAG – Architecture Overview

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
