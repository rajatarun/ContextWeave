# Ingestion & Retrieval Walkthrough

This document traces a single file — `architecture.md` — from the moment it is
uploaded to S3 all the way through to a grounded answer returned by the Query
API. Every step is concrete: the same file, the same signals, the same nodes.

---

## Sample File

The walkthrough uses the following snippet as its **input document**.

**File**: `architecture.md`
**S3 key**: `raw/contextweave/architecture.md`
**Evidence weight**: `1.0` (authoritative architecture document)

```markdown
# ExpertiseRAG – Architecture Overview

ExpertiseRAG is a serverless, event-driven GraphRAG platform built on AWS.
It combines Amazon Bedrock Knowledge Bases with Neptune Analytics to answer
deep, evidence-backed questions about a developer's expertise.

## Core Components

| Service            | Role                                      |
|--------------------|-------------------------------------------|
| AWS Lambda         | Preprocessing and query orchestration     |
| Amazon S3          | Raw and derived artifact storage          |
| Amazon Bedrock     | Embeddings (Titan V2) + LLM synthesis     |
| Neptune Analytics  | Vector + graph hybrid store               |
| AWS Step Functions | Ingestion workflow orchestration          |
| AWS KMS            | SSE-KMS encryption for all S3 objects     |

## Design Decisions

### 1. GraphRAG over plain vector search
Neptune Analytics supports vector search and graph traversal in a single query.
After semantic retrieval, the system walks graph edges to surface co-occurring
skills, repeated patterns, and AWS service co-usage across repositories.

### 2. Hierarchical chunking (1500 / 300 tokens)
Parent chunks preserve full section context; child chunks are the retrieval
units for precision. 60-token overlap ensures continuity at boundaries.

### 3. Evidence weighting
Source credibility is encoded at retrieval time. architecture.md and CLAUDE.md
carry weight 1.0; resume carries weight 0.3.
```

---

## Part 1 – Ingestion

Ingestion transforms the raw file into four things:

1. **Extracted text** — clean prose ingested into Bedrock Knowledge Base
2. **Graph entities and edges** — nodes and relationships written to Neptune Analytics
3. **Routing metadata** — DocumentType and ChunkingStrategy nodes written to Neptune
4. **Expertise signals** — structured skill/service/pattern observations

### Step 1 — Upload to S3

The developer (or GitHub Actions) copies the file to the `raw/` prefix:

```bash
aws s3 cp architecture.md \
  s3://expertise-rag-artifacts-239571291755-dev/raw/contextweave/architecture.md
```

S3 emits an `ObjectCreated` event on the `raw/` prefix.

```
S3 Bucket
└── raw/
    └── contextweave/
        └── architecture.md   ← new object triggers event
```

---

### Step 2 — Preprocessor Lambda is triggered

The S3 notification invokes `expertise-rag-preprocessor-dev`.

```
S3 ObjectCreated
    └──► PreprocessorFunction (Python 3.12, 1024 MB, 600s timeout)
             src/preprocessor/handler.py
```

The handler reads the file from S3, determines the file type (`.md`), and
routes it to `extract_markdown()` in `src/preprocessor/extractors.py`.

---

### Step 3 — Markdown extraction (`extract_markdown`)

`extract_markdown()` runs three passes over the content.

#### Pass A — Clean text

Headings, table rows, and paragraphs are extracted into a flat prose block
suitable for Bedrock ingestion.

```
Extracted text (written to derived/contextweave/architecture.extracted.txt):

"ExpertiseRAG Architecture Overview
ExpertiseRAG is a serverless, event-driven GraphRAG platform built on AWS.
It combines Amazon Bedrock Knowledge Bases with Neptune Analytics to answer
deep, evidence-backed questions about a developer's expertise.

Core Components
AWS Lambda — Preprocessing and query orchestration
Amazon S3 — Raw and derived artifact storage
Amazon Bedrock — Embeddings (Titan V2) + LLM synthesis
Neptune Analytics — Vector + graph hybrid store
AWS Step Functions — Ingestion workflow orchestration
AWS KMS — SSE-KMS encryption for all S3 objects
..."
```

#### Pass B — AWS service signals

The extractor runs `_AWS_SERVICE_PATTERNS` regex over every line. Each match
becomes an `ExpertiseSignal` with category `aws_service`.

| Matched text        | Signal value        | Source file        | Weight |
|---------------------|---------------------|--------------------|--------|
| `AWS Lambda`        | `AWS Lambda`        | architecture.md    | 1.0    |
| `Amazon S3`         | `Amazon S3`         | architecture.md    | 1.0    |
| `Amazon Bedrock`    | `Amazon Bedrock`    | architecture.md    | 1.0    |
| `Neptune Analytics` | `Neptune Analytics` | architecture.md    | 1.0    |
| `AWS Step Functions`| `AWS Step Functions`| architecture.md    | 1.0    |
| `AWS KMS`           | `AWS KMS`           | architecture.md    | 1.0    |

#### Pass C — Architecture pattern signals

`_PATTERN_KEYWORDS` regex matches design pattern terms.

| Matched text      | Signal value    | Category  |
|-------------------|-----------------|-----------|
| `serverless`      | `serverless`    | pattern   |
| `event-driven`    | `event-driven`  | pattern   |
| `GraphRAG`        | `GraphRAG`      | pattern   |

---

### Step 3.5 — Routing analyzer classifies the document ★ NEW

**Before** the graph is built, `routing_analyzer.analyze_document()` runs
text-structural analysis on the extracted content to classify the document
type and recommend the best chunking strategy.

```python
# src/preprocessor/handler.py (called after extract_markdown)
analysis = analyze_document(
    text=extracted_text,
    filename="architecture.md",
    signals=expertise_signals
)
```

#### Text statistics computed

| Statistic          | Value   | How computed                                        |
|--------------------|---------|-----------------------------------------------------|
| `heading_count`    | 6       | Lines starting with `#`                             |
| `code_char_ratio`  | 0.04    | Characters inside code fences ÷ total characters    |
| `avg_sentence_len` | 12.3    | Words ÷ sentence count                              |
| `has_yaml_keys`    | false   | No `key: value` density threshold reached           |
| `has_plantuml`     | false   | No `@startuml` marker                               |
| `aws_service_count`| 6       | AWS service signals already extracted               |

#### Classification decision

```
Priority order: diagram_derived → code → structured_data → technical_spec → narrative

has_plantuml = false  → not diagram_derived
code_char_ratio < 0.35 → not code
has_yaml_keys = false → not structured_data
heading_count (6) ≥ 3 AND aws_service_count (6) ≥ 2 → ✓ technical_spec
```

**Result**: `doc_type = "technical_spec"`

#### Chunking strategy recommendation

| DocumentType     | ChunkingStrategy | Rationale                         |
|------------------|------------------|-----------------------------------|
| `technical_spec` | `hierarchical`   | Section headings define context   |

**Result**: `chunking_strategy = "hierarchical"` (1500/300 tokens)

The `DocumentTypeAnalysis` is returned:

```python
DocumentTypeAnalysis(
    doc_type="technical_spec",
    chunking_strategy="hierarchical",
    text_stats={
        "heading_count": 6,
        "code_char_ratio": 0.04,
        "avg_sentence_len": 12.3,
        "has_yaml_keys": False,
        "has_plantuml": False
    },
    confidence=0.90
)
```

---

### Step 4 — Graph Builder constructs nodes and edges

`GraphBuilder` in `src/preprocessor/graph_builder.py` converts the signals into
graph primitives.  Node IDs are deterministic slugs, so the same entity seen in
multiple files always merges into one node.

#### Expertise nodes created

```json
[
  { "node_id": "person_rajat_arun",       "node_type": "Person",      "label": "Rajat Arun" },
  { "node_id": "repository_contextweave", "node_type": "Repository",  "label": "contextweave" },
  { "node_id": "document_architecture_md","node_type": "Document",    "label": "architecture.md",
    "properties": { "file_type": "markdown", "weight": 1.0, "doc_type": "technical_spec" } },

  { "node_id": "awsservice_aws_lambda",        "node_type": "AWSService", "label": "AWS Lambda" },
  { "node_id": "awsservice_amazon_s3",         "node_type": "AWSService", "label": "Amazon S3" },
  { "node_id": "awsservice_amazon_bedrock",    "node_type": "AWSService", "label": "Amazon Bedrock" },
  { "node_id": "awsservice_neptune_analytics", "node_type": "AWSService", "label": "Neptune Analytics" },
  { "node_id": "awsservice_aws_step_functions","node_type": "AWSService", "label": "AWS Step Functions" },
  { "node_id": "awsservice_aws_kms",           "node_type": "AWSService", "label": "AWS KMS" },

  { "node_id": "pattern_serverless",   "node_type": "Pattern", "label": "serverless" },
  { "node_id": "pattern_event_driven", "node_type": "Pattern", "label": "event-driven" },
  { "node_id": "pattern_graphrag",     "node_type": "Pattern", "label": "GraphRAG" }
]
```

#### Routing nodes created (via `add_routing_metadata`) ★ NEW

```json
[
  {
    "node_id": "documenttype_technical_spec",
    "node_type": "DocumentType",
    "properties": {
      "label": "technical_spec",
      "question_type": "architecture",
      "avg_heading_count": 6,
      "avg_code_ratio": 0.04,
      "avg_sentence_len": 12.3
    }
  },
  {
    "node_id": "chunkingstrategy_hierarchical",
    "node_type": "ChunkingStrategy",
    "properties": { "label": "hierarchical" }
  }
]
```

#### Expertise edges created

```json
[
  { "from": "person_rajat_arun",       "rel": "BUILT",               "to": "repository_contextweave" },
  { "from": "repository_contextweave", "rel": "CONTAINS",            "to": "document_architecture_md" },
  { "from": "document_architecture_md","rel": "USES_AWS_SERVICE",    "to": "awsservice_aws_lambda" },
  { "from": "document_architecture_md","rel": "USES_AWS_SERVICE",    "to": "awsservice_amazon_s3" },
  { "from": "document_architecture_md","rel": "USES_AWS_SERVICE",    "to": "awsservice_amazon_bedrock" },
  { "from": "document_architecture_md","rel": "USES_AWS_SERVICE",    "to": "awsservice_neptune_analytics" },
  { "from": "document_architecture_md","rel": "USES_AWS_SERVICE",    "to": "awsservice_aws_step_functions" },
  { "from": "document_architecture_md","rel": "USES_AWS_SERVICE",    "to": "awsservice_aws_kms" },
  { "from": "document_architecture_md","rel": "DEMONSTRATES_PATTERN","to": "pattern_serverless" },
  { "from": "document_architecture_md","rel": "DEMONSTRATES_PATTERN","to": "pattern_event_driven" },
  { "from": "document_architecture_md","rel": "DEMONSTRATES_PATTERN","to": "pattern_graphrag" }
]
```

#### Routing edges created ★ NEW

```json
[
  {
    "from": "document_architecture_md",
    "rel":  "HAS_TYPE",
    "to":   "documenttype_technical_spec",
    "weight": 1.0
  },
  {
    "from": "document_architecture_md",
    "rel":  "CHUNKED_WITH",
    "to":   "chunkingstrategy_hierarchical",
    "weight": 1.0
  }
]
```

The `EFFECTIVE_FOR` edges (`RAGStrategy → DocumentType`) were already seeded at
deploy time by `seed_routing_graph()` and are not recreated per document.

---

### Step 5 — Derived artifacts written to S3

The preprocessor writes all outputs under the `derived/` prefix.

```
S3 Bucket
└── derived/
    └── contextweave/
        ├── architecture.extracted.txt    ← clean text for Bedrock KB
        ├── architecture.derived.json     ← full extraction envelope
        ├── graph_entities.json           ← all nodes (merged across files)
        ├── graph_edges.json              ← all edges (incl. routing edges)
        ├── expertise_signals.json        ← deduplicated signals
        └── processing_manifest.json      ← stats, routing_summary, error log
```

The `processing_manifest.json` now includes a `routing_summary` section:

```json
{
  "routing_summary": {
    "technical_spec:hierarchical": 1
  }
}
```

---

### Step 6 — Step Functions orchestrates Bedrock ingestion

The `expertise-rag-ingestion-dev` state machine runs the following states:

```
PreprocessRawFiles
    └──► StartIngestionJob         (calls bedrock-agent:StartIngestionJob)
             └──► WaitForIngestion (waits 30 seconds)
                      └──► CheckIngestionStatus
                                └── COMPLETE → IngestionSuccess
                                └── IN_PROGRESS → back to WaitForIngestion
                                └── FAILED → IngestionFailed
```

---

### Step 7 — Bedrock Knowledge Base ingests extracted text

During `StartIngestionJob`, Bedrock reads `derived/contextweave/architecture.extracted.txt`
and applies **hierarchical chunking** (recommended by the routing analyzer):

```
Parent chunk (≤ 1500 tokens) — preserves full section context
│
├── Child chunk 1 (≤ 300 tokens, 60-token overlap with next)
│     "ExpertiseRAG is a serverless, event-driven GraphRAG platform..."
│
├── Child chunk 2 (≤ 300 tokens)
│     "Core Components: AWS Lambda — Preprocessing and query orchestration
│      Amazon S3 — Raw and derived artifact storage..."
│
└── Child chunk 3 (≤ 300 tokens)
      "Design Decisions: GraphRAG over plain vector search.
       Neptune Analytics supports vector search and graph traversal..."
```

Claude 3 Haiku parses each chunk during ingestion to extract additional
technical signals (component relationships, AWS service references, patterns).

Each child chunk is then embedded by **Titan Text Embeddings V2** into a
1024-dimensional vector and stored in Neptune Analytics alongside the graph
nodes and edges from Step 4.

---

### Ingestion Complete — What now exists in Neptune Analytics

```
Neptune Analytics Graph
│
├── (person_rajat_arun) ──BUILT──► (repository_contextweave)
│
├── (repository_contextweave) ──CONTAINS──► (document_architecture_md)
│                                              │
│                              ┌──────────────┼──────────────────────────────────┐
│                              ▼              ▼                                  ▼
│                  USES_AWS_SERVICE    DEMONSTRATES_PATTERN        (vector embeddings
│                  ──► Amazon Bedrock  ──► serverless               for each child chunk
│                  ──► AWS Lambda      ──► event-driven             stored alongside
│                  ──► Neptune Analytics──► GraphRAG                the graph nodes)
│                  ──► Amazon S3
│                  ──► AWS Step Functions
│                  ──► AWS KMS
│
├── Routing sub-graph (seeded at deploy + updated per document): ★ NEW
│   ├── (document_architecture_md) ──HAS_TYPE──► (documenttype_technical_spec)
│   ├── (document_architecture_md) ──CHUNKED_WITH──► (chunkingstrategy_hierarchical)
│   └── EFFECTIVE_FOR edges (RAGStrategy ──► DocumentType, weights from ROUTING_PRIORS):
│       ├── (ragstrategy_graph_first) ──[weight:0.80]──► (documenttype_technical_spec)
│       ├── (ragstrategy_hybrid)      ──[weight:0.70]──► (documenttype_technical_spec)
│       └── ...
```

---

## Part 2 – Retrieval

A recruiter sends the following question to the API:

```bash
POST /query-expertise
{
  "question": "What AWS services has this developer built production systems with?",
  "topK": 10,
  "includeGraphExpansion": true
}
```

### Step 1 — Question classification (`classify_question`)

`src/query_api/synthesizer.py` applies regex patterns to categorise the question.

| Pattern matched                                   | Question type  |
|---------------------------------------------------|----------------|
| `"AWS services"` + `"built"` + `"production"` | `skill_depth`  |

Question type `skill_depth` tells the system to focus on concrete,
evidence-backed service usage rather than generic overviews.

---

### Step 1.5 — RAGRouter selects retrieval strategy ★ NEW

`src/query_api/rag_router.py` reads `EFFECTIVE_FOR` edge weights from Neptune
to pick the optimal retrieval strategy for `skill_depth` questions.

#### Neptune query executed

```cypher
MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
WHERE d.question_type = 'skill_depth'
RETURN r.label AS strategy, e.weight AS weight,
       coalesce(e.feedback_count, 0) AS feedback_count
ORDER BY e.weight DESC
```

#### Routing decision

| Strategy         | Weight | Feedback count |
|------------------|--------|----------------|
| `graph_first`    | 0.80   | 0 (prior)      |
| `semantic_search`| 0.60   | 0 (prior)      |

**Winner**: `graph_first` (highest weight = 0.80)

```python
RetrievalConfig(
    strategy="graph_first",
    include_graph=True,       # always forced for graph_first
    boost_keywords=False,
    use_neptune_chunks=False,
    strategy_confidence=0.80
)
```

The `include_graph=True` flag will force graph expansion in Step 4 regardless
of the `includeGraphExpansion` request parameter.

---

### Step 2 — Retrieve from Bedrock Knowledge Base (`retrieve_with_strategy`)

`src/query_api/retriever.py` calls `bedrock-agent-runtime.retrieve()` with
`HYBRID` search mode (semantic + keyword):

```python
bedrock_agent_runtime.retrieve(
    knowledgeBaseId="<kb-id>",
    retrievalQuery={"text": "AWS services developer built production systems"},
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 10,
            "overrideSearchType": "HYBRID"
        }
    }
)
```

Bedrock embeds the question (Titan V2 → 1024-dim vector) and finds the closest
child chunks. The **three child chunks** from `architecture.md` all rank highly
because their embeddings are close to the question embedding.

Raw retrieval result (before weighting):

| Chunk                          | Raw score | Source file      | Weight |
|--------------------------------|-----------|------------------|--------|
| "Core Components: AWS Lambda…" | 0.91      | architecture.md  | 1.0    |
| "serverless, event-driven…"    | 0.87      | architecture.md  | 1.0    |
| "GraphRAG over plain vector…"  | 0.79      | architecture.md  | 1.0    |

**Effective score** = raw score × source weight.
Because `architecture.md` has weight `1.0`, the effective scores are unchanged.
A resume chunk scoring `0.95` would be downweighted to `0.95 × 0.3 = 0.285`.

---

### Step 3 — Deduplication (`deduplicate_chunks`)

Jaccard similarity is computed between each pair of chunks. Near-duplicates
(similarity > threshold) are dropped, keeping only the highest-scoring copy.
In this case all three chunks are distinct and all are retained.

---

### Step 4 — Graph expansion (`expand_graph_context`)

Because `retrieval_config.include_graph = True` (forced by `graph_first`
strategy), `src/query_api/graph_expander.py` runs openCypher queries against
Neptune Analytics, seeded by the text from the retrieved chunks.

#### Query A — AWS service context

```cypher
MATCH (doc:Document)-[:USES_AWS_SERVICE]->(svc:AWSService)
WHERE doc.properties.weight >= 0.7
WITH svc, count(DISTINCT doc) AS doc_count
RETURN svc.label AS service, svc.properties.frequency AS frequency, doc_count
ORDER BY frequency DESC
LIMIT 20
```

Returns: `Amazon Bedrock (22)`, `AWS Lambda (18)`, `Neptune Analytics (15)`,
`Amazon S3 (14)`, `AWS Step Functions (10)`, `AWS KMS (8)` …

#### Query B — Repeated patterns

```cypher
MATCH (p:Person)-[:BUILT]->(repo:Repository)-[:DEMONSTRATES_PATTERN]->(pat:Pattern)
WITH pat, count(DISTINCT repo) AS repo_count
WHERE repo_count >= 2
RETURN pat.label AS pattern, repo_count
ORDER BY repo_count DESC
```

Returns: `serverless (3 repos)`, `event-driven (3 repos)`, `GraphRAG (2 repos)`

#### Query C — Skill neighbourhood

Extracts technology terms from the retrieved chunk text, then finds co-occurring
skills and services in the graph to broaden the answer's evidence base.

Returns: `Python`, `AWS SAM`, `GitHub Actions OIDC`, `infrastructure-as-code`

---

### Graph context assembled

```json
{
  "inferred_skills": [
    "Amazon Bedrock", "AWS Lambda", "Neptune Analytics",
    "Amazon S3", "AWS Step Functions", "AWS KMS"
  ],
  "repeated_patterns": ["serverless", "event-driven", "GraphRAG"],
  "aws_context": [
    { "service": "Amazon Bedrock",    "frequency": 22 },
    { "service": "AWS Lambda",        "frequency": 18 },
    { "service": "Neptune Analytics", "frequency": 15 }
  ],
  "skill_neighbourhood": ["Python", "AWS SAM", "GitHub Actions OIDC"],
  "graph_entities_used": [
    "document_architecture_md",
    "awsservice_amazon_bedrock",
    "awsservice_aws_lambda",
    "pattern_serverless"
  ]
}
```

---

### Step 5 — Answer synthesis (`synthesize_answer`)

`src/query_api/synthesizer.py` calls **Bedrock Converse API** with a grounded
system prompt that injects the retrieved chunks and graph context:

```
System prompt (abbreviated):
  You are an expert technical interviewer...
  Answer ONLY using the provided source chunks.
  Cite sources by file name. Do not hallucinate.

  --- SOURCE CHUNKS ---
  [architecture.md, score 0.91]
  "Core Components: AWS Lambda — Preprocessing and query orchestration;
   Amazon S3 — Raw and derived artifact storage; ..."

  [architecture.md, score 0.87]
  "ExpertiseRAG is a serverless, event-driven GraphRAG platform..."

  --- GRAPH CONTEXT ---
  Inferred skills: Amazon Bedrock, AWS Lambda, Neptune Analytics, ...
  Repeated patterns: serverless (3 repos), event-driven (3 repos)
```

The model produces a structured JSON response with `confidence = 0.94`.

---

### Step 6 — Feedback written back to Neptune ★ NEW

After synthesis, `rag_router.update_feedback()` is called:

```python
update_feedback(
    strategy="graph_first",
    question_type="skill_depth",
    confidence=0.94,        # ≥ 0.70 → reinforce
    graph_id=NEPTUNE_GRAPH_ID
)
```

Because `confidence (0.94) ≥ 0.70`, the `EFFECTIVE_FOR` edge weight is
incremented:

```cypher
MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
WHERE r.label = 'graph_first' AND d.question_type = 'skill_depth'
SET e.weight = min(1.0, e.weight + 0.05),
    e.feedback_count = coalesce(e.feedback_count, 0) + 1
```

**Before**: `graph_first → skill_depth` weight = 0.80, feedback_count = 0
**After**:  `graph_first → skill_depth` weight = 0.85, feedback_count = 1

The next `skill_depth` question will see `graph_first` at weight 0.85, making
it even more likely to be selected. Over dozens of queries, weights converge
to reflect which strategies actually produce high-confidence answers.

---

### Step 7 — API response returned

```json
{
  "answer": "Based on architecture.md (authoritative source, weight 1.0), this developer
             has built production systems with Amazon Bedrock (Knowledge Bases + Titan
             Embeddings V2 + Converse API), AWS Lambda (Python 3.12, three distinct
             functions), Amazon S3 (SSE-KMS, versioning, lifecycle policies), Neptune
             Analytics (vector + graph hybrid store, openCypher), AWS Step Functions
             (STANDARD type, X-Ray tracing), and AWS KMS (customer-managed key with
             annual rotation). The serverless and event-driven patterns appear across
             three separate repositories, indicating repeated, deliberate application
             rather than one-off usage.",

  "sources": [
    { "file": "architecture.md", "weight": 1.0, "score": 0.91 },
    { "file": "architecture.md", "weight": 1.0, "score": 0.87 },
    { "file": "architecture.md", "weight": 1.0, "score": 0.79 }
  ],

  "inferredSkills": [
    "Amazon Bedrock", "AWS Lambda", "Neptune Analytics",
    "Amazon S3", "AWS Step Functions", "AWS KMS"
  ],

  "repeatedPatterns": ["serverless", "event-driven", "GraphRAG"],

  "confidence": 0.94,
  "questionType": "skill_depth",

  "graphEntitiesUsed": [
    "document_architecture_md",
    "awsservice_amazon_bedrock",
    "awsservice_aws_lambda",
    "pattern_serverless"
  ],

  "retrievalCount": 3,
  "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "latencyMs": 2340,

  "routingDecision": {
    "strategy": "graph_first",
    "questionType": "skill_depth",
    "strategyConfidence": 0.80,
    "graphExpansionForced": true,
    "keywordBoostApplied": false,
    "neptuneVectorsUsed": false
  }
}
```

---

## End-to-End Summary

```
Upload                  Preprocess                    Ingest                  Query
──────                  ──────────                    ──────                  ─────
aws s3 cp               extract_markdown()            Bedrock KB              classify_question()
  architecture.md   ──► _AWS_SERVICE_PATTERNS     ──► Titan V2 embed      ──► RAGRouter             ★ NEW
  → raw/contextweave    _PATTERN_KEYWORDS              1024-dim vectors        select_strategy()
                        routing_analyzer()  ★ NEW      hierarchical chunks     → graph_first
                          classify doc_type          ──► Neptune Analytics  ──► retrieve_with_strategy()
                          → technical_spec               graph + vectors        HYBRID search
                          → hierarchical             ──► seed EFFECTIVE_FOR  ──► deduplicate_chunks()
                        GraphBuilder()                   edges at deploy     ──► expand_graph_context()
                          expertise nodes/edges                                  openCypher queries
                          DocumentType node  ★ NEW                          ──► synthesize_answer()
                          HAS_TYPE edge      ★ NEW                               Bedrock Converse
                          CHUNKED_WITH edge  ★ NEW                          ──► update_feedback()    ★ NEW
                        → derived/contextweave                                   confidence → Neptune
                          .extracted.txt                                     ──► JSON response
                          graph_entities.json                                     answer + sources
                          graph_edges.json                                        inferredSkills
                          expertise_signals.json                                  repeatedPatterns
                          processing_manifest.json                               confidence
                                                                                 routingDecision ★ NEW
```

### Why GraphRAG is better than plain vector search

A plain vector store would retrieve the closest chunks and stop there. GraphRAG
goes further: after retrieval it walks the graph to find that `serverless` and
`event-driven` appear in **three repositories**, not just one. That cross-repo
frequency is impossible to express in a flat vector index, but it is exactly
what distinguishes a developer who has applied a pattern repeatedly from one who
mentioned it once.

### Why adaptive routing improves answer quality

Without routing, every question uses the same strategy. With routing:
- `skill_depth` questions get `graph_first` (high Neptune traversal depth)
- `comparison` questions get `hybrid` (Neptune vectors + keyword boost)
- `credential` questions get `keyword_boosted` (exact-match term reranking)

After 30+ queries, the `EFFECTIVE_FOR` weights shift to reflect which strategies
actually produce high-confidence answers for each question type. The system
self-improves without any retraining or manual tuning.

### Why source weighting matters

If a resume chunk scores `0.95` on retrieval but `architecture.md` only scores
`0.79`, naive top-K would surface the resume chunk first. Source weighting
corrects this: `0.95 × 0.3 = 0.285` vs `0.79 × 1.0 = 0.79`. The authoritative
architecture document always wins.
