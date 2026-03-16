# ExpertiseRAG Graph Schema

## Overview

The ExpertiseRAG graph is stored in Neptune Analytics and uses openCypher as
the query language. It contains two distinct sub-graphs:

1. **Expertise graph** — nodes and edges written by the Preprocessor Lambda
   representing skills, patterns, AWS services, and documents
2. **Routing graph** — nodes and edges representing RAG strategy effectiveness,
   document types, and chunking strategies; updated by every query via the
   feedback loop

Both sub-graphs are written to and read from the same Neptune Analytics graph.
Expertise nodes are exported to `derived/<repo>/graph_entities.json` and
`derived/<repo>/graph_edges.json`. Routing nodes are seeded at deploy time and
updated at runtime.

---

## Part 1 — Expertise Graph

### Node Types

#### Person
The central node representing the developer whose expertise is being captured.

```json
{
  "node_id": "person_rajat_arun",
  "node_type": "Person",
  "label": "Rajat Arun",
  "properties": { "name": "Rajat Arun" }
}
```

#### Repository
A git repository or project ingested into the system.

```json
{
  "node_id": "repository_contextweave",
  "node_type": "Repository",
  "properties": {
    "repo_id": "contextweave",
    "production_status": "production"
  }
}
```

#### Document
A specific file within a repository. Carries a `doc_type` property set by
the routing analyzer, linking it to the routing sub-graph.

```json
{
  "node_id": "document_architecture_md",
  "node_type": "Document",
  "properties": {
    "source_file": "architecture.md",
    "file_type": "markdown",
    "weight": 1.0,
    "doc_type": "technical_spec"
  }
}
```

#### Skill
```json
{
  "node_id": "skill_graphrag",
  "node_type": "Skill",
  "properties": { "category": "skill", "frequency": 12 }
}
```

#### Pattern
```json
{
  "node_id": "pattern_serverless",
  "node_type": "Pattern",
  "properties": { "pattern_name": "serverless", "frequency": 16 }
}
```

#### Technology
```json
{
  "node_id": "technology_python",
  "node_type": "Technology",
  "properties": { "frequency": 20 }
}
```

#### AWSService
```json
{
  "node_id": "awsservice_amazon_bedrock",
  "node_type": "AWSService",
  "properties": { "service_name": "Amazon Bedrock", "frequency": 22 }
}
```

#### ArchitectureStyle
```json
{
  "node_id": "architecturestyle_rag",
  "node_type": "ArchitectureStyle",
  "properties": { "full_name": "Retrieval-Augmented Generation" }
}
```

#### Evidence
```json
{
  "node_id": "evidence_graphrag_implementation",
  "node_type": "Evidence",
  "properties": { "source": "architecture.md", "type": "implementation", "strength": "strong" }
}
```

#### Claim
```json
{
  "node_id": "claim_expert_aws_serverless",
  "node_type": "Claim",
  "properties": { "claim_type": "skill_claim" }
}
```

---

### Expertise Edge Types

#### BUILT — `Person → Repository`
The developer built or was a primary contributor to this repository.

#### CONTAINS — `Repository → Document`
The repository contains this document.

#### USES_TECH — `Document/Repo → Technology`
This document or repository uses the specified technology.

#### USES_AWS_SERVICE — `Document/Repo → AWSService`
This document or repository uses the specified AWS service.

#### DEMONSTRATES_PATTERN — `Document/Person → Pattern`
This document or the developer demonstrates this pattern.

#### SUPPORTS_CLAIM — `Evidence → Claim`
This evidence supports (backs up) the claim.

#### INDICATES_SKILL — `Document → Skill`
This document indicates (suggests) this skill.

#### DEMONSTRATES_SKILL — `Person → Skill`
The developer demonstrates this skill (stronger assertion than INDICATES_SKILL).

#### STRENGTHENS — `Skill → Pattern` or `Technology → AWSService`
Corroborating relationship: using this skill/tech strengthens the related
pattern/service association. Weight starts at 0.5.

---

## Part 2 — Routing Graph

The routing graph enables the RAGRouter to make data-driven retrieval decisions
and improves continuously as queries are answered.

### Routing Node Types

#### DocumentType
One node per document type. Used in two roles:
1. As the target of `HAS_TYPE` edges from Document nodes (ingestion side)
2. As a proxy for question type in `EFFECTIVE_FOR` edges (query side)

```json
{
  "node_id": "documenttype_technical_spec",
  "node_type": "DocumentType",
  "properties": {
    "label": "technical_spec",
    "question_type": "architecture",
    "avg_heading_count": 5,
    "avg_code_ratio": 0.12,
    "avg_sentence_len": 12.3,
    "doc_count": 2
  }
}
```

Document type labels: `technical_spec`, `narrative`, `structured_data`, `code`,
`diagram_derived`.

Question type labels (used as `question_type` property): `skill_depth`,
`architecture`, `project`, `comparison`, `credential`, `general`.

#### ChunkingStrategy
One node per chunking strategy.

```json
{
  "node_id": "chunkingstrategy_hierarchical",
  "node_type": "ChunkingStrategy",
  "properties": { "label": "hierarchical" }
}
```

Strategy labels: `hierarchical`, `sentence`, `fixed_512`, `fixed_256`.

#### RAGStrategy
One node per retrieval strategy.

```json
{
  "node_id": "ragstrategy_graph_first",
  "node_type": "RAGStrategy",
  "properties": { "label": "graph_first" }
}
```

Strategy labels: `graph_first`, `hybrid`, `keyword_boosted`, `semantic_search`.

---

### Routing Edge Types

#### HAS_TYPE — `Document → DocumentType`
Records the routing analyzer's classification of this document.

```json
{
  "from": "document_architecture_md",
  "rel":  "HAS_TYPE",
  "to":   "documenttype_technical_spec",
  "weight": 1.0
}
```

#### CHUNKED_WITH — `Document → ChunkingStrategy`
Records which chunking strategy was recommended for this document.

```json
{
  "from": "document_architecture_md",
  "rel":  "CHUNKED_WITH",
  "to":   "chunkingstrategy_hierarchical",
  "weight": 1.0
}
```

#### EFFECTIVE_FOR — `RAGStrategy → DocumentType`
The core learning edge. Weight represents how effective this RAG strategy is
for questions of this type. Updated after every query.

```json
{
  "from": "ragstrategy_graph_first",
  "rel":  "EFFECTIVE_FOR",
  "to":   "documenttype_architecture",
  "weight": 0.85,
  "properties": {
    "feedback_count": 12,
    "seeded": true
  }
}
```

**Initial prior weights** (seeded at deploy time):

| RAGStrategy | Question type | Initial weight |
|---|---|---|
| `graph_first` | `skill_depth` | 0.80 |
| `graph_first` | `architecture` | 0.80 |
| `keyword_boosted` | `credential` | 0.80 |
| `hybrid` | `comparison` | 0.70 |
| `hybrid` | `architecture` | 0.70 |
| `keyword_boosted` | `project` | 0.70 |
| `semantic_search` | `general` | 0.70 |
| … | … | … |

**Weight update rules**:
- `answer confidence ≥ 0.70` → `weight += 0.05` (reinforcement, cap 1.00)
- `answer confidence < 0.40` → `weight -= 0.02` (penalisation, floor 0.10)
- `0.40 ≤ confidence < 0.70` → no change

---

## Example openCypher Queries

### Expertise queries

#### Top skills by frequency across all repositories
```cypher
MATCH (p:Person)-[:DEMONSTRATES_SKILL]->(s:Skill)
RETURN s.label AS skill, s.properties.frequency AS frequency
ORDER BY frequency DESC
LIMIT 20
```

#### Repositories that demonstrate the serverless pattern
```cypher
MATCH (repo:Repository)-[:DEMONSTRATES_PATTERN]->(pat:Pattern)
WHERE toLower(pat.label) = 'serverless'
RETURN repo.label AS repository, pat.label AS pattern
```

#### AWS services used across multiple repositories
```cypher
MATCH (repo:Repository)-[:USES_AWS_SERVICE]->(svc:AWSService)
WITH svc, count(DISTINCT repo) AS repo_count, collect(DISTINCT repo.label) AS repos
WHERE repo_count >= 2
RETURN svc.label AS service, repo_count, repos
ORDER BY repo_count DESC
```

#### Skill neighbourhood: skills co-occurring with Python
```cypher
MATCH (doc:Document)-[:USES_TECH]->(tech:Technology)
WHERE toLower(tech.label) = 'python'
MATCH (doc)-[:USES_TECH|INDICATES_SKILL]->(related)
RETURN related.label AS co_occurring_skill, labels(related)[0] AS type
ORDER BY co_occurring_skill
LIMIT 20
```

#### Full person expertise summary
```cypher
MATCH (p:Person)
OPTIONAL MATCH (p)-[:DEMONSTRATES_SKILL]->(s:Skill)
OPTIONAL MATCH (p)-[:DEMONSTRATES_PATTERN]->(pat:Pattern)
OPTIONAL MATCH (p)-[:BUILT]->(repo:Repository)-[:USES_AWS_SERVICE]->(svc:AWSService)
RETURN
    p.label AS person,
    collect(DISTINCT s.label)[..20] AS skills,
    collect(DISTINCT pat.label)[..15] AS patterns,
    collect(DISTINCT svc.label)[..20] AS aws_services,
    count(DISTINCT repo) AS repo_count
LIMIT 1
```

---

### Routing queries

#### Current strategy weights for a question type
```cypher
MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
WHERE d.question_type = 'architecture'
RETURN r.label AS strategy, e.weight AS weight,
       coalesce(e.feedback_count, 0) AS queries_answered
ORDER BY e.weight DESC
```

#### Document type distribution (what has been ingested)
```cypher
MATCH (doc:Document)-[:HAS_TYPE]->(dt:DocumentType)
RETURN dt.label AS doc_type, count(doc) AS doc_count
ORDER BY doc_count DESC
```

#### Most-used chunking strategy
```cypher
MATCH (doc:Document)-[:CHUNKED_WITH]->(cs:ChunkingStrategy)
RETURN cs.label AS strategy, count(doc) AS doc_count
ORDER BY doc_count DESC
```

#### Strategies that have been reinforced most (feedback count)
```cypher
MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
WHERE e.feedback_count > 0
RETURN r.label AS strategy, d.question_type AS question_type,
       e.weight AS weight, e.feedback_count AS total_feedback
ORDER BY e.feedback_count DESC
LIMIT 20
```

#### Routing graph health: see all EFFECTIVE_FOR weights
```cypher
MATCH (r:RAGStrategy)-[e:EFFECTIVE_FOR]->(d:DocumentType)
RETURN r.label AS strategy, d.question_type AS question_type,
       e.weight AS weight, coalesce(e.feedback_count, 0) AS feedback_count
ORDER BY d.question_type, e.weight DESC
```

---

## Node ID Convention

Node IDs are deterministic slugs:
```
<node_type_lower>_<normalised_label>
```
where `normalised_label` replaces non-alphanumeric characters with underscores.

Examples:
- `Person` / "Rajat Arun" → `person_rajat_arun`
- `AWSService` / "Amazon Bedrock" → `awsservice_amazon_bedrock`
- `Pattern` / "event-driven" → `pattern_event_driven`
- `DocumentType` / "technical_spec" → `documenttype_technical_spec`
- `RAGStrategy` / "graph_first" → `ragstrategy_graph_first`

## Edge ID Convention

```
<from_id>__<relationship_type_lower>__<to_id>
```

Example:
```
person_rajat_arun__demonstrates_skill__skill_graphrag
ragstrategy_graph_first__effective_for__documenttype_architecture
document_architecture_md__has_type__documenttype_technical_spec
```

---

## Confidence and Weight Scoring

**Expertise nodes** have a `confidence` property (0.0–1.0):
- Starts at the source weight of the first file where the entity appears
- Increases by `+0.05` each time the entity is seen in an additional file
- Capped at 1.0

**Expertise edges** have a `weight` property (0.0–1.0):
- Set to `source_weight × frequency_boost` at creation
- Increases by `+0.10` on each additional observation
- Capped at 1.0

**`EFFECTIVE_FOR` routing edges** have a `weight` property (0.0–1.0):
- Seeded from `ROUTING_PRIORS` in `src/shared/models.py` (range 0.50–0.80)
- Updated after each query: `+0.05` on high confidence, `−0.02` on low confidence
- Clamped to [0.10, 1.00]
- `feedback_count` tracks total number of updates to each edge
