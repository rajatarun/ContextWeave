# ExpertiseRAG Graph Schema

## Overview

The ExpertiseRAG graph is stored in Neptune Analytics and uses openCypher as
the query language. All nodes and edges are written by the Preprocessor Lambda
and exported to `derived/<repo>/graph_entities.json` and
`derived/<repo>/graph_edges.json`.

---

## Node Types

### Person
The central node representing the developer whose expertise is being captured.

```json
{
  "node_id": "person_jane_smith",
  "node_type": "Person",
  "label": "Jane Smith",
  "properties": {
    "name": "Jane Smith"
  }
}
```

### Repository
A git repository or project ingested into the system.

```json
{
  "node_id": "repository_my_saas_platform",
  "node_type": "Repository",
  "properties": {
    "repo_id": "my_saas_platform",
    "production_status": "production",
    "description": "Multi-tenant SaaS platform"
  }
}
```

### Document
A specific file within a repository.

```json
{
  "node_id": "document_architecture_md",
  "node_type": "Document",
  "properties": {
    "source_file": "architecture.md",
    "file_type": "markdown",
    "weight": 1.0
  }
}
```

### Skill
A professional or technical skill demonstrated by the developer.

```json
{
  "node_id": "skill_graphrag",
  "node_type": "Skill",
  "properties": {
    "category": "skill",
    "frequency": 12
  }
}
```

### Pattern
An architecture or design pattern used in one or more projects.

```json
{
  "node_id": "pattern_serverless",
  "node_type": "Pattern",
  "properties": {
    "pattern_name": "serverless",
    "frequency": 16
  }
}
```

### Technology
A programming language, framework, library, or tool.

```json
{
  "node_id": "technology_python",
  "node_type": "Technology",
  "properties": {
    "frequency": 20
  }
}
```

### AWSService
An Amazon Web Services product or service.

```json
{
  "node_id": "awsservice_amazon_bedrock",
  "node_type": "AWSService",
  "properties": {
    "service_name": "Amazon Bedrock",
    "frequency": 22
  }
}
```

### ArchitectureStyle
A high-level architectural paradigm (e.g., RAG, microservices).

```json
{
  "node_id": "architecturestyle_rag",
  "node_type": "ArchitectureStyle",
  "properties": {
    "full_name": "Retrieval-Augmented Generation"
  }
}
```

### Evidence
A specific piece of supporting evidence (e.g., a demonstrated implementation).

```json
{
  "node_id": "evidence_graphrag_implementation",
  "node_type": "Evidence",
  "properties": {
    "source": "architecture.md",
    "type": "implementation",
    "strength": "strong"
  }
}
```

### Claim
An expertise claim that is backed by one or more Evidence nodes.

```json
{
  "node_id": "claim_expert_aws_serverless",
  "node_type": "Claim",
  "properties": {
    "claim_type": "skill_claim",
    "supported_by": ["evidence_graphrag_implementation"]
  }
}
```

---

## Edge Types

### BUILT
`Person → Repository`
The developer built or was a primary contributor to this repository.

### CONTAINS
`Repository → Document`
The repository contains this document.

### USES_TECH
`Document → Technology` or `Repository → Technology`
This document or repository uses the specified technology.

### USES_AWS_SERVICE
`Document → AWSService` or `Repository → AWSService`
This document or repository uses the specified AWS service.

### DEMONSTRATES_PATTERN
`Document → Pattern` or `Person → Pattern`
This document or the developer demonstrates this pattern.

### SUPPORTS_CLAIM
`Evidence → Claim`
This evidence supports (backs up) the claim.

### INDICATES_SKILL
`Document → Skill`
This document indicates (suggests) this skill.

### DEMONSTRATES_SKILL
`Person → Skill`
The developer demonstrates this skill (stronger assertion than INDICATES_SKILL).

### STRENGTHENS
`Skill → Pattern` or `Technology → AWSService`
Corroborating relationship – using this skill/tech strengthens the related
pattern/service association.

---

## Example openCypher Queries

### Top skills by frequency across all repositories
```cypher
MATCH (p:Person)-[:DEMONSTRATES_SKILL]->(s:Skill)
RETURN s.label AS skill, s.properties.frequency AS frequency
ORDER BY frequency DESC
LIMIT 20
```

### Repositories that demonstrate the serverless pattern
```cypher
MATCH (repo:Repository)-[:DEMONSTRATES_PATTERN]->(pat:Pattern)
WHERE toLower(pat.label) = 'serverless'
RETURN repo.label AS repository, pat.label AS pattern
```

### AWS services used across multiple repositories (cross-repo evidence)
```cypher
MATCH (repo:Repository)-[:USES_AWS_SERVICE]->(svc:AWSService)
WITH svc, count(DISTINCT repo) AS repo_count, collect(DISTINCT repo.label) AS repos
WHERE repo_count >= 2
RETURN svc.label AS service, repo_count, repos
ORDER BY repo_count DESC
```

### Skill neighbourhood: skills co-occurring with Python
```cypher
MATCH (doc:Document)-[:USES_TECH]->(tech:Technology)
WHERE toLower(tech.label) = 'python'
MATCH (doc)-[:USES_TECH|INDICATES_SKILL]->(related)
RETURN related.label AS co_occurring_skill, labels(related)[0] AS type
ORDER BY co_occurring_skill
LIMIT 20
```

### Evidence backing expertise claims
```cypher
MATCH (e:Evidence)-[:SUPPORTS_CLAIM]->(c:Claim)
RETURN c.label AS claim, e.label AS evidence, e.properties.strength AS strength
ORDER BY c.label
```

### Full person expertise summary
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

## Node ID Convention

Node IDs are deterministic slugs constructed as:
```
<node_type_lower>_<normalised_label>
```

Where `normalised_label` replaces non-alphanumeric characters with underscores.

Examples:
- `Person` / "Jane Smith" → `person_jane_smith`
- `AWSService` / "Amazon Bedrock" → `awsservice_amazon_bedrock`
- `Pattern` / "event-driven" → `pattern_event_driven`

This ensures deduplication across multiple files referencing the same entity.

## Edge ID Convention

Edge IDs are constructed as:
```
<from_id>__<relationship_type_lower>__<to_id>
```

Example:
```
person_jane_smith__demonstrates_skill__skill_graphrag
```

---

## Confidence Scoring

Each node has a `confidence` property (0.0–1.0):
- Starts at the source weight of the first file where the entity appears
- Increases by 0.05 each time the entity is seen in an additional file
- Capped at 1.0

Each edge has a `weight` property (0.0–1.0):
- Set to the source weight × frequency boost at creation
- Increases by 0.1 on each additional observation
- Capped at 1.0
