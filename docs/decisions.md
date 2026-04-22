# Architecture Decision Records (ADRs)

This file captures key architecture decisions reflected in the current
implementation.

## ADR-001 — Use PostgreSQL/pgvector as the primary vector store
- **Status:** Accepted
- **Date:** 2026-04-22

### Context
The query path requires reliable, low-latency semantic retrieval and a place to
store reusable response cache entries.

### Decision
Use PostgreSQL with pgvector for:
- chunk embedding retrieval (`chunks` table)
- semantic response caching (`query_cache` table)

### Consequences
- Retrieval and caching share one operational datastore.
- SQL-based observability/debugging for retrieval and cache behavior.
- Requires pgvector extension management and index tuning.

---

## ADR-002 — Use Memgraph for expertise graph and adaptive routing graph
- **Status:** Accepted
- **Date:** 2026-04-22

### Context
The system needs graph traversals for evidence expansion and mutable routing
weights for strategy learning.

### Decision
Model both:
- expertise entities/relationships
- routing entities (`RAGStrategy`, `DocumentType`, `EFFECTIVE_FOR`)

inside Memgraph using openCypher.

### Consequences
- A single graph runtime for both semantic context expansion and routing policy.
- Routing can adapt online without model retraining.
- Operational dependence on Memgraph connectivity and graph health.

---

## ADR-003 — Apply adaptive strategy routing with confidence-based feedback
- **Status:** Accepted
- **Date:** 2026-04-22

### Context
Different question types benefit from different retrieval strategies.

### Decision
At query time:
1. classify question type,
2. select the best strategy from current `EFFECTIVE_FOR` weights,
3. update edge weights after synthesis confidence is known.

Weight update policy:
- reinforce on high confidence (`>= 0.70`)
- penalize on low confidence (`< 0.40`)
- clamp to `[0.10, 1.00]`

### Consequences
- Strategy behavior improves incrementally from production feedback.
- Adds learning-state drift risk; reset/seed operations remain important.

---

## ADR-004 — Keep CAG semantic cache in front of full RAG pipeline
- **Status:** Accepted
- **Date:** 2026-04-22

### Context
Repeated and near-duplicate questions add unnecessary retrieval/synthesis cost
and latency.

### Decision
Perform cache lookup before full RAG execution and bypass cache for
obviously time-sensitive questions.

### Consequences
- Lower average latency and model invocation cost on repeated queries.
- Requires TTL and similarity-threshold tuning to avoid stale reuse.

---

## ADR-005 — Preserve selected legacy naming for API compatibility
- **Status:** Accepted
- **Date:** 2026-04-22

### Context
Some request/response and internal parameter names reference legacy Neptune
terminology.

### Decision
Retain certain legacy names (for compatibility) while implementing graph logic
against Memgraph.

### Consequences
- Reduces immediate breaking changes for existing callers.
- Documentation must explicitly clarify legacy naming vs runtime behavior.
