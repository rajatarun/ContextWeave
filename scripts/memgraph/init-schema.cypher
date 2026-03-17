-- scripts/memgraph/init-schema.cypher
-- Unique constraints and indexes for the ExpertiseRAG knowledge graph.
--
-- Run after deploy:
--   cat scripts/memgraph/init-schema.cypher \
--     | docker exec -i memgraph mgconsole

-- ── Unique constraints ────────────────────────────────────────────────────────
CREATE CONSTRAINT ON (n:Node) ASSERT n.id IS UNIQUE;

-- ── Indexes for common query patterns ─────────────────────────────────────────
CREATE INDEX ON :Node(created_at);
CREATE INDEX ON :Node(status);

-- ── Verify ────────────────────────────────────────────────────────────────────
SHOW CONSTRAINT INFO;
SHOW INDEX INFO;
