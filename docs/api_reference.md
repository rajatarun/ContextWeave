# ExpertiseRAG API Reference

## Base URL

```
https://<api-id>.execute-api.us-east-1.amazonaws.com/<environment>
```

The full URL is available in the CloudFormation stack output `QueryExpertiseURL`.

---

## Endpoints

### POST /query-expertise

Submit a natural language question about the developer's expertise. The system
classifies the question, consults the Neptune routing graph to select the optimal
retrieval strategy, retrieves evidence, expands graph context, and synthesises a
grounded answer. After synthesis, confidence feedback is written back to Neptune
to improve future routing decisions.

#### Request

**Content-Type**: `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | – | The question to answer (max 2000 chars) |
| `topK` | integer | No | 10 | Number of chunks to retrieve (1–50) |
| `includeGraphExpansion` | boolean | No | `true` | Enable Neptune graph expansion. Note: `graph_first` and `hybrid` strategies force graph expansion regardless of this flag. |
| `minConfidence` | float | No | `0.3` | Minimum retrieval score threshold (0.0–1.0) |
| `questionType` | string | No | auto-classified | Force classification: `skill_depth`, `architecture`, `project`, `comparison`, `credential`, `general` |

**Example request body**:
```json
{
  "question": "What AWS services has this developer built production systems with?",
  "topK": 10,
  "includeGraphExpansion": true,
  "minConfidence": 0.3
}
```

#### Response

**HTTP 200 OK**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Grounded prose answer |
| `sources` | array | Cited evidence chunks |
| `inferredSkills` | array | Skills inferred from evidence + graph |
| `repeatedPatterns` | array | Patterns seen across multiple repos |
| `confidence` | float | Answer confidence (0.0–1.0) |
| `questionType` | string | Classified question type |
| `graphEntitiesUsed` | array | Graph nodes consulted during expansion |
| `retrievalCount` | integer | Number of chunks retrieved |
| `modelId` | string | Bedrock model used for synthesis |
| `latencyMs` | integer | Total pipeline latency in milliseconds |
| `routingDecision` | object | Routing agent decision details (see below) |

**`routingDecision` object**:

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | string | RAG strategy selected: `graph_first`, `hybrid`, `keyword_boosted`, `semantic_search` |
| `questionType` | string | Question type used for routing lookup |
| `strategyConfidence` | float | `EFFECTIVE_FOR` edge weight that drove the decision (0.0–1.0) |
| `graphExpansionForced` | boolean | `true` if strategy forced graph expansion (graph_first / hybrid) |
| `keywordBoostApplied` | boolean | `true` if keyword-overlap reranking was applied |
| `neptuneVectorsUsed` | boolean | `true` if Neptune vector search was used in addition to Bedrock KB |

**Source object**:
```json
{
  "file": "architecture.md",
  "excerpt": "...",
  "weight": 1.0,
  "score": 0.94,
  "effectiveScore": 0.94,
  "sourceUri": "s3://bucket/raw/repo/architecture.md"
}
```

**Example response**:
```json
{
  "answer": "Based on architecture.md and CLAUDE.md (both authoritative sources), this developer has built production systems with...",
  "sources": [
    { "file": "architecture.md", "weight": 1.0, "score": 0.94, "effectiveScore": 0.94 }
  ],
  "inferredSkills": ["AWS Lambda", "Amazon Bedrock", "Neptune Analytics", "GraphRAG"],
  "repeatedPatterns": ["serverless", "event-driven", "infrastructure-as-code"],
  "confidence": 0.95,
  "questionType": "architecture",
  "graphEntitiesUsed": ["Amazon Bedrock", "Neptune Analytics", "serverless"],
  "retrievalCount": 4,
  "modelId": "us.amazon.nova-pro-v1:0",
  "latencyMs": 11785,
  "routingDecision": {
    "strategy": "graph_first",
    "questionType": "architecture",
    "strategyConfidence": 0.82,
    "graphExpansionForced": true,
    "keywordBoostApplied": false,
    "neptuneVectorsUsed": false
  }
}
```

#### Error Responses

**HTTP 400 Bad Request**:
```json
{
  "error": "Invalid request",
  "details": "Body must be JSON with a non-empty 'question' field."
}
```

**HTTP 500 Internal Server Error**:
```json
{
  "error": "Service misconfigured",
  "details": "KNOWLEDGE_BASE_ID environment variable is not set"
}
```

---

### GET /health

Health check endpoint. Returns current configuration state and routing graph
document-type distribution.

#### Response

**HTTP 200 OK**:
```json
{
  "status": "healthy",
  "knowledgeBaseId": "KBID1234567890",
  "neptuneGraphId": "g-ABCDEF123456",
  "environment": "dev",
  "routingGraph": {
    "documentTypeDistribution": [
      { "doc_type": "technical_spec", "doc_count": 2 },
      { "doc_type": "structured_data", "doc_count": 1 },
      { "doc_type": "diagram_derived", "doc_count": 2 }
    ]
  }
}
```

`documentTypeDistribution` shows how many documents of each type have been
ingested and classified by the routing analyzer. An empty array indicates no
documents have been processed yet or Neptune is unreachable.

---

## Example curl Commands

### Health check
```bash
curl -s https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/health | jq .
```

### Query with default settings
```bash
curl -s -X POST \
  https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/query-expertise \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What AWS services has this developer built production systems with?",
    "topK": 10
  }' | jq .
```

### Show only the routing decision
```bash
curl -s -X POST \
  https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/query-expertise \
  -H "Content-Type: application/json" \
  -d '{"question": "How expert is this developer with Neptune Analytics?"}' \
  | jq .routingDecision
```

### Force a specific question type (bypass auto-classification)
```bash
curl -s -X POST \
  https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/query-expertise \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Describe the event-driven design",
    "questionType": "architecture"
  }' | jq '{answer, routingDecision}'
```

### Query with graph expansion disabled
```bash
curl -s -X POST \
  https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/query-expertise \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What programming languages does this developer use?",
    "topK": 5,
    "includeGraphExpansion": false
  }' | jq .answer
```

### Using the Python script
```bash
# Direct Bedrock retrieve (no synthesis)
python scripts/query_expertise.py bedrock \
  --knowledge-base-id KBID1234567890 \
  --question "What AWS services has this developer used?"

# Full API call
python scripts/query_expertise.py api \
  --endpoint https://<api-id>.execute-api.us-east-1.amazonaws.com/dev \
  --question "What architecture patterns does this developer repeatedly apply?" \
  --top-k 10
```

---

## Question Types, Routing and Strategy Selection

The system auto-classifies questions and routes to the optimal retrieval strategy:

| Type | Triggered by | Default strategy | Prior weight |
|------|-------------|------------------|-------------|
| `skill_depth` | "expert", "proficient", "years", "how well", "experience", "deep" | `graph_first` | 0.80 |
| `architecture` | "architect", "design", "pattern", "serverless", "infrastructure" | `graph_first` | 0.80 |
| `project` | "built", "project", "implemented", "deployed", "shipped" | `keyword_boosted` | 0.70 |
| `comparison` | "vs", "prefer", "trade-off", "versus", "better" | `hybrid` | 0.70 |
| `credential` | "certif", "course", "training", "degree", "award" | `keyword_boosted` | 0.80 |
| `general` | (fallback) | `semantic_search` | 0.70 |

These weights are **initial priors**. After each query, the winning strategy's
`EFFECTIVE_FOR` edge weight is updated in Neptune based on the answer confidence:
- `confidence ≥ 0.70` → weight `+0.05` (reinforcement)
- `confidence < 0.40` → weight `−0.02` (penalisation)
- Neutral zone (`0.40–0.70`) → no change

The router always picks the strategy with the highest current weight for the
detected question type. Strategies compete and the graph learns which works best.

### Retrieval strategy behaviours

| Strategy | Bedrock KB | Neptune vectors | Keyword boost | Graph expansion |
|---|---|---|---|---|
| `semantic_search` | ✓ | ✗ | ✗ | Optional (request flag) |
| `graph_first` | ✓ | ✗ | ✗ | Always forced |
| `keyword_boosted` | ✓ | ✗ | ✓ (25/75 blend) | Optional |
| `hybrid` | ✓ | ✓ | ✓ | Always forced |

---

## Generation Model Configuration

The synthesis step requires a Bedrock foundation model. Set the
`GENERATION_MODEL_ID` environment variable on the `QueryAPIFunction` Lambda:

```yaml
# In template.yaml Globals or per-function:
Environment:
  Variables:
    GENERATION_MODEL_ID: anthropic.claude-3-5-sonnet-20241022-v2:0
```

**Recommended models** (as of 2026):

| Model | ID | Use case |
|-------|-----|----------|
| Claude 3.5 Sonnet v2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Best quality, recommended |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` | Low latency / cost |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | AWS-native, balanced |

All models must be **enabled in Bedrock Model Access** in your account.
