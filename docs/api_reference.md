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
retrieves relevant evidence from the Bedrock Knowledge Base, optionally expands
the graph context via Neptune Analytics, and synthesizes a grounded answer.

#### Request

**Content-Type**: `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | – | The question to answer (max 2000 chars) |
| `topK` | integer | No | 10 | Number of chunks to retrieve (1–50) |
| `includeGraphExpansion` | boolean | No | `true` | Enable Neptune Analytics graph expansion |
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
  "confidence": 0.94,
  "questionType": "architecture",
  "graphEntitiesUsed": ["Amazon Bedrock", "Neptune Analytics", "serverless"],
  "retrievalCount": 8,
  "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "latencyMs": 2341
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

Health check endpoint. Returns current configuration state.

#### Response

**HTTP 200 OK**:
```json
{
  "status": "healthy",
  "knowledgeBaseId": "KBID1234567890",
  "neptuneGraphId": "g-ABCDEF123456",
  "environment": "dev"
}
```

---

## Example curl Commands

### Health check
```bash
curl -s https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/health | jq .
```

### Query expertise
```bash
curl -s -X POST \
  https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/query-expertise \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What AWS services has this developer built production systems with?",
    "topK": 10
  }' | jq .
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

## Question Types and Optimisation

The system auto-classifies questions to optimise retrieval:

| Type | Triggered by | Best for |
|------|-------------|----------|
| `skill_depth` | "expert", "proficient", "years", "how well" | Skill level questions |
| `architecture` | "architect", "design", "pattern", "serverless" | Architecture questions |
| `project` | "built", "project", "implemented", "shipped" | Project history |
| `comparison` | "vs", "prefer", "trade-off", "better" | Technology comparisons |
| `credential` | "certified", "education", "training" | Credentials |
| `general` | (fallback) | General expertise queries |

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

**Recommended models** (as of 2025):

| Model | ID | Use case |
|-------|-----|----------|
| Claude 3.5 Sonnet v2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Best quality, recommended |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` | Low latency / cost |
| Amazon Nova Pro | `amazon.nova-pro-v1:0` | AWS-native, balanced |

All models must be **enabled in Bedrock Model Access** in your account.

See `src/query_api/synthesizer.py` for the full TODO marker and configuration
instructions.
