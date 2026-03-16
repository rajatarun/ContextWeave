# Inspect or reset Neptune routing weights

View or reset the adaptive RAG routing EFFECTIVE_FOR edge weights in Neptune Analytics.

**Step 1 — Get Neptune graph ID**:
```bash
aws cloudformation describe-stacks --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`NeptuneGraphId`].OutputValue' \
  --output text
```

**Step 2 — Query current weights** using the IngestTriggerFunction with a status check, or invoke directly:
```bash
aws lambda invoke \
  --function-name expertise-rag-ingestion-trigger-dev \
  --payload '{"action": "status"}' \
  /tmp/routing-status.json && cat /tmp/routing-status.json
```

Display a table of all `EFFECTIVE_FOR` edge weights:

| RetrievalStrategy | QuestionType | Weight |
|---|---|---|
| graph_first | skill_depth | ? |
| graph_first | architecture | ? |
| hybrid | comparison | ? |
| keyword_boosted | project | ? |
| keyword_boosted | credential | ? |
| semantic_search | general | ? |

If weights look degraded (many < 0.30), offer to re-seed with initial priors by invoking:
```bash
aws lambda invoke \
  --function-name expertise-rag-ingestion-trigger-dev \
  --payload '{"action": "seed_routing"}' \
  /tmp/seed-result.json && cat /tmp/seed-result.json
```

**Initial prior weights** (for reference):
- `graph_first` → skill_depth: 0.70, architecture: 0.70
- `hybrid` → comparison: 0.70
- `keyword_boosted` → project: 0.70, credential: 0.70
- `semantic_search` → general: 0.60
