# Test query API locally or against deployed endpoint

Test the POST /query-expertise endpoint. Usage: /test-query [question]

Question: `$ARGUMENTS`

If a question is provided, construct a query payload and test it. Otherwise use the default test question "What AWS services does this developer have the most experience with?".

**Option 1 — Local SAM invoke** (if .aws-sam/build exists):
```bash
sam local invoke QueryAPIFunction --event events/api_query_event.json
```

**Option 2 — Against deployed dev endpoint**:
First get the API endpoint from the stack:
```bash
aws cloudformation describe-stacks --stack-name expertise-rag-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
  --output text
```

Then POST to the endpoint with:
```json
{
  "question": "<your question>",
  "maxResults": 5,
  "includeGraphExpansion": true
}
```

Show the full response including `answer`, `confidence`, `questionType`, `routingDecision`, and `sources`. If confidence is low (< 0.40), note that the routing weights will be penalized.
