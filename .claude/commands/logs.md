# Tail Lambda CloudWatch logs

Stream recent logs from a Lambda function. Usage: /logs [function-name]

Function argument: `$ARGUMENTS` (defaults to `preprocessor` if not specified)

Map argument to log group:
- `preprocessor` → `/aws/lambda/expertise-rag-preprocessor-dev`
- `query` or `query-api` → `/aws/lambda/expertise-rag-query-api-dev`
- `ingestion` or `ingestion-trigger` → `/aws/lambda/expertise-rag-ingestion-trigger-dev`

Run:
```bash
aws logs tail <log-group> --follow --format short
```

Show the last 50 log lines. Look for ERROR or WARN entries and highlight them. If you see a Python traceback, read the relevant source file and explain the error with a suggested fix.

For structured JSON logs (AWS Lambda Powertools format), parse and display:
- `message` field
- `error` field (if present)
- `extra` fields relevant to the operation
