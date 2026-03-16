# Show CloudFormation stack status

Get a comprehensive status report for the ContextWeave stacks. Usage: /stack-status [env]

Environment: `$ARGUMENTS` (defaults to `dev`)

Run:
```bash
aws cloudformation describe-stacks --stack-name expertise-rag-<env> \
  --query 'Stacks[0].{Status:StackStatus,Updated:LastUpdatedTime,Outputs:Outputs}'
```

Display:
1. **Stack status** (CREATE_COMPLETE, UPDATE_IN_PROGRESS, ROLLBACK_IN_PROGRESS, etc.)
2. **Last updated** timestamp
3. **Stack outputs** as a formatted table:
   - API endpoint URL
   - S3 artifacts bucket name
   - Neptune graph ID
   - Bedrock Knowledge Base ID
   - Step Functions state machine ARN
4. **Recent stack events** (last 10):
```bash
aws cloudformation describe-stack-events --stack-name expertise-rag-<env> \
  --query 'StackEvents[:10].{Time:Timestamp,Resource:LogicalResourceId,Status:ResourceStatus,Reason:ResourceStatusReason}'
```

If the stack is in a ROLLBACK or FAILED state, highlight the failing resource and its reason, then read the relevant template section from `template.yaml` to suggest a fix.
