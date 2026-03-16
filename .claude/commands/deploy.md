# Deploy to AWS environment

Deploy the ContextWeave stack to the specified environment. Usage: /deploy [env]

Environment argument: `$ARGUMENTS` (defaults to `dev` if not specified)

Determine the target environment from the argument (dev, staging, or prod). Then:

1. First run `sam validate --lint` to catch issues early
2. Run `sam build --parallel --cached`
3. Deploy using the appropriate config:
   - `dev`: `sam deploy --config-env default`
   - `staging`: `sam deploy --config-env staging`
   - `prod`: Ask for explicit confirmation before deploying to production, then run `sam deploy --config-env prod`

After deployment, show the stack outputs (API endpoint, S3 bucket, KB ID).

**Stack names**:
- dev: `expertise-rag-dev`
- staging: `expertise-rag-staging`
- prod: `expertise-rag-prod`

**AWS Account**: `239571291755` (teamweave), region: `us-east-1`
