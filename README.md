ContextWeave

A Governed Retrieval Infrastructure on AWS

Production-grade Resume RAG system built with:
	•	AWS Lambda (Node.js 20)
	•	Amazon Bedrock (LLM + Embeddings)
	•	Bedrock Guardrails
	•	Amazon RDS PostgreSQL + pgvector
	•	Amazon S3 (Document storage)
	•	API Gateway (API Key protected)
	•	AWS WAF (Rate limiting)
	•	CloudFormation (Full IaC)

⸻

🚀 What This Project Solves

Recruiters don’t want to read PDFs.

They want:
	•	Instant skill validation
	•	Contextual answers
	•	No hallucinations
	•	No unsafe responses
	•	No data leakage

This system provides:

✔ Retrieval-Augmented Generation
✔ Strict Guardrails
✔ Resume-only scope enforcement
✔ PII blocking
✔ Prompt injection protection
✔ Rate limiting (20/min per IP equivalent)
✔ Fully reproducible infrastructure

⸻

🏗 Architecture Overview

S3 (Resume + Research Docs)
        ↓
Lambda Ingestion
        ↓
Bedrock Embeddings
        ↓
Postgres (pgvector)
        ↓
API Gateway (/chat)
        ↓
Lambda Retrieval
        ↓
Bedrock LLM
        ↓
Guardrail Output Filter


⸻

📦 Project Structure

.
├── src/
│   └── index.mjs              # Full Lambda handler
├── scripts/
│   ├── content-policy.json
│   ├── topic-policy.json
│   ├── word-policy.json
│   ├── sensitive-policy.json
│   └── contextual-grounding.json
├── deploy.sh                  # Build + deploy lambda
├── guardrail.sh               # Creates Bedrock Guardrail
├── template.yml               # Full CloudFormation stack
├── package.json
└── README.md


⸻

🔐 Guardrail Coverage

Content Policy
	•	Hate
	•	Insults
	•	Sexual
	•	Violence
	•	Misconduct
	•	Prompt attacks

Topic Policy
	•	Off-topic assistant usage
	•	Personal private info
	•	Fabrication / impersonation
	•	Confidential employer info
	•	Jailbreak attempts

Word Policy
	•	Prompt override attempts
	•	DAN jailbreak phrases
	•	System override phrases

Sensitive Information Policy

Blocks:
	•	SSN
	•	Bank details
	•	Passwords
	•	AWS keys
	•	Credit cards
	•	IP addresses
	•	Personal address
	•	Vehicle IDs
	•	And more

Includes:
	•	LinkedIn URL anonymization

Contextual Grounding
	•	Grounding threshold: 0.80
	•	Relevance threshold: 0.80

⸻

🧱 Infrastructure Deployment

1️⃣ Upload Lambda Zip

aws s3 cp function.zip s3://<your-bucket>/lambda/function.zip


⸻

2️⃣ Deploy CloudFormation

aws cloudformation deploy \
  --template-file template.yml \
  --stack-name contextweave-stack \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    LambdaCodeS3Bucket=<bucket> \
    LambdaCodeS3Key=lambda/function.zip \
    DatabasePassword=YourStrongPassword123! \
    BedrockModelId=anthropic.claude-3-sonnet-20240229-v1:0


⸻

🛡 Create Guardrail

chmod +x guardrail.sh
./guardrail.sh

Take the returned:

guardrailId

Then redeploy Lambda with:

export GUARDRAIL_ID=<id>
export GUARDRAIL_VERSION=1
./deploy.sh


⸻

📥 Ingest Documents

Upload files to:

s3://<DocsBucket>/docs/

Supported:
	•	PDF
	•	DOCX
	•	TXT
	•	MD
	•	JSON
	•	CSV

Ingestion is automatic via S3 event trigger.

⸻

💬 Chat Endpoint

POST /chat

Example:

curl https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/chat \
  -H "x-api-key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"question":"What cloud technologies does Tarun specialize in?"}'

Response:

{
  "answer": "...",
  "citations": [...]
}


⸻

🧠 How Retrieval Works
	1.	Embed question using Titan embedding model
	2.	Query pgvector using cosine similarity
	3.	Retrieve Top-K chunks
	4.	Construct grounded prompt
	5.	Generate answer
	6.	Apply output guardrail
	7.	Return structured response

⸻

⚙ Environment Variables

Required:

DATABASE_URL
BEDROCK_MODEL_ID
BEDROCK_EMBED_MODEL_ID
GUARDRAIL_ID
GUARDRAIL_VERSION

Optional:

DEFAULT_TOP_K
MAX_CONTEXT_CHARS
CHUNK_SIZE
CHUNK_OVERLAP


⸻

🔒 Security Controls
	•	Private RDS (no public access)
	•	Lambda in private subnets
	•	S3 VPC endpoint
	•	API key required
	•	WAF rate limiting
	•	Guardrails applied on INPUT and OUTPUT
	•	No raw model exposure
	•	No direct database exposure

⸻

📊 Observability

Structured logs:
	•	vector_op_selftest
	•	qvec_preview
	•	retrieval
	•	ingest_extract
	•	ingest_chunk
	•	guardrail_blocked_input
	•	guardrail_blocked_output

CloudWatch ready.

⸻

💰 Cost Considerations
	•	RDS (db.t4g.medium default)
	•	NAT Gateway (largest infra cost)
	•	Bedrock usage (tokens + embeddings)
	•	Lambda compute
	•	WAF + API Gateway

This is production architecture, not a toy demo.

⸻

🎯 Intended Use

This is not a generic chatbot.

It is:

A governed, infrastructure-first AI system designed to answer strictly within the boundaries of a professional resume 
🏷 Suggested Project Title
