#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"

aws bedrock create-guardrail   --region "$REGION"   --name "resume-rag-guardrail"   --description "Guardrail for personal resume RAG bot answering questions about work experience and skills"   --blocked-input-messaging "I can only answer questions about professional experience, skills, and background. Please ask something relevant."   --blocked-outputs-messaging "I was unable to generate a response within the scope of this resume assistant. Please rephrase your question."   --content-policy-config file://scripts/content-policy.json   --topic-policy-config file://scripts/topic-policy.json   --word-policy-config file://scripts/word-policy.json   --sensitive-information-policy-config file://scripts/sensitive-policy.json   --contextual-grounding-policy-config file://scripts/contextual-grounding.json   --tags '[
    { "key": "Project",     "value": "ResumeBot"  },
    { "key": "Environment", "value": "Production" },
    { "key": "Owner",       "value": "Personal"   },
    { "key": "ManagedBy",   "value": "IaC"        }
  ]'
