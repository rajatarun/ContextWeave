"""
Content extractors for the ExpertiseRAG preprocessor Lambda.

Each extractor receives raw file content and returns:
- extracted_text  : clean text suitable for Bedrock KB ingestion
- summary         : short prose summary
- expertise_signals: list of ExpertiseSignal dicts
- metadata        : additional structured key/values

Supported file types:
  - Markdown (.md)
  - YAML (.yaml, .yml)
  - PlantUML (.puml, .plantuml, .pu)
  - Plain text (.txt, .rst)
"""
from __future__ import annotations

import re
import textwrap
from typing import Any

import yaml  # PyYAML – bundled with Lambda layer or requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Lowercase, strip non-alphanum to underscores for stable IDs."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _extract_headings(md: str) -> list[str]:
    return re.findall(r"^#{1,4}\s+(.+)$", md, re.MULTILINE)


def _extract_code_blocks(md: str) -> list[dict[str, str]]:
    """Return list of {lang, code} for fenced code blocks."""
    pattern = r"```(\w*)\n(.*?)```"
    results = []
    for m in re.finditer(pattern, md, re.DOTALL):
        lang = m.group(1).strip().lower() or "text"
        code = m.group(2).strip()
        if code:
            results.append({"lang": lang, "code": code})
    return results


# AWS service name patterns (non-exhaustive; extend as needed)
_AWS_SERVICE_PATTERNS = re.compile(
    r"\b(Amazon|AWS)\s+(S3|Lambda|DynamoDB|RDS|Aurora|Bedrock|SageMaker|"
    r"Neptune|Glue|Kinesis|SNS|SQS|EventBridge|Step Functions|ECS|EKS|"
    r"CloudFront|Route\s*53|IAM|KMS|Secrets Manager|Parameter Store|"
    r"CloudWatch|X-Ray|API Gateway|ALB|NLB|VPC|EC2|Fargate|ECR|"
    r"CodePipeline|CodeBuild|CDK|SAM|CloudFormation|AppSync|Cognito|"
    r"Amplify|Timestream|OpenSearch|Elasticsearch|MemoryDB|ElastiCache|"
    r"Athena|Redshift|QuickSight|Data Firehose|MSK|Transfer Family|"
    r"DataSync|Backup|Organizations|Control Tower|Security Hub|GuardDuty|"
    r"WAF|Shield|Macie|Inspector)\b",
    re.IGNORECASE,
)

# Common technology / framework patterns
_TECH_PATTERNS = re.compile(
    r"\b(Python|TypeScript|JavaScript|Java|Go|Rust|C\+\+|Kotlin|Swift|"
    r"Node\.js|React|Next\.js|Vue|Angular|FastAPI|Django|Flask|Spring Boot|"
    r"GraphQL|REST|gRPC|OpenAPI|Terraform|Pulumi|Ansible|Docker|Kubernetes|"
    r"Helm|ArgoCD|GitOps|Prometheus|Grafana|Jaeger|OpenTelemetry|"
    r"PostgreSQL|MySQL|MongoDB|Redis|Cassandra|Neo4j|"
    r"Kafka|RabbitMQ|NATS|Celery|"
    r"LangChain|LlamaIndex|RAG|LLM|GPT|Claude|Bedrock|Titan|"
    r"pgvector|Pinecone|Weaviate|Chroma|FAISS)\b",
    re.IGNORECASE,
)

# Architecture / design pattern keywords
_PATTERN_KEYWORDS = re.compile(
    r"\b(event[- ]driven|microservices?|serverless|hexagonal|clean architecture|"
    r"CQRS|event sourcing|saga pattern|strangler fig|sidecar|service mesh|"
    r"domain[- ]driven design|DDD|BFF|API Gateway pattern|"
    r"circuit breaker|bulkhead|retry|backoff|idempotent|"
    r"infrastructure[- ]as[- ]code|IaC|GitOps|blue[- ]green|canary|"
    r"zero[- ]downtime|eventual consistency|ACID|BASE|CAP theorem|"
    r"multi[- ]tenant|single[- ]tenant|multi[- ]region|active[- ]active|"
    r"async|asynchronous|pub[- ]sub|fan[- ]out|scatter[- ]gather|"
    r"GraphRAG|RAG|vector search|semantic search|embedding)\b",
    re.IGNORECASE,
)


def _find_aws_services(text: str) -> list[str]:
    return list({m.group(0) for m in _AWS_SERVICE_PATTERNS.finditer(text)})


def _find_technologies(text: str) -> list[str]:
    return list({m.group(0) for m in _TECH_PATTERNS.finditer(text)})


def _find_patterns(text: str) -> list[str]:
    return list({m.group(0) for m in _PATTERN_KEYWORDS.finditer(text)})


def _build_signals(text: str, source_file: str, weight: float) -> list[dict]:
    signals = []
    for svc in _find_aws_services(text):
        signals.append({
            "signal_type": "aws_service",
            "value": svc,
            "source_file": source_file,
            "weight": weight,
            "frequency": text.lower().count(svc.lower()),
        })
    for tech in _find_technologies(text):
        signals.append({
            "signal_type": "technology",
            "value": tech,
            "source_file": source_file,
            "weight": weight,
            "frequency": text.lower().count(tech.lower()),
        })
    for pat in _find_patterns(text):
        signals.append({
            "signal_type": "pattern",
            "value": pat,
            "source_file": source_file,
            "weight": weight,
            "frequency": text.lower().count(pat.lower()),
        })
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Markdown extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_markdown(content: str, source_file: str, weight: float = 0.6) -> dict[str, Any]:
    """Extract structured information from a Markdown file."""
    # Strip HTML comments
    text = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    # Strip image references
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Normalise link refs to just the label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    headings = _extract_headings(content)
    code_blocks = _extract_code_blocks(content)
    signals = _build_signals(text, source_file, weight)

    # Derive a short summary from the first 3 headings + first non-empty paragraph
    first_para = ""
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            first_para = textwrap.shorten(line, width=300)
            break

    summary_parts = [f"File: {source_file}"]
    if headings:
        summary_parts.append("Sections: " + ", ".join(headings[:5]))
    if first_para:
        summary_parts.append(first_para)
    summary = " | ".join(summary_parts)

    return {
        "extracted_text": text,
        "summary": summary,
        "expertise_signals": signals,
        "metadata": {
            "headings": headings,
            "code_block_count": len(code_blocks),
            "code_languages": list({cb["lang"] for cb in code_blocks}),
            "char_count": len(text),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# YAML extractor
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_yaml(obj: Any, prefix: str = "") -> list[str]:
    """Recursively flatten YAML to 'key: value' strings."""
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_key = f"{prefix}.{k}" if prefix else k
            lines.extend(_flatten_yaml(v, child_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            lines.extend(_flatten_yaml(item, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


def extract_yaml(content: str, source_file: str, weight: float = 0.7) -> dict[str, Any]:
    """Extract structured information from a YAML file (e.g. repo-signals.yaml)."""
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        return {
            "extracted_text": content,
            "summary": f"YAML parse error in {source_file}: {exc}",
            "expertise_signals": [],
            "metadata": {"parse_error": str(exc)},
        }

    flat_lines = _flatten_yaml(parsed)
    extracted_text = "\n".join(flat_lines)

    # For repo-signals.yaml, try to extract well-known keys
    metadata: dict[str, Any] = {}
    expertise_signals = []

    if isinstance(parsed, dict):
        # Skills section
        for skills_key in ("skills", "technologies", "tech_stack", "stack"):
            if skills_key in parsed:
                skills_data = parsed[skills_key]
                if isinstance(skills_data, list):
                    for sk in skills_data:
                        label = str(sk.get("name", sk) if isinstance(sk, dict) else sk)
                        expertise_signals.append({
                            "signal_type": "skill",
                            "value": label,
                            "source_file": source_file,
                            "weight": weight,
                            "frequency": 1,
                        })

        # Patterns section
        for pat_key in ("patterns", "architecture_patterns", "design_patterns"):
            if pat_key in parsed:
                for p in (parsed[pat_key] if isinstance(parsed[pat_key], list) else []):
                    label = str(p.get("name", p) if isinstance(p, dict) else p)
                    expertise_signals.append({
                        "signal_type": "pattern",
                        "value": label,
                        "source_file": source_file,
                        "weight": weight,
                        "frequency": 1,
                    })

        # AWS services section
        for svc_key in ("aws_services", "aws", "cloud_services"):
            if svc_key in parsed:
                for svc in (parsed[svc_key] if isinstance(parsed[svc_key], list) else []):
                    label = str(svc.get("name", svc) if isinstance(svc, dict) else svc)
                    expertise_signals.append({
                        "signal_type": "aws_service",
                        "value": label,
                        "source_file": source_file,
                        "weight": weight,
                        "frequency": 1,
                    })

        metadata = {k: v for k, v in parsed.items() if isinstance(v, (str, int, float, bool))}

    # Also run regex extraction over the raw YAML text
    expertise_signals.extend(_build_signals(content, source_file, weight))

    summary = f"YAML config: {source_file}. Keys: {', '.join(list(parsed.keys())[:10]) if isinstance(parsed, dict) else 'list'}."

    return {
        "extracted_text": extracted_text,
        "summary": summary,
        "expertise_signals": expertise_signals,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PlantUML extractor → textual architecture summary
# ─────────────────────────────────────────────────────────────────────────────

# Diagram type detection
_PUML_TYPES = {
    "@startuml": "sequence or component",
    "@startc4": "C4 architecture",
    "@startmindmap": "mindmap",
    "@startgantt": "gantt",
    "@startjson": "JSON structure",
}

# Component / service patterns in PlantUML
_PUML_COMPONENT = re.compile(
    r'(?:component|actor|database|queue|cloud|node|rectangle|package|'
    r'boundary|control|entity|collections|storage|frame)\s+"?([^"\n{]+)"?',
    re.IGNORECASE,
)

# Relationship arrows (PlantUML syntax: A --> B : label)
_PUML_ARROW = re.compile(
    r'"?([^"\n<>]+?)"?\s*(?:-->|->|<--|<-|\.\.>|\.\.|--)\s*"?([^"\n<>:]+?)"?\s*(?::\s*(.+))?$',
    re.MULTILINE,
)

# C4 macro calls: Component(alias, "Label", "Tech", "Desc")
_C4_ELEMENT = re.compile(
    r'(?:Person|System|Container|Component|Boundary|Rel|BiRel|Lay_)\w*\s*\(\s*(\w+)\s*,\s*"([^"]+)"',
    re.IGNORECASE,
)

# AWS PlantUML sprites: !include <awslib/...>
_AWS_INCLUDE = re.compile(
    r'!include\s+<awslib/([^/\n]+)/([^>\n]+)>',
    re.IGNORECASE,
)


def extract_plantuml(content: str, source_file: str, weight: float = 0.8) -> dict[str, Any]:
    """
    Convert PlantUML source into a textual architecture summary.

    Strategy:
    1. Detect diagram type (sequence, C4, component, AWS)
    2. Extract components / actors / services
    3. Extract relationships / arrows with labels
    4. Extract AWS service references from sprite includes
    5. Generate plain-text description suitable for LLM ingestion
    """
    diagram_type = "unknown"
    for marker, dtype in _PUML_TYPES.items():
        if marker.lower() in content.lower():
            diagram_type = dtype
            break

    components: list[str] = []
    relationships: list[str] = []
    aws_services: list[str] = []

    # Extract plain component names
    for m in _PUML_COMPONENT.finditer(content):
        label = m.group(1).strip().strip('"')
        if label and len(label) > 1:
            components.append(label)

    # Extract C4 elements
    for m in _C4_ELEMENT.finditer(content):
        label = m.group(2).strip()
        if label:
            components.append(label)

    # Extract relationships
    for m in _PUML_ARROW.finditer(content):
        src = m.group(1).strip().strip('"')
        dst = m.group(2).strip().strip('"')
        label = (m.group(3) or "").strip()
        if src and dst and len(src) > 1 and len(dst) > 1:
            rel = f"{src} → {dst}"
            if label:
                rel += f" [{label}]"
            relationships.append(rel)

    # Extract AWS service sprite names
    for m in _AWS_INCLUDE.finditer(content):
        svc_group = m.group(1)   # e.g. "Compute"
        svc_name = m.group(2).replace("_", " ").replace("-", " ")
        aws_services.append(f"AWS {svc_name}")

    # Also run regex extraction on raw content
    aws_services.extend(_find_aws_services(content))
    aws_services = list(set(aws_services))

    # Build the prose summary
    parts = [f"Architecture diagram ({diagram_type}) from {source_file}."]

    if components:
        unique_components = list(dict.fromkeys(components))[:20]
        parts.append(f"Components/services: {', '.join(unique_components)}.")

    if relationships:
        parts.append("Data flows and relationships:")
        for rel in relationships[:15]:
            parts.append(f"  - {rel}")

    if aws_services:
        unique_aws = list(dict.fromkeys(aws_services))[:15]
        parts.append(f"AWS services referenced: {', '.join(unique_aws)}.")

    extracted_text = "\n".join(parts)

    # Also include the raw PlantUML for completeness
    extracted_text += "\n\nRaw PlantUML source:\n" + content

    signals = _build_signals(content, source_file, weight)
    for svc in set(aws_services):
        signals.append({
            "signal_type": "aws_service",
            "value": svc,
            "source_file": source_file,
            "weight": weight,
            "frequency": content.lower().count(svc.lower()),
        })

    summary = f"PlantUML {diagram_type} diagram ({source_file}): {', '.join(list(dict.fromkeys(components))[:5])}."

    return {
        "extracted_text": extracted_text,
        "summary": summary,
        "expertise_signals": signals,
        "metadata": {
            "diagram_type": diagram_type,
            "component_count": len(set(components)),
            "relationship_count": len(relationships),
            "aws_service_count": len(set(aws_services)),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plain text extractor (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(content: str, source_file: str, weight: float = 0.4) -> dict[str, Any]:
    """Minimal extraction for plain text files."""
    text = re.sub(r"\n{3,}", "\n\n", content).strip()
    signals = _build_signals(text, source_file, weight)
    first_line = text.split("\n")[0][:200] if text else ""
    return {
        "extracted_text": text,
        "summary": f"Text file {source_file}: {first_line}",
        "expertise_signals": signals,
        "metadata": {"char_count": len(text)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def extract(content: str, source_file: str, weight: float) -> dict[str, Any]:
    """
    Dispatch to the right extractor based on file extension.
    Returns a dict with keys: extracted_text, summary, expertise_signals, metadata.
    """
    lower = source_file.lower()
    if lower.endswith((".puml", ".plantuml", ".pu", ".wsd")):
        return extract_plantuml(content, source_file, weight)
    elif lower.endswith((".yaml", ".yml")):
        return extract_yaml(content, source_file, weight)
    elif lower.endswith((".md", ".markdown", ".mdx")):
        return extract_markdown(content, source_file, weight)
    else:
        return extract_text(content, source_file, weight)
