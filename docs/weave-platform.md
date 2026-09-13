# The Weave platform — sixteen repositories and how they connect

**Method.** Every repository under `rajatarun` was read: README, entry points, exposed interfaces (HTTP routes, MCP tools, events), consumed services, data stores, models, tests, last activity, and every cross-repository reference in code, templates and CI. Four had already been examined to the line (mcp-observatory, ContextWeave, DeviceWeave, CipherWeave). Where this version corrects the previous one, it says so (§7).

---

## 1. The sixteen, one card each

Legend — **Exposes:** what others can call · **Consumes:** what it calls · **Stores:** state · **Weave role:** where it sits in §4.

### mcp-observatory — the control plane
Propose/commit gate for tool calls (argument-hashed, replay-protected commit tokens), six-signal risk vector, policy matrix by tool criticality, deterministic fallback router, shadow lane, span exporter. **Exposes:** Python library on PyPI (`mcp-observatory` 0.1.0/0.2.0/0.2.1) and — separately published — `@weaveaijs/mcp-observatory` 0.3.0 on npm (`InvocationWrapper`), a JS port that lives outside these sixteen repos. **Consumes:** nothing. **Stores:** proposals/commits/nonces (in-memory or Postgres). **Maturity:** 5.1k LOC, 36 tests, active Sep 2026; properties proved; two shipped defects fixed on the branch, **unreleased** (pyproject still 0.2.1). **Weave role:** every side effect passes through it — five Python consumers plus RoutineWeave in JS. **Gaps:** cut 0.3.0; fail closed on default secrets; upstream input-size limit.

### ContextWeave — the knowledge and learning layer
GraphRAG over Memgraph + pgvector with a semantic response cache and an adaptive router (Thompson sampling, health verdict, propensity logging, off-policy evaluation, `POST /feedback` human reward, `confidenceReported`). **Exposes:** `POST /query-expertise`, `POST /feedback`, `GET /health`, `GET /routing-decisions`. **Consumes:** Bedrock (Titan V2, Converse), Memgraph, Postgres, mcp-observatory (wrapper around model calls), shared stack (VPC, metrics table). **Stores:** chunks, `query_cache`, `routing_decisions`, routing graph. **Maturity:** 9.6k LOC, 44 tests, active; PR #141 open (5 commits). **Weave role:** the one component that learns from outcomes; the natural memory for every other agent. **Gaps:** live-store integration tests. `POST /feedback` had zero rows of human ratings ever recorded — the handler existed but had no API Gateway route, so nothing could reach it; fixed alongside E10.

### DeviceWeave — physical action surface
Utterance → scene/intent/device resolution (cosine + behaviour fusion, α step at 10 events) → policy engine (LLM-compiled rules, deterministic BLOCK/MODIFY/ALLOW) → six provider adapters (Kasa, Govee, SwitchBot, MyQ, Ring, Wyze); SMS ingress; vision intent source; Bedrock/Gemini/Ollama providers. **Exposes:** `/execute`, `/learn`, `/devices`, `/scenes`, `/presence`, `/learnings`, `/providers`, `/policies/author`, `/ingest`, `/health`. **Consumes:** Bedrock Converse, Memgraph (behaviour history), DynamoDB (registry, policies, conversations), Open-Meteo, mcp-observatory `==0.2.0`, shared stack. **Maturity:** 11.4k LOC — the largest — **0 tests**, last commit Jun 2026. **Weave role:** the modality where "no undo" makes the gate obviously necessary. **Gaps:** tests; the fusion threshold applied to a mixture whose meaning changes with α (D2/D3); pinned to a gate version that lacks the fail-open fix.

### TeamWeave — orchestration
JSON `team.json` (agents, workflow steps, schemas, RAG settings, tool hooks) executed on Step Functions by a worker that invokes Bedrock agents/models, validates structured output, runs pre/post tools, and — with `DPO_TRAINING_BUCKET` set — **invokes each step twice, keeps the lower-risk answer, and writes chosen/rejected pairs to S3**. **Exposes:** `/team/task`, `/team/task/{run_id}`, `/improve/*`, CRUD for `/agents /teams /roles /departments`, `/observability/metrics` (AMP), `/observability/agent-metrics` (shared DynamoDB spans), conversation handler; optional SIWE authorizer from AuthChain's stack. **Consumes:** Bedrock, pgvector RAG (its own `rag.py`), Gemini (research), mcp-observatory (vendored wrapper), shared stack. **Stores:** DynamoDB runs/tasks, S3 configs/artifacts/DPO pairs. **Maturity:** 12.3k LOC, 22 test files, Apr 2026; stack `tarun-content-team`. **Weave role:** the pipeline engine and the *only* reader of the shared telemetry table. **Gaps:** its own RAG and its own agent provisioning duplicate ContextWeave and DeployWeave.

### ToolWeave — REST action surface
FastMCP server: OpenAPI specs from S3 → catalogue in DynamoDB → `pre_tool` (Bedrock plans a call, field semantics from DataDictionary) → `post_tool` (GET executes; writes become proposals) → `commit_api_call` (token-verified execution). **Exposes:** MCP tools `pre_tool`, `post_tool`, `commit_api_call`, `reload_catalog`. **Consumes:** Bedrock Converse, DataDictionary MCP, mcp-observatory `>=0.2.1` (`ToolProposer`, wrapper API), shared metrics table via `SharedStackLookup`. **Maturity:** 2.8k LOC, 0 tests, Apr 2026. **Weave role:** the cleanest end-to-end demonstration of the thesis. **Gaps:** tests; `change-me-in-production` secret default.

### DataDictionary — field semantics
FastMCP over DynamoDB + Bedrock: propose/commit data elements (gated by mcp-observatory), get/search/list/by-context. **Exposes:** six MCP tools. **Consumes:** mcp-observatory from git `main`. **Maturity:** 0.6k LOC, 1 test. **Weave role:** ToolWeave's semantic lookup; a pattern for "governed writes to a registry." **Gaps:** no README; same secret default.

### CipherWeave — cryptographic policy
MCP tool `get_encryption_strategy`: risk graph in Memgraph → **Eq. 1/2 decayed evidence aggregation and Algorithm 1 bounded BFS (implemented in `scoring.py`, PR #28, merged Sep 11)** → profile lattice CHEAP→BALANCED→HARDENED→QUANTUM_SAFE with a compliance floor → drift detector (max-of-three statistic, EWMA baselines, cold-start policy) → hybrid X25519 + **real ML-KEM-768 via `liboqs-python`, fail-closed on a missing backend** (SEC-1 fixed) → HKDF per profile; JIT registration of unknown endpoints via Bedrock classification. **Consumes:** Memgraph, KMS, Bedrock, shared stack. **Maturity:** 3.5k LOC; 23 paper-property tests + suite + a 440-episode labelled corpus with a calibrated E2 harness; **nothing else in the portfolio calls it**. **Weave role:** policy for *how* a call is protected, sitting beside the gate that decides *whether* it runs. **Gaps:** `_zero_bytes` still uses `ctypes.memmove` on immutable bytes (SEC-6); no consumer.

### AuthChain — identity
SIWE (EIP-4361) nonce/verify → HS256 JWT → Lambda authorizer; plus a self-contained RAG (`/ingest`, `/chat`) on pgvector with Titan v1. **Exposes:** `/siwe/nonce`, `/siwe/verify`, `/siwe/me`, `/siwe/session`, `/ingest`, `/chat`, and an **exported authorizer ARN that TeamWeave's template can adopt as its default authorizer**. **Maturity:** 3.2k LOC, 1 test, Mar 2026. **Weave role:** the identity edge — the only repo that answers "who is calling". **Gaps:** its RAG is a third copy; `DATABASE_URL` not wired in the template.

### DeployWeave — model operations
FastMCP tools `model_selector` (latency/A-B aware choice from its own `deployweave-model-metrics` table), `team_provisioner`, `adapter_resolver` (LoRA catalogue), `agent_lifecycle`; a token-wallet **invocation gateway** (reserve → invoke → commit, 402 on exhaustion), reservation cleanup, orphan reconciliation, TTL-driven cleanup via Streams→SQS. **Maturity:** 2.6k LOC, 0 tests, Apr 2026; README names "MCP Client (Claude / TeamWeave)" as its caller but no repo calls it. **Weave role:** the model-lifecycle half of a flywheel that exists only in intent. **Gaps:** its metrics table duplicates the shared spans; its provisioner duplicates TeamWeave's `/agents`.

### TrainWeave — fine-tuning
One Lambda launches an EC2 Spot `g4dn.xlarge` that pulls a JSONL dataset and `train.py` from S3, runs LoRA, syncs checkpoints, uploads the adapter, self-terminates; S3 traffic via the shared VPC endpoint. **Maturity:** 0.6k LOC, 0 tests. **Weave role:** the middle of the flywheel — but it does not read TeamWeave's DPO bucket, and DeployWeave's adapter catalogue does not read its output. Both edges are documentation, not code.

### ScreenWeave — web action/observation surface
Playwright crawler on EC2 walks a site and persists every visual state to S3; a visual-QA worker classifies screenshots with local heuristics and escalates to Claude by tier. **Exposes:** MCP tools `crawl_url`, `get_session_status`, `get_screenshots`, `get_full_session`, `get_metrics`; `POST /visual-qa`. **Consumes:** Bedrock (tiered). **Maturity:** 3.1k LOC, 0 tests, Apr 2026; **no repo references it and it references none**. **Weave role:** an unused ingestion source — crawled artifacts are exactly what ContextWeave ingests — and an unused MCP surface the gate could govern.

### RoutineWeave — scheduling
Node/TypeScript: JSON tasks in S3 → registrar creates per-task EventBridge cron rules → scheduler renders the prompt, calls Gemini (structured by Nova), publishes to SNS. **Exposes:** `GET/POST /tasks`, `GET/PUT/DELETE /tasks/{name}`. **Consumes:** Gemini, Bedrock Nova, `@weaveaijs/mcp-observatory` (JS), shared metrics table. **Maturity:** 2.6k LOC, 9 test files, May 2026. **Weave role:** the time trigger the orchestration layer lacks. **Gaps:** off-stack model (Gemini) and language; its task JSON is a second prompt registry.

### ai-content-orchestrator — an application
Weekly LinkedIn-draft and newsletter pipeline: admin API, article lifecycle, EventBridge schedules, SES delivery. **Consumes:** **TeamWeave `POST /team/task`** (primary) with Gemini fallback, S3, DynamoDB; template references the SIWE authorizer. **Maturity:** 1.6k LOC, 0 tests, May 2026. **Weave role:** the one downstream consumer that proves TeamWeave is a platform.

### TaskWeave — superseded
517 lines of LangChain/LangGraph JSON-task framework, last touched Feb 2026; TeamWeave is its successor in every respect. **Weave role:** none; archive.

### IntentWeave, PromptWeave — empty
Names that describe capabilities the portfolio duplicates today (§3): intent resolution exists in DeviceWeave (`intent_parser`, `scene_catalog`, `llm_resolver`) and ToolWeave (`pre_tool`); prompt/task definitions exist in TeamWeave (`goal_template`), RoutineWeave (task JSON) and the content orchestrator.

---

## 2. What actually connects them (evidence)

After discarding hits that were only the shared account name, the connective tissue is four kinds of thing:

**(a) A control-plane library.** `mcp-observatory` is imported by ToolWeave, DataDictionary, DeviceWeave, TeamWeave, ContextWeave (Python) and RoutineWeave (JS, via the npm package). Six of sixteen.

**(b) A shared infrastructure stack — `tarun-teamweave-shared`.** VPC, private subnets, S3 gateway endpoint, and the `OBSERVATORY_METRICS` DynamoDB table. CipherWeave, ContextWeave, DeviceWeave, RoutineWeave, TeamWeave, ToolWeave and TrainWeave deploy into it (some via a `SharedStackLookup` custom resource, some via CI-time output lookups). Seven of sixteen. This was invisible to the dependency grep because it is a CloudFormation stack, not a repo.

**(c) A shared telemetry sink.** The spans that the gate emits land in one table, written by ToolWeave, TeamWeave, DeviceWeave, RoutineWeave and ContextWeave, and read by TeamWeave's `GET /observability/agent-metrics` and `GET /observability`, plus DeployWeave's model selector. The table is shared; the *item shape* is not. An audit (`mcp-observatory/docs/integration-audit.md`, F1-F2) found four incompatible partition-key schemes among those writers, with the readers querying only one of them: ToolWeave writes `PK`/`SK` into a table keyed on `pk`/`sk` so DynamoDB rejected every write, ScreenWeave writes a tool name where readers enumerate operation names, and the shared library exporter writes a `SPAN#` namespace nothing reads. `mcp-observatory/contracts/observatory_metrics_item.json` now pins the shape, and each repository holds its own writer or reader to it.

**(d) Four service-call edges.** ai-content-orchestrator → TeamWeave; ToolWeave → DataDictionary; TeamWeave → AuthChain's authorizer (optional parameter); DeployWeave ← "Claude / TeamWeave" (documented, not wired).

**(e) A brand namespace.** `@weaveaijs` on npm holds `mcp-observatory` 0.3.0 and **`tantu` 0.4.0, a React design system** — and every one of these sixteen repos carries a branch named `weave-unified-ui`. The unified console is evidently planned; nothing in the sixteen implements it yet.

```
                              tarun-teamweave-shared  (VPC · subnets · S3 endpoint · OBSERVATORY_METRICS)
                              ───────────────────────────────────────────────────────────────────────────
  AuthChain ──authorizer──► TeamWeave ◄──/team/task── ai-content-orchestrator
                               │ writes DPO pairs (S3)          · · · · ► TrainWeave · · · · ► DeployWeave      (intended, no code)
                               ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │                     mcp-observatory  (py + @weaveaijs js)            │
        │   ToolWeave ──► DataDictionary      DeviceWeave      RoutineWeave      ContextWeave   │
        └──────────────────────────────────────────────────────────────────────┘
                                          ▼ spans
                                  OBSERVATORY_METRICS ◄── read only by TeamWeave /agent-metrics

  CipherWeave (no callers)     ScreenWeave (no callers, no callees)     TaskWeave (superseded)     IntentWeave / PromptWeave (empty)
```

---

## 3. What is duplicated (what weaving removes)

| Capability | Copies | Where |
|---|---|---|
| RAG over pgvector | **3** | ContextWeave (graph + cache + router), TeamWeave `rag.py`, AuthChain `/ingest` `/chat` |
| Bedrock agent provisioning | **2** | TeamWeave `/agents` CRUD + Provision Lambda; DeployWeave `team_provisioner` / `agent_lifecycle` |
| Observatory wrapper singleton with a default secret | **4** | ToolWeave, DataDictionary (`change-me-in-production`), mcp-observatory itself (`dev-secret`, `dev-commit-secret`), TeamWeave vendored wrapper |
| Per-model metrics | **2** | DeployWeave `deployweave-model-metrics` (only it writes it); shared `OBSERVATORY_METRICS` (five writers) |
| Gemini client | **3** | TeamWeave `gemini.py`, RoutineWeave `GeminiClient.ts`, DeviceWeave `llm_provider/gemini.py` (+ content orchestrator) |
| Intent → structured action | **2** | DeviceWeave (scene → regex → cosine → LLM cascade); ToolWeave `pre_tool` (Bedrock tool-use loop) |
| "LLM compiles policy offline, deterministic evaluator enforces online" | **2** | DeviceWeave `policy_authoring` → `policy_engine`; CipherWeave `infer_policy_from_metadata` (JIT) → risk graph → profile |
| Prompt / task definitions as JSON | **3** | TeamWeave `team.json` goal templates, RoutineWeave tasks, content orchestrator prompts |
| Version pin of the gate | **5 different** | `>=0.1.0`, `==0.2.0`, `>=0.2.1`, `git+main`, npm `^0.3.0` |

The seventh row is the interesting one: two repositories independently arrived at the same architecture — a language model authors rules *out of band* and something deterministic enforces them *in band* — without naming it. That is the portfolio's second idea after the gate, and it is currently unnamed and unshared.

---

## 4. The weave — target shape and the edges to add

### 4.1 Layers

```
 L6  Applications          ai-content-orchestrator · RoutineWeave tasks · (the unified console, tantu)
 L5  Orchestration         TeamWeave (pipelines)  ·  RoutineWeave (time triggers)
 L4  Action surfaces       ToolWeave (REST) · DeviceWeave (physical) · ScreenWeave (web) · DeployWeave (models)
 L3  Knowledge             ContextWeave (GraphRAG + cache + learning router) · DataDictionary (field semantics)
 L2  Control plane         mcp-observatory (whether a call runs; what happened) · CipherWeave (how it is protected)
 L1  Identity              AuthChain (who is calling)
 L0  Substrate             tarun-teamweave-shared (network, telemetry table) · Bedrock · Memgraph · pgvector
      ── flywheel ──       TeamWeave DPO pairs → TrainWeave LoRA → DeployWeave adapters → TeamWeave model_aliases
```

Two things this shape asserts that the code does not yet: every L4 surface goes through L2 (ScreenWeave does not; DeployWeave's gateway does not), and every L2/L4 component can consult L3 (none does).

### 4.2 Edges to add, ranked by value per effort

| # | Edge | Why | Effort |
|---|---|---|---|
| **E1** | **Release `mcp-observatory` 0.3.0 and pin every consumer to it; make the JS package track the same version** | Five consumers run a gate whose fail-open and token-malleability are fixed only on a branch; DeviceWeave `==0.2.0` will never get them | days |
| **E2** | **One `observatory_wrapper` module inside `mcp-observatory` (`mcp_observatory.aws`) replacing the four vendored copies; fail closed on any default secret** | Four copies, four secret defaults, four drift points | days |
| **E3** | **TeamWeave → ContextWeave for RAG; delete `rag.py`; AuthChain drops `/ingest` `/chat`** | Three RAGs become one, and the one that learns | 1 week |
| **E4** | **TeamWeave `/agents` delegates to DeployWeave `team_provisioner`/`agent_lifecycle`** (or DeployWeave's tools are folded into TeamWeave) | Two provisioners of the same Bedrock agents | 1 week |
| **E5** | **DeployWeave `model_selector` reads the shared `OBSERVATORY_METRICS` spans instead of its own table** | The shared table already has latency, tokens, cost and risk per model across five products; DeployWeave's table has only what DeployWeave wrote | days |
| **E6** | **Close the flywheel in code:** TrainWeave takes `s3://{DPO_TRAINING_BUCKET}/{team}/{step}/…` as its dataset source; on adapter upload it registers the adapter in DeployWeave's catalogue; `adapter_resolver` output feeds `team.json` `model_aliases` | The three repos describe one loop and implement three disconnected steps | 2 weeks |
| **E7** | **ScreenWeave crawl artifacts → ContextWeave `raw/` prefix; ScreenWeave MCP tools wrapped by the gate** | An unused ingestion source and an ungoverned action surface, both one adapter away | 1 week |
| **E8** | **CipherWeave consulted at token issue time**: the gate's commit token carries the cipher profile the call must use; ToolWeave's executor honours it | Gives CipherWeave its first caller, and the gate a second dimension ("this call runs *and* over this channel strength") | 1–2 weeks |
| **E9** | **AuthChain's authorizer as the default on every HTTP API** (TeamWeave already supports it as a parameter) | One identity layer instead of none on most surfaces | days |
| **E10** | **A single `/observability` service reading `OBSERVATORY_METRICS` + `routing_decisions` + ContextWeave `/health`**, consumed by the console | Today one TeamWeave endpoint reads the shared table; the router's health verdict and human ratings are elsewhere | 1–2 weeks |

### 4.3 Name the second idea

DeviceWeave and CipherWeave both do *offline LLM policy compilation, online deterministic enforcement*. Extract the pattern as a small library — a compiled-rule schema, a validator, a versioned store, and an evaluator with BLOCK/MODIFY/ALLOW precedence — and have both consume it. This is the natural content for **IntentWeave**'s empty shell *if* the name is stretched, but the honest name is `policy-compiler`; the recommendation is to archive the empty repos rather than invent scope to fill their names. Likewise **PromptWeave** would be the versioned registry for TeamWeave goal templates and RoutineWeave tasks; that is a real need, but a `prompts/` module in TeamWeave with versioning solves it without a sixteenth repository.

### 4.4 Consolidation

| Action | Repos |
|---|---|
| Archive | TaskWeave (superseded), IntentWeave, PromptWeave (empty) |
| Mark as *applications*, not platform | ai-content-orchestrator, RoutineWeave, ScreenWeave |
| Fold or delegate | AuthChain RAG → ContextWeave; TeamWeave RAG → ContextWeave; DeployWeave provisioning ↔ TeamWeave (one owner) |

After consolidation the platform is **nine repos** — mcp-observatory, CipherWeave, AuthChain, ContextWeave, DataDictionary, ToolWeave, DeviceWeave, DeployWeave+TrainWeave, TeamWeave — plus applications, plus the console.

---

## 5. The selling point, restated with the whole portfolio in view

> Every side effect an AI agent takes on AWS — a REST call, a device actuation, a model invocation, a key derivation, a browser session — passes through one gate that binds authorisation to the exact action, scores it on whatever evidence exists, refuses critical actions on too little, chooses the channel strength it must run over, records what happened in one table, and feeds a knowledge layer that learns which strategies people accept.

What the whole-portfolio read adds to that sentence: **the data plane shares one table but not one item shape** (five writers, four incompatible key schemes, two readers that see only one of them -- see `mcp-observatory/docs/integration-audit.md`), and the reading of it is not unified either; **the identity layer exists** and is adopted by one API; **a flywheel is designed** (preference pairs → LoRA → adapters → aliases) and disconnected; and **a second architectural idea recurs unnamed** (offline policy compilation). The market position — governed execution for agents, open-source gate as the wedge, AWS reference architecture second — stands; E1–E2 are its prerequisites, and E3–E6 are what make "platform" a true description.

---

## 6. Research-worthy, revised

| # | Concept | Readiness |
|---|---|---|
| R1 | Confidence-feedback router that silently stopped learning (ContextWeave) | publish now |
| R2 | Argument-bound execution capabilities; properties proved, two found false as shipped (mcp-observatory) | publish now |
| R3 | Offline LLM policy compilation / online deterministic enforcement — **now with two independent instances** (DeviceWeave, CipherWeave), which makes it a pattern paper rather than a system description | position paper now; empirical after DeviceWeave's E2 ablation |
| R4 | Confidence semantics for composed agent systems | position paper now |
| R5 | Does self-confidence predict human ratings in adaptive RAG | after ≥300 rated answers |
| **R6** | **Per-flow post-quantum migration urgency from a risk graph, with a calibrated drift detector on a labelled synthetic corpus** (CipherWeave) | **publishable now as a simulation study** — the paper reports E2 at a 1% FPR threshold calibrated on held-out benign traffic and states plainly that the corpus is synthetic; the residual limitation is external validity, not absence of mechanism |
| R7 | *New:* the preference-pair flywheel — risk-scored dual invocation as a free DPO data source (TeamWeave) | after E6 wires it and one adapter is trained on it |

---

---
*Maintained in the ContextWeave repository because ContextWeave is the platform's knowledge layer and this document is ingested into it. Update it when an integration edge in §4.2 lands.*