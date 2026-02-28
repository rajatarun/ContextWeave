import { BedrockRuntimeClient, InvokeModelCommand, ApplyGuardrailCommand } from "@aws-sdk/client-bedrock-runtime";
import { S3Client, ListObjectsV2Command, GetObjectCommand } from "@aws-sdk/client-s3";
import { NodeHttpHandler } from "@smithy/node-http-handler";
import pg from "pg";
import fs from "fs";
import crypto from "crypto";

const { Pool } = pg;

const {
  DATABASE_URL,
  BEDROCK_MODEL_ID,
  BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v1",
  EMBED_DIM = "1536",
  EMBED_NORMALIZE = "true",
  DEFAULT_TOP_K = "6",
  MAX_CONTEXT_CHARS = "12000",
  CHUNK_SIZE = "1200",
  CHUNK_OVERLAP = "200",
  MAX_INGEST_FILES = "50"
} = process.env;

const GUARDRAIL_ID = process.env.GUARDRAIL_ID || "5o1zncxdyabz";
const GUARDRAIL_VERSION = process.env.GUARDRAIL_VERSION || "1";

function jlog(obj) { try { console.log(JSON.stringify(obj)); } catch { console.log(String(obj)); } }

function httpRes(statusCode, body) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "content-type,authorization,x-api-key",
      "access-control-allow-methods": "OPTIONS,POST"
    },
    body: JSON.stringify(body)
  };
}

function getMethod(event) { return event?.requestContext?.http?.method || event?.httpMethod || "UNKNOWN"; }
function getPath(event) { return event?.rawPath || event?.requestContext?.http?.path || event?.path || "/"; }
function stripStage(path) { return (path || "/").replace(/^\/(prod|\$default)(?=\/)/, ""); }

async function streamToBuffer(stream) {
  const chunks = [];
  for await (const c of stream) chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c));
  return Buffer.concat(chunks);
}

function sha256(s) { return crypto.createHash("sha256").update(s).digest("hex"); }
function extOfKey(key) { const i = key.lastIndexOf("."); return i >= 0 ? key.slice(i + 1).toLowerCase() : ""; }

function chunkText(text, size, overlap) {
  const clean = (text || "").replace(/\r\n/g, "\n").replace(/\u0000/g, "");
  const out = [];
  let i = 0;
  while (i < clean.length) {
    const end = Math.min(clean.length, i + size);
    const slice = clean.slice(i, end).trim();
    if (slice) out.push(slice);
    i = Math.max(0, end - overlap);
    if (end === clean.length) break;
  }
  return out;
}

function pgVectorLiteral(arr) { return "[" + arr.map(x => (Number.isFinite(x) ? x : 0)).join(",") + "]"; }

function parseEmbedDims(modelId, dims) {
  if (modelId.includes("titan-embed-text-v1")) return 1536;
  if (modelId.includes("titan-embed-text-v2")) {
    if (![256, 512, 1024].includes(dims)) throw new Error("Titan v2 supports dimensions 256, 512, 1024 only.");
    return dims;
  }
  return dims;
}

const embedDims = parseEmbedDims(String(BEDROCK_EMBED_MODEL_ID), Number(EMBED_DIM));
const normalize = String(EMBED_NORMALIZE).toLowerCase() !== "false";

const bedrock = new BedrockRuntimeClient({
  requestHandler: new NodeHttpHandler({ connectionTimeout: 10000, socketTimeout: 90000 })
});

const s3 = new S3Client({});

const caPath = "/var/task/rds-ca-bundle.pem";
const ca = fs.existsSync(caPath) ? fs.readFileSync(caPath, "utf8") : undefined;

if (!DATABASE_URL) throw new Error("DATABASE_URL is required");
if (!BEDROCK_MODEL_ID) throw new Error("BEDROCK_MODEL_ID is required");
if (!BEDROCK_EMBED_MODEL_ID) throw new Error("BEDROCK_EMBED_MODEL_ID is required");

const dbUrl = new URL(DATABASE_URL);

const pool = new Pool({
  host: dbUrl.hostname,
  port: Number(dbUrl.port || "5432"),
  user: decodeURIComponent(dbUrl.username),
  password: decodeURIComponent(dbUrl.password),
  database: (dbUrl.pathname || "/postgres").replace("/", "") || "postgres",
  ssl: ca ? { ca, rejectUnauthorized: true } : undefined,
  connectionTimeoutMillis: 10000
});

async function applyGuardrail(source, text) {
  const cmd = new ApplyGuardrailCommand({
    guardrailIdentifier: GUARDRAIL_ID,
    guardrailVersion: GUARDRAIL_VERSION,
    source,
    content: [{ text: { text } }]
  });
  const r = await bedrock.send(cmd);
  const action = r?.action;
  const outputs = r?.outputs || [];
  const outText = outputs?.[0]?.text?.text;
  const blocked = action && String(action).toUpperCase() === "BLOCKED";
  return { blocked: !!blocked, text: typeof outText === "string" && outText.length ? outText : text, action: action || null };
}

async function embedText(text) {
  const modelId = String(BEDROCK_EMBED_MODEL_ID);
  const payload = modelId.includes("titan-embed-text-v2") ? { inputText: text, dimensions: embedDims, normalize } : { inputText: text };

  const cmd = new InvokeModelCommand({
    modelId,
    contentType: "application/json",
    accept: "application/json",
    body: new TextEncoder().encode(JSON.stringify(payload))
  });

  const r = await bedrock.send(cmd);
  const parsed = JSON.parse(new TextDecoder().decode(r.body));
  const v = parsed?.embedding || parsed?.embeddingsByType?.float || parsed?.embeddingsByType?.["float"];
  if (!Array.isArray(v)) throw new Error("Embedding response missing embedding vector");
  if (v.length !== embedDims) throw new Error(`Embedding dim mismatch. Got ${v.length}, expected ${embedDims}.`);
  return v;
}

async function genAnswer(question, context) {
  const modelId = String(BEDROCK_MODEL_ID);
  const isClaude = modelId.includes("anthropic.") || modelId.toLowerCase().includes("claude");

  const prompt =
`You are a recruiter-facing assistant.
You MUST answer ONLY using the Context.
If Context does not contain the answer, reply exactly: "I don't know based on the provided documents."

Context:
${context}

Question:
${question}

Answer (cite document titles when relevant):`;

  const payload = isClaude ? {
    anthropic_version: "bedrock-2023-05-31",
    max_tokens: 600,
    temperature: 0.2,
    messages: [{ role: "user", content: prompt }]
  } : {
    max_tokens: 600,
    temperature: 0.2,
    messages: [{ role: "user", content: prompt }]
  };

  const cmd = new InvokeModelCommand({
    modelId,
    contentType: "application/json",
    accept: "application/json",
    body: new TextEncoder().encode(JSON.stringify(payload))
  });

  const r = await bedrock.send(cmd);
  const parsed = JSON.parse(new TextDecoder().decode(r.body));
  const a1 = parsed?.content?.map(p => p?.text).filter(Boolean).join("");
  const a2 = parsed?.output?.message?.content?.map(p => p?.text).filter(Boolean).join("");
  const a3 = parsed?.generation || parsed?.generated_text || parsed?.text || parsed?.outputText;
  const answer = (a1 || a2 || a3 || "").trim();
  if (!answer) throw new Error("Model returned empty response");
  return answer;
}

async function ensureDoc(client, { doc_id, title, doc_type, source, source_uri, tags }) {
  await client.query(
    `INSERT INTO rag_documents (doc_id, title, doc_type, source, source_uri, tags)
     VALUES ($1,$2,$3,$4,$5,$6)
     ON CONFLICT (doc_id)
     DO UPDATE SET title=EXCLUDED.title, doc_type=EXCLUDED.doc_type, source=EXCLUDED.source,
                   source_uri=EXCLUDED.source_uri, tags=EXCLUDED.tags, updated_at=now()`,
    [doc_id, title, doc_type, source, source_uri, tags || []]
  );
}

async function upsertChunk(client, { doc_id, chunk_id, title, section, content, metadata, embedding }) {
  const emb = pgVectorLiteral(embedding);
  await client.query(
    `INSERT INTO rag_chunks (doc_id, chunk_id, title, section, content, metadata, embedding)
     VALUES ($1,$2,$3,$4,$5,$6,$7::vector)
     ON CONFLICT (doc_id, chunk_id)
     DO UPDATE SET title=EXCLUDED.title, section=EXCLUDED.section, content=EXCLUDED.content,
                   metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding`,
    [doc_id, chunk_id, title || null, section || null, content, metadata || {}, emb]
  );
}

async function extractTextFromS3({ bucket, key }) {
  const obj = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
  const buf = await streamToBuffer(obj.Body);
  const ext = extOfKey(key);

  let text = "";
  if (ext === "txt" || ext === "md" || ext === "csv" || ext === "json") {
    text = buf.toString("utf8");
  } else if (ext === "pdf") {
    const pdfParseMod = await import("pdf-parse");
    const pdfParse = pdfParseMod.default || pdfParseMod;
    const parsed = await pdfParse(buf);
    text = parsed?.text || "";
  } else if (ext === "docx") {
    const mammoth = await import("mammoth");
    const out = await mammoth.extractRawText({ buffer: buf });
    text = out?.value || "";
  } else {
    text = buf.toString("utf8");
  }

  return { text, bytes: buf.length, ext };
}

async function handleIngest(payload, requestId) {
  const bucket = payload?.bucket;
  const prefix = payload?.prefix || "";
  const docType = payload?.docType || "doc";
  const tags = Array.isArray(payload?.tags) ? payload.tags : [];
  const maxFiles = Math.max(1, Math.min(Number(payload?.maxFiles || MAX_INGEST_FILES), 500));

  if (!bucket) return httpRes(400, { error: "Missing bucket" });

  const tList = Date.now();
  const listed = await s3.send(new ListObjectsV2Command({ Bucket: bucket, Prefix: prefix }));
  const listMs = Date.now() - tList;

  const supportedExt = new Set(["pdf", "docx", "txt", "md", "json", "csv"]);

  const skipped = (listed.Contents || [])
    .filter(o => o && o.Key)
    .filter(o => o.Key.endsWith("/") || (o.Size ?? 0) === 0 || !supportedExt.has(extOfKey(o.Key)))
    .map(o => ({ key: o.Key, size: o.Size ?? 0 }))
    .slice(0, 50);

  const keys = (listed.Contents || [])
    .filter(o => o && o.Key)
    .filter(o => !o.Key.endsWith("/"))
    .filter(o => (o.Size ?? 0) > 0)
    .filter(o => supportedExt.has(extOfKey(o.Key)))
    .map(o => o.Key)
    .slice(0, maxFiles);

  jlog({ at: "ingest_list", requestId, bucket, prefix, filesFound: keys.length, listMs });
  jlog({ at: "ingest_skip_markers", requestId, skippedCount: skipped.length, sample: skipped });

  if (keys.length === 0) return httpRes(200, { requestId, bucket, prefix, filesFound: 0, ingested: 0, listMs, errors: [] });

  const client = await pool.connect();
  const tDb = Date.now();
  let ingested = 0;
  const errors = [];

  try {
    await client.query("BEGIN");

    for (const key of keys) {
      const sourceUri = `s3://${bucket}/${key}`;
      const docId = sha256(sourceUri);
      const title = key.split("/").pop() || key;

      let extracted;
      try { extracted = await extractTextFromS3({ bucket, key }); }
      catch (e) { errors.push({ key, step: "get/extract", error: e.message }); continue; }

      const rawText = (extracted.text || "").trim();
      jlog({ at: "ingest_extract", requestId, key, bytes: extracted.bytes, ext: extracted.ext, textChars: rawText.length });

      if (!rawText) { errors.push({ key, step: "extract", error: "empty_text" }); continue; }

      await ensureDoc(client, { doc_id: docId, title, doc_type: docType, source: "s3", source_uri: sourceUri, tags });

      const chunks = chunkText(rawText, Number(CHUNK_SIZE), Number(CHUNK_OVERLAP));
      jlog({ at: "ingest_chunk", requestId, key, chunks: chunks.length });

      let chunkIndex = 0;

      for (const c of chunks) {
        const chunkId = `${docId}:${chunkIndex}`;
        const meta = { bucket, key, ext: extracted.ext, bytes: extracted.bytes, embedModel: String(BEDROCK_EMBED_MODEL_ID), embedDims };

        let emb;
        try { emb = await embedText(c); }
        catch (e) { errors.push({ key, chunkIndex, step: "embed", error: e.message }); break; }

        await upsertChunk(client, { doc_id: docId, chunk_id: chunkId, title, section: null, content: c, metadata: meta, embedding: emb });

        chunkIndex += 1;
      }

      ingested += 1;
    }

    await client.query("COMMIT");
  } catch (e) {
    try { await client.query("ROLLBACK"); } catch {}
    throw e;
  } finally {
    client.release();
  }

  const dbMs = Date.now() - tDb;
  return httpRes(200, { requestId, ingested, listMs, dbMs, errors });
}

async function handleChat(payload, requestId) {
  const rawQuestion = (payload?.question || "").trim();
  if (!rawQuestion) return httpRes(400, { error: "Missing question" });

  const topK = Math.max(1, Math.min(20, Number(payload?.topK || DEFAULT_TOP_K)));
  const maxChars = Math.max(1000, Math.min(50000, Number(payload?.maxContextChars || MAX_CONTEXT_CHARS)));

  const grIn = await applyGuardrail("INPUT", rawQuestion);
  if (grIn.blocked) return httpRes(400, { error: "guardrail_blocked_input", message: grIn.text, requestId });

  const question = grIn.text;

  const qEmb = await embedText(question);
  const qvecStr = pgVectorLiteral(qEmb);

  jlog({ at: "qvec_preview", requestId, qvecLen: qvecStr.length, preview: qvecStr.slice(0, 80) });

  const client = await pool.connect();
  let rows = [];
  try {
    const sql = `
      SELECT doc_id, chunk_id, title, content,
             (1 - (embedding <=> $1::vector)) AS score
      FROM rag_chunks
      WHERE embedding IS NOT NULL
      ORDER BY embedding <=> $1::vector
      LIMIT $2
    `;

    const r = await client.query(sql, [qvecStr, topK]);
    rows = r.rows || [];
  } finally {
    client.release();
  }

  let context = "";
  const citations = [];
  for (const r of rows) {
    if (!r?.content) continue;
    if (context.length + r.content.length + 10 > maxChars) break;
    context += (context ? "\n\n---\n\n" : "") + r.content;
    citations.push({ doc_id: r.doc_id, chunk_id: r.chunk_id, title: r.title || null, score: Number(r.score) });
  }

  const rawAnswer = await genAnswer(question, context);
  const grOut = await applyGuardrail("OUTPUT", rawAnswer);

  return httpRes(200, { requestId, answer: grOut.text, citations: grOut.blocked ? [] : citations });
}

export const handler = async (event) => {
  const requestId = event?.requestContext?.requestId || crypto.randomUUID();

  const method = getMethod(event);
  const rawPath = getPath(event);
  const path = stripStage(rawPath);

  const isS3Event = Array.isArray(event?.Records) && event.Records?.[0]?.eventSource === "aws:s3";

  if (isS3Event) {
    const rec = event.Records[0];
    const bucket = rec?.s3?.bucket?.name;
    const key = decodeURIComponent(rec?.s3?.object?.key || "").replace(/\+/g, " ");
    if (key.endsWith("/")) return httpRes(200, { requestId, skipped: true, reason: "folder_marker", key });
    const prefix = key.includes("/") ? key.split("/").slice(0, -1).join("/") + "/" : "";
    return await handleIngest({ bucket, prefix }, requestId);
  }

  if (method === "OPTIONS") return httpRes(204, {});

  let body = {};
  try { body = event?.body ? JSON.parse(event.body) : {}; } catch {}

  if (path === "/ingest") return await handleIngest(body, requestId);
  if (path === "/chat") return await handleChat(body, requestId);

  return httpRes(404, { error: "Not found", path, requestId });
};
