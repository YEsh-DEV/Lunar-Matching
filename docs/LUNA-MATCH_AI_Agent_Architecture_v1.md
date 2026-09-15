# LUNA-MATCH — AI Explanation & Research Agent Architecture (v1)
### Separate service. Consumes the Core Backend's `/summary` contract as its only coupling point. Never mixed into core engine code.

---

## 0. Scope & Two Distinct Chat Surfaces

This system has **two separate agent surfaces**, different in purpose, context, and lifecycle. They share the same underlying RAG knowledge base and inference backend, but are never the same session and never share conversation memory.

| | **Surface 1: Result Explainer** | **Surface 2: Research Companion** |
|---|---|---|
| Trigger | Automatically after a registration job reaches `DONE` | User opens it manually, anytime, independent of any job |
| Context | Bound to one specific `job_id`; receives that job's full `/summary` payload as grounding context | No job context at all — general knowledge only |
| Session lifetime | Tied to that job; can be revisited while the job's data exists | Fresh session each time it's opened; no memory of any prior job or prior research session |
| Knowledge sources | This job's metrics/graphs/craters + general RAG corpus | General RAG corpus only |
| Example question | "Why is the inlier ratio only 78%?" | "What's the difference between OHRC and TMC-2?" |

---

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Core Backend (separate service, v3 architecture doc)          │
│  Produces GET /jobs/{id}/summary   (schema v1.2, JSON)          │
└───────────────────────────┬──────────────────────────────────────┘
                            │  (fetched once, immediately after DONE,
                            │   NOT re-fetched per chat message)
┌───────────────────────────▼──────────────────────────────────────┐
│  AGENT SERVICE  (this document, independent FastAPI service)     │
│                                                                    │
│  ┌────────────────────┐        ┌──────────────────────────┐      │
│  │ Context Assembler   │        │ Research Companion        │      │
│  │ (per-job, one-time) │        │ (stateless per session)   │      │
│  └──────────┬──────────┘        └─────────────┬────────────┘      │
│             │                                   │                  │
│  ┌──────────▼──────────────────────────────────▼────────────┐    │
│  │            Query Router (fast-path vs generative-path)     │    │
│  └──────────┬───────────────────────────────────┬────────────┘    │
│             │                                    │                 │
│    ┌────────▼────────┐               ┌───────────▼───────────┐    │
│    │ FAST PATH        │               │ GENERATIVE PATH        │    │
│    │ Deterministic    │               │ RAG retrieval +         │    │
│    │ template answer, │               │ compact fast LLM,       │    │
│    │ 0 LLM calls      │               │ streamed response       │    │
│    │ <50ms            │               │ target: first token<1s  │    │
│    └──────────────────┘               └────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Context Handoff Contract

The Agent Service never reformats or reinterprets backend numbers — it consumes the exact `/summary` v1.2 payload (see Core Backend Architecture v3, Section 4.2) verbatim as `AgentContextBundle.registration_data`. This is the single coupling point between the two systems.

```python
class AgentContextBundle(BaseModel):
    job_id: str
    registration_data: SummaryResponse   # verbatim, imported type, never re-derived
    fetched_at: datetime
    context_version: Literal["1.0"]
```

Assembly timing: fetched **once**, immediately when the job transitions to `DONE` (webhook or short poll from the Agent Service), cached in the Agent Service's own store keyed by `job_id`. Every subsequent chat message against that job reuses the cached bundle — the backend is never re-queried per message, which is both a latency and a coupling-safety requirement.

---

## 3. RAG Knowledge Base

### 3.1 Corpus sources (sensor/topic-tagged markdown, already assembled)
```
knowledge/
├── isro/chandrayaan2_mission.md
├── ohrc/ohrc_specifications.md
├── tmc2/tmc2_specifications.md
├── iirs/iirs_specifications.md
├── lro/lroc_nac_reference.md
├── registration/image_registration_metrics.md    # RMSE/SSIM/inlier-ratio definitions, in plain language
└── craters/lunar_crater_science.md                # NEW: crater size-frequency dating, morphology basics
```

### 3.2 Ingestion pipeline
```python
def chunk_document(path: str, max_tokens: int = 300, overlap: int = 50) -> list[Chunk]: ...
def embed_chunks(chunks: list[Chunk], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray: ...
def build_index(embeddings: np.ndarray) -> FaissIndex: ...
```
Each `Chunk` carries metadata: `{document_name, sensor, mission, section_title}` — enables filtered retrieval (e.g. restrict to `sensor=OHRC` when the active job's `img_a` is OHRC-derived).

### 3.3 Retrieval
```python
def retrieve(query: str, top_k: int = 4, score_threshold: float = 0.35,
             sensor_filter: str | None = None) -> list[tuple[Chunk, float]]: ...
def format_context_for_prompt(results: list[tuple[Chunk, float]]) -> str: ...
```
`top_k` capped at 4 deliberately — larger contexts increase LLM prefill latency for no proportional gain in answer quality at this corpus size.

---

## 4. Latency Architecture — Target: <1s for the common case

This is the central design constraint. Three mechanisms, applied in order:

### 4.1 Fast path (no LLM call at all — target <50ms)
A query classifier (cheap keyword/regex match, not an LLM call) checks if the question maps directly to a field already present in `AgentContextBundle.registration_data`. Examples: "what's the RMSE", "how many inliers", "is this reliable", "what grade did this get". These are answered by **template substitution directly from the JSON** — the exact same `quality_assessment.reasoning` and metric values already computed by the deterministic backend layer. Zero network calls, zero model inference.

```python
def try_fast_path(query: str, context: AgentContextBundle) -> ChatResponse | None:
    """Returns a populated ChatResponse if query matches a known metric-question
    pattern, else None (falls through to generative path)."""
```

### 4.2 Generative path (RAG + compact fast-inference LLM — target: first token <1s, full answer <3-4s)
Only invoked when the fast path returns `None`. Requirements to hit sub-1s first-token:
- Use a **low-latency inference provider** (e.g. Groq or an equivalent fast-inference API) — this is a hard requirement, not a preference; standard OpenAI-class latency (2-5s to first token) will not meet the target.
- Cap `max_tokens` to ~400-600 for explanations; free-form research questions may allow more, but still capped.
- Retrieval (`top_k=4`) happens in parallel with context-bundle lookup, not sequentially — both are cheap local operations (~10-50ms), never the bottleneck.
- **Stream the response token-by-token (SSE)** — even if total generation takes 3-4s, perceived latency is governed by first-token time, which is the actual <1s target, not total completion time. This distinction must be reflected in both the API contract (Section 5) and the frontend's rendering (progressive display, not spinner-then-dump).

### 4.3 Prefetch / warm cache
Immediately when a job reaches `DONE` (before any user asks anything), the Agent Service pre-runs RAG retrieval for a small set of anticipated questions ("explain these results", "is this reliable") and caches the result. If the user's first message matches one of these, the generative path skips retrieval entirely and goes straight to generation — shaving another retrieval round-trip off the critical path.

---

## 5. API Specification

### 5.1 Result Explainer (job-bound)
```
POST /agent/explain/{job_id}/start
  → registers the AgentContextBundle for this job (idempotent; safe to call
    multiple times, only fetches from backend once)
  Response: 202 { "job_id": str, "context_ready": bool }

POST /agent/explain/{job_id}/message
  Request: { "query": str, "stream": bool = true }
  Response (stream=true): text/event-stream, SSE frames:
    event: token   data: {"text": str}
    event: done    data: {"used_fast_path": bool, "latency_ms": float,
                           "sources": [{"document","section","score"}]}
    event: error   data: {"error_code": str, "message": str}
  Response (stream=false): 200
    { "text_response": str, "used_fast_path": bool, "latency_ms": float,
      "sources": [...] }
```

### 5.2 Research Companion (job-independent)
```
POST /research/session
  Response: { "session_id": str }        # fresh, no prior memory

POST /research/{session_id}/message
  Request: { "query": str, "stream": bool = true }
  Response: same shape as 5.1 (no job context, no "used_fast_path" — always
  generative path, since there is no structured metrics data to fast-path against)
```

### 5.3 Common response schema (non-streaming form, for reference)
```python
class ChatResponse(BaseModel):
    text_response: str
    used_fast_path: bool
    latency_ms: float
    sources: list[SourceRef]            # empty list if fast_path or no RAG hit
    error_code: str | None = None

class SourceRef(BaseModel):
    document: str
    section: str
    score: float
```

---

## 6. Session & State Model

- **Result Explainer session**: keyed by `job_id`. State = `{AgentContextBundle, message_history: list[Message]}`. Persists as long as the underlying job's data exists on the backend; no independent TTL beyond that.
- **Research Companion session**: keyed by a generated `session_id`, in-memory only, no persistence requirement, no cross-session memory — genuinely fresh every time, exactly as specified. TTL: expire after a short idle period (e.g. 30 min) to bound memory use.
- Message history for either surface is stored server-side only for the duration needed to maintain conversational context within that session — it is not a permanent chat log store; if durable history is wanted later, that is an explicit future extension, not implied by this spec.

---

## 7. Reliability & Degradation

- If the RAG index or LLM provider is unavailable, the **fast path must still work** — this is the whole point of separating it out. A user asking "what's the RMSE" gets an instant, correct, deterministic answer even if the entire generative layer is down.
- If the generative path times out (hard timeout, e.g. 8s total), return `error_code="GENERATION_TIMEOUT"` with the deterministic `quality_assessment.reasoning` from the context bundle as a fallback body — never leave the user with nothing.
- The Agent Service must never be a dependency of the Core Backend in either direction beyond the one-time `/summary` fetch — if the Agent Service is entirely down, registration jobs must complete normally and `/summary` must still be fully usable standalone.

---

## 8. Non-Goals
- No fine-tuning or training of any model — retrieval-augmented generation over a fixed, curated corpus only.
- No cross-job memory (the Result Explainer for job A must never leak context into job B's session).
- No unbounded conversation history growth — cap stored turns (e.g. last 10 messages) per session to keep prompt size, and therefore latency, bounded and predictable.
- No agent-initiated actions on the Core Backend (no re-running jobs, no parameter changes) — this layer is read-only and explanatory by design.
