# magicpin AI Challenge — Vera Merchant Assistant Strategy

**Document Status**: Authoritative Strategy & Implementation Tracker  
**Last Updated**: 2026-09-27  
**Target Score**: 45–50 / 50 (Zero Operational Penalties + High-Grounded Composition)

---

## 1. Executive Summary & Objective

magicpin's **Vera** is an AI assistant communicating with local merchants (dentists, salons, gyms, restaurants, pharmacies) over WhatsApp to grow their business via Google Business Profile (GBP) management, campaign management, offer recommendations, and customer retention.

### The Problem with Today's Vera
- Sends generic, discount-heavy offers (*"10% off"*).
- Trapped in qualification loops (asking more questions when the merchant said *"I want to join"*).
- Gets caught in WhatsApp Business automated auto-reply loops (*"Thank you for contacting us..."*).
- Lacks conversation diversity (mostly reactive reminders rather than curiosity/knowledge-driven nudges).

### Our Core Solution
Build a high-precision, stateful AI assistant powered by a **4-Context Composition Engine** and an **Operational Guardrail Layer**:
- **Personalized & Grounded**: Every fact, number, offer, and citation traces back directly to the provided context. Zero hallucinations.
- **Category-Attuned Voice**: Matches clinical peer tone for dentists, energetic coaching for gyms, operator-to-operator for restaurants, precise counsel for pharmacies, and approachable expert for salons.
- **Intent-Driven State Machine**: Instantly switches to execution mode when merchants commit, detects and gracefully exits auto-replies, and de-escalates hostility.
- **Zero-Tolerance Operational Discipline**: Eliminates URLs from message bodies, adheres to single CTAs in the final sentence, enforces anti-repetition, and responds well within the 30-second budget.

---

## 2. The 4-Context Architectural Framework

Every message composed by Vera follows the contract:
$$\text{compose}(\text{category}, \text{merchant}, \text{trigger}, \text{customer?}) \longrightarrow \text{message}$$

```
                    ┌─────────────────────────────────────────┐
Judge ── HTTP ─────►│  FastAPI Application (bot.py)           │
      ◄─────────────│                                          │
                    │  ┌────────────┐  ┌────────────────────┐ │
                    │  │ContextStore│  │ ConversationStore  │ │
                    │  │(in-memory  │  │ (per conv_id turns,│ │
                    │  │ scope+id+v)│  │  state & sent_set) │ │
                    │  └──────┬─────┘  └─────────┬──────────┘ │
                    │         │                  │            │
                    │         ▼                  ▼            │
                    │  ┌──────────────────────────────────┐   │
                    │  │  Router & Fact Pre-Extractor     │   │
                    │  │  (pre-pull 3-5 grounded facts)   │   │
                    │  └──────────────────┬───────────────┘   │
                    │                     │                   │
                    │                     ▼                   │
                    │  ┌──────────────────────────────────┐   │
                    │  │  LLM / Grounded Fallback Composer│   │
                    │  │  (temp=0, category tone, CTA)    │   │
                    │  └──────────────────┬───────────────┘   │
                    │                     │                   │
                    │                     ▼                   │
                    │  ┌──────────────────────────────────┐   │
                    │  │  Post-Composition Validator      │   │
                    │  │  (no URLs, single CTA, no jargon)│   │
                    │  └──────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
```

1. **CategoryContext** (`category`): Slow-changing vertical pack. Sets voice profile, allowed vocabulary, taboo phrases, benchmark peer stats (`peer_stats`), weekly research/compliance digests, patient/customer content library, and seasonal beats.
2. **MerchantContext** (`merchant`): Business state. Identity (name, owner, locality, languages), subscription status, 30d/7d performance metrics, active/expired offers, conversation history, customer aggregate counts, and derived signals.
3. **TriggerContext** (`trigger`): The specific event prompting the message *right now*. Scope (`merchant` or `customer`), kind (`research_digest`, `perf_dip`, `recall_due`, `festival_upcoming`, etc.), urgency (1–5), payload, and suppression key.
4. **CustomerContext** (`customer`, optional): Used exclusively for customer-facing messages sent *on behalf of the merchant*. Captures customer identity, language mix, visit history, lifecycle state (`new`, `active`, `lapsed_soft`, `lapsed_hard`, `churned`), slot preferences, and consent scope.

---

## 3. High-Scoring Rubric Dimensions & Mitigation Matrix

Messages are scored by an LLM judge on 5 dimensions (0–10 each = 50 total). The table below outlines how our strategy guarantees maximum points on each dimension:

| Dimension | Target | How Our Strategy Delivers It |
|---|:---:|---|
| **1. Specificity** | 10 / 10 | Pre-extract 3–5 exact verifiable numbers, dates, source citations (e.g. *"JIDA Oct 2026 p.14"*, *"38% lower caries"*, *"2,100 patients"*, batch numbers, exact prices). Never use generic phrases like "special discount". |
| **2. Category Fit** | 10 / 10 | Strict adherence to `CategoryContext.voice`. Clinical-peer tone for dentists with "Dr." prefix; operator register for restaurants ("covers", "AOV"); coach register for gyms. Immediate rejection of vertical taboo words. |
| **3. Merchant Fit** | 10 / 10 | Address owner by first name (`owner_first_name`). Reference their real metrics, real active offers, and honor `identity.languages` (using Hindi-English code-mix when `hi` is present). |
| **4. Trigger Relevance** | 10 / 10 | Establish "why now" immediately in the first sentence referencing the trigger payload. Distinguish external industry shifts from internal performance alerts. |
| **5. Engagement Compulsion** | 10 / 10 | Deploy psychological levers: loss aversion, social proof, effort externalization (*"I've drafted it — just say go"*), curiosity hooks, or low-friction single binary CTAs (YES/NO). |

---

## 4. Hard Operational Guardrails (Zero-Penalty Policy)

Every mechanical penalty drops overall rank regardless of literary quality. Our post-LLM validation layer enforces the following checks:

| Violation | Judge Penalty | Our Built-In Defense |
|---|:---:|---|
| **URL in Message Body** | **-3 points** | **Zero tolerance**: All URLs stripped from `body` (resolves discrepancy between brief §5 and testing brief F.4). |
| **Missing Action Schema Field** | **-2 points** | Enforce all required action keys: `conversation_id`, `merchant_id`, `send_as`, `trigger_id`, `cta`, `suppression_key`, `rationale`, `body`, plus nullable `customer_id` (`None` for merchant-facing). Never omit any of the 6 fields explicitly flagged in F.2 (`conversation_id`, `send_as`, `trigger_id`, `cta`, `suppression_key`, `rationale`). |
| **Empty Body on `send`** | **-2 points** | Assert `len(body.strip()) > 0`. Fallback to category-grounded template if LLM outputs empty. |
| **Verbatim Body Repetition** | **-2 points / repeat** | Track `sent_bodies` per `conversation_id`. Block and re-generate if token similarity > 90%. |
| **Fabricated Facts / Numbers** | **-2 points** | Regex-verify every number/percentage in `body` against input context payload. If ungrounded, purge or re-prompt. |
| **Exposed Internal Jargon** | **-1 point** | Denylist filter: Words like `"CTR"`, `"signals"`, `"suppression_key"`, `"delta_7d"`, `"payload"` must never enter `body`. |
| **Multi-Choice CTA Clutter** | Score Penalty | Ensure exactly one primary CTA, positioned in the final sentence. |
| **Timeouts (> 30s)** | **-1 point / timeout** | Hard LLM call timeout at 15–20s. Return empty `actions: []` gracefully if approaching budget limit. |
| **Health Check Failure (3×)** | **Disqualification / -10 pts** | §2.4 cites disqualification; §10 table cites -10 penalty. Treated as disqualification risk — `/v1/healthz` is an instant in-memory read. |

---

### Key Spec Discrepancies & Resolutions
1. **URL Prohibition**: `challenge-brief.md` §5 mentions URLs are "allowed when they add clear value", but `api-call-examples.md` Example F.4 enforces a **hard fail, -3 penalty** ("Meta would reject"). **Resolution**: Strict zero-tolerance for URLs in the message body.
2. **Dual Submission Models**: `challenge-brief.md` §7 describes a static deliverable (`bot.py` with `compose()` + `submission.jsonl` + `README.md`), whereas `challenge-testing-brief.md` mandates a stateful live HTTP server evaluated via `/v1/*` endpoints. **Resolution**: `bot.py` is built dual-mode — it provides the top-level `compose(...)` function AND runs as a FastAPI HTTP server for live testing. Additionally, we generate `submission.jsonl` for the 30 canonical pairs.
3. **Healthz Failure Severity**: `challenge-testing-brief.md` §2.4 specifies 3 consecutive failures = **disqualification**, while the §10 failure table notes a **-10 operational penalty**. **Resolution**: Treat as disqualification risk; `/v1/healthz` is an instantaneous, sub-millisecond in-memory dictionary read with zero external dependencies.

---

## 5. Multi-Turn Conversation State Machine (`/v1/reply`)

The simulator tests 3 critical scenarios in Phase 4 (+30 points potential). We handle them with deterministic state tracking:

```
[Inbound Message Received]
           │
           ├─► 1. Auto-Reply Pattern Detected? ("Thank you for contacting...", canned repeat)
           │       ├── Turn 1: action="send" (short prompt to flag for owner)
           │       ├── Turn 2: action="wait" (wait_seconds=86400 / 24h backoff)
           │       └── Turn 3+: action="end" (exit gracefully, zero turns wasted)
           │
           ├─► 2. Affirmative Intent Detected? ("let's do it", "confirm", "proceed", "yes send")
           │       └── action="send" in ACTION MODE (provide draft, set next step, NEVER re-qualify)
           │
           ├─► 3. Hostile / Opt-out? ("stop", "spam", "useless", "bothering")
           │       └── action="end" (or 1-line polite apology + suppress merchant for 30d)
           │
           └─► 4. Soft Deferral? ("busy right now", "call later")
                   └── action="wait" (wait_seconds=1800-14400, back off politely)
```

---

## 6. Project Status & Implementation Tracking

### What Has Been Done Till Now
- [x] **Full Codebase & Brief Audit**: Read and analyzed `challenge-brief.md`, `challenge-testing-brief.md`, `engagement-design.md`, `engagement-research.md`, `examples/api-call-examples.md`, and `examples/case-studies.md`.
- [x] **Reverse-Engineered Judge Mechanics**: Studied `judge_simulator.py` to identify scoring weights, prompt formats, penalties, and test routines (`warmup`, `phase2_short`, `auto_reply_hell`, `intent_transition`, `hostile`, `full_evaluation`).
- [x] **Spec Discrepancy Reconciliation**:
  - Reconciled URL prohibition (hard ban on URLs in body per Example F.4).
  - Reconciled dual submission model (building `bot.py` with both `compose()` and FastAPI `/v1/*` server).
  - Reconciled healthz penalty ambiguity (§2.4 DQ vs §10 -10 penalty).
- [x] **Environment Checked & Verified**: Executed CLI check (`python3 --version && python3 -m pip show fastapi uvicorn pydantic httpx requests`) on local system (`/opt/miniconda3/lib/python3.13`):
  - Python: `3.13.5`
  - `fastapi`: `0.137.1`
  - `uvicorn`: `0.49.0`
  - `pydantic`: `2.11.7` (Confirmed Pydantic v2 syntax `BaseModel`, `field_validator`, etc.)
  - `httpx`: `0.28.1`
  - `requests`: `2.32.4`
  - Additional installed LLM SDKs detected: `anthropic 0.109.2`, `groq 1.4.0`, `ollama 0.6.2`
  *(Note: LLM provider API keys are currently unset in the local environment; `bot.py` is architected with a deterministic grounded fallback synthesizer plus plug-and-play LLM provider configuration).*
- [x] **Strategy & Guardrails Formalized**: Documented 4-context architecture, grounding fact pre-extraction, validation filters, and multi-turn state machines.
- [x] **Step 1: Dataset Generation**: Ran `dataset/generate_dataset.py` expanding seeds to 50 merchants, 200 customers, 100 triggers, and the 30 `test_pairs.json`.
- [x] **Step 2: Core Server Implementation (`bot.py`)**:
  - Implemented FastAPI app with in-memory stores (`contexts`, `conversations`, `suppression`).
  - Implemented `/v1/healthz`, `/v1/metadata`, and `/v1/context` with strictly idempotent versioning.
  - Implemented fact pre-extractor and vertical tone routing across all 5 categories.
  - Implemented zero-tolerance post-composition validator (URL stripping, jargon denylist, single CTA, anti-repetition).
  - Exported top-level `compose(...)` function for module interoperability.
- [x] **Step 3: Multi-Turn Logic (`/v1/reply`)**:
  - Canned auto-reply streak counter (Turn 1 send $\to$ Turn 2 wait $\to$ Turn 3 end).
  - Immediate intent transition to ACTION mode (zero qualifying questions).
  - Hostility opt-out and soft deferral `wait` handlers.
- [x] **Step 4: Local Simulator Testing & Harness Audit**:
  - Validated via `judge_simulator.py` with 100% `[PASS]` on `warmup`, `auto_reply`, `intent`, and `hostile`.
  - Audited `judge_simulator.py` diff against clean original source: verified 100% untouched scoring logic (`LLMScorer.SYSTEM`, rubric dimensions, penalty rules, `ScoreResult`). Only configuration plumbing (`os.environ` fallback reads) and an offline heuristic `MockJudgeProvider` were introduced.
- [x] **Step 5: Full Evaluation, Plagiarism Audit & Edge-Case Hardening**:
  - Implemented true dual-tier architecture: optional Frontier LLM synthesizer via standard `urllib` (when `OPENAI_API_KEY`/`GROQ_API_KEY`/`LLM_API_KEY` is present) backed by deterministic 3-variant grounded synthesizer.
  - Rewrote 100% of case-study-adjacent prose from scratch, eliminating near-verbatim plagiarism risks (max similarity across all 30 submission rows vs all 10 Case Studies is **14.12%**, isolated strictly to shared proper nouns).
  - Fixed `/v1/context` idempotency bug: comparison updated to `>=` returning 409 `stale_version` on duplicate version pushes per Example 1.5.
  - Eliminated all fabricated default numbers (e.g. 245, 196, 57, 38) from context extraction; missing fields now use clean qualitative phrasing.
  - Hardened `/v1/reply` with **Rule 4.5 Off-Topic Interceptor**: gracefully declines out-of-scope administrative/tax/loan asks (*"I'll have to leave GST filing to your CA..."*) and redirects back to core business marketing (Example 2.7).
  - Made Tier 2 Dentist Research Digest fully adaptive to Phase 3 dynamic digest injection, consuming `digest.get("summary")` rather than static fluoride copy.
  - Enforced `/v1/tick` operational volume ceilings (§5): strict 20 actions/tick cap and max 1 message per merchant per tick.
  - Implemented and executed `test_guardrails_and_dataset.py` over the complete expanded dataset (100 triggers × 50 merchants across 26 trigger families):
    - 0 URL leaks (`http://`, `https://` hard assertion).
    - 0 forbidden jargon leaks (`CTR`, `signals`, `payload`, `suppression_key`, `delta_7d` hard assertion).
    - 100% required field presence (`conversation_id`, `merchant_id`, `send_as`, `trigger_id`, `cta`, `suppression_key`, `rationale`).
    - 100% length compliance (30 to 480 characters).
  - Regenerated and validated `submission.jsonl` (30 test pairs from `dataset/expanded/test_pairs.json`).
  - Authored comprehensive `README.md` with honest, uninflated performance caveats.
