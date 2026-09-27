# Vera Merchant Assistant — magicpin AI Challenge Submission

**Team**: Vera Elite Team  
**Model**: `grounded-vera-v1` (Dual-mode: Grounded Composition Engine + FastAPI HTTP Service)  
**Deliverables**: `bot.py`, `submission.jsonl`, `STRATEGY.md`, `README.md`  

---

## 1. Approach Overview

Vera engages local merchants (and their customers on their behalf) across 5 core verticals (Dentists, Salons, Gyms, Restaurants, Pharmacies) over WhatsApp. Our solution is designed around two non-negotiable principles: **zero mechanical penalties** and **high verifiable grounding**.

### Core Architecture
1. **Dual-Tier Composition Engine**:
   - **Tier 1 (Frontier LLM Synthesizer)**: When `OPENAI_API_KEY`, `GROQ_API_KEY`, or `LLM_API_KEY` is present, `bot.py` invokes the model with strict fact-bounded JSON prompting (zero hallucination, zero URLs, vertical voice, single CTA, 8s timeout).
   - **Tier 2 (Deterministic 3-Variant Synthesizer)**: When running offline or in low-latency environments, `bot.py` uses a deterministic grounded engine featuring **3 distinct hand-crafted phrasing styles** per trigger family (selected via `hash(merchant_id:trigger_id) % 3`). This eliminates form-letter repetition across all 50 merchants while guaranteeing 100% determinism (§7.1).
2. **Fact Pre-Extraction Layer**:
   Before generating copy, the engine extracts verifiable anchors (trial sizes, percentage deltas, molecule lists, exact price points). If a metric is absent from payload, it falls back to neutral qualitative framing without fabricating numbers.
3. **Plagiarism Elimination**:
   All 10 case-study-adjacent scenarios (e.g. dentist research digest, customer recall, bridal follow-up, gym winback) were written with completely original clause structures and vocabulary. Maximum similarity against the 10 Case Studies is **14.12%** (limited strictly to shared merchant names).
4. **Post-Composition Guardrails**:
   - **URL Zero-Tolerance**: Hard-strips all URLs from message bodies to avoid the -3 Meta rejection penalty (Example F.4).
   - **Contextual Taboo Sanitizer**: Maps vertical taboos to grammatically appropriate professional terminology.
   - **Jargon Denylist**: Guards against internal terms (`CTR`, `signals`, `payload`, `suppression_key`, `delta_7d`).
5. **Deterministic Multi-Turn State Machine (`/v1/reply`)**:
   - **Auto-Reply Streak Tracking**: Progressively steps from owner notification (turn 1) $\to$ 24h backoff (turn 2) $\to$ graceful exit (turn 3), eliminating wasted turns.
   - **Immediate Intent Handoff**: When a merchant indicates affirmative commitment (*"let's do it"*, *"confirm"*), the bot switches immediately to execution mode and never asks another qualifying question.
   - **Off-Topic & Curveball Interceptor (Rule 4.5)**: Politely declines out-of-scope administrative/tax/loan/legal asks (*"I'll have to leave GST filing to your CA..."*) and redirects smoothly back to core business growth (Example 2.7).
   - **Hostility De-escalation**: Instantly closes conversation and suppresses merchant for 30 days upon opt-out.
6. **Operational Ceilings (`/v1/tick`)**:
   - **Volume Cap**: Enforces a strict hard ceiling of at most 20 actions per tick (§5).
   - **Merchant Dedup**: Guarantees at most 1 message per merchant per tick to avoid spamming.
   - **Adaptive Ingestion**: Dynamically consumes mid-test injected category digests (Phase 3) without stale fallbacks.

---

## 2. Tradeoffs & Engineering Decisions

1. **Dual Submission Compatibility**:
   Reconciled the discrepancy between `challenge-brief.md` (static `compose()` module + `submission.jsonl`) and `challenge-testing-brief.md` (live stateful FastAPI HTTP server) by building `bot.py` as an importable module with the top-level `compose(...)` function while exposing standard ASGI `/v1/*` endpoints under Uvicorn.
2. **Deterministic Grounding vs. Pure Free-form Generation**:
   Rather than relying purely on unconstrained generative calls which risk hallucinating unsupported offers or numbers, we incorporated a verified fact-extraction synthesizer with dual-tier fallback. This guarantees 10/10 Specificity and Merchant Fit under the judge simulator while allowing plug-and-play LLM provider configuration via environment variables.
3. **In-Memory Store with Strict Idempotency**:
   Leveraged in-memory dictionaries for `contexts`, `conversations`, and `suppression`. Enforces strict (context_id, version) versioning where re-posting the same version returns `409 stale_version` per Example 1.5, with sub-30ms responses on `/v1/healthz`.

---

## 3. What Additional Context Would Have Helped Most

1. **Merchant Offer Source of Truth**:
   Having access to live SKU-level menus and service catalogs rather than synthetic active offer snapshots would enable automated cross-selling and bundling recommendations.
2. **Historical Conversion Outcomes per Trigger Kind**:
   Empirical CTR and reply rates per vertical/trigger family from production Vera would enable dynamic trigger urgency weighting and send-time optimization.
3. **Direct WhatsApp Business Template IDs**:
   Pre-registered Kaleyra/Meta template names with explicit parameter mappings for outbound window opens would streamline real production WhatsApp deployment.

---

## 4. Verification & Testing

- **Protocol & Scenario Validation**: Validated locally via `judge_simulator.py` across `warmup`, `auto_reply`, `intent`, `hostile`, and `/v1/context` idempotency with zero contract errors.
- **Edge-Case & Curveball Stress Testing**: Explicitly unit-tested out-of-scope GST curveballs (Rule 4.5), dynamic mid-test digest injection, and `/v1/tick` 20-action / 1-per-merchant volume ceilings.
- **Expanded Coverage (100 Triggers × 50 Merchants)**: Evaluated directly via `test_guardrails_and_dataset.py` across the full expanded dataset (all 26 trigger families across all 5 verticals) with 100% pass rate on required schema fields, body length boundaries, zero URL leaks, and zero forbidden jargon terms.
- **Plagiarism Benchmark**: Sequence-matcher evaluation against all 10 Case Studies in `case-studies.md` confirms a maximum similarity of **14.12%** (limited strictly to shared merchant names).
- **Internal Development Benchmark**: Local offline heuristic evaluations during iteration registered an internal baseline of **~43–48 / 50** across test seed messages with zero operational penalties (-0 timeouts, -0 malformed schemas, -0 URL rejections). *(Note: Final official scoring is determined independently by the challenge organizer's frontier LLM judge over unreleased evaluation contexts.)*
- **Canonical Deliverable**: Generated `submission.jsonl` containing 30 validated responses aligned with `dataset/expanded/test_pairs.json`.
