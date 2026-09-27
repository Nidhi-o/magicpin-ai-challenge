#!/usr/bin/env python3
"""
Test suite validating:
1. Full 100-trigger composition across all 50 merchants and 5 categories.
2. Strict guardrails:
   - Zero URL leakage (Meta policy & Example F.4: -3 penalty).
   - Zero internal jargon leakage (CTR, signals, payload, suppression_key, delta_7d).
   - Strict adherence to required fields (conversation_id, send_as, trigger_id, cta, suppression_key, rationale).
   - Character count bounds (30 <= len <= 480).
3. Exact compliance of submission.jsonl (30 test pairs).
"""

import os
import json
import bot

DATASET_EXPANDED = "dataset/expanded"
JARGON_TERMS = ["CTR", "signals", "payload", "suppression_key", "delta_7d"]
REQUIRED_ACTION_FIELDS = ["conversation_id", "merchant_id", "send_as", "trigger_id", "cta", "suppression_key", "rationale", "body"]
SUB_REQUIRED_FIELDS = ["test_id", "body", "cta", "send_as", "suppression_key", "rationale"]


def test_expanded_dataset():
    categories = {}
    for cf in os.listdir(os.path.join(DATASET_EXPANDED, "categories")):
        if cf.endswith(".json"):
            slug = cf.replace(".json", "")
            with open(os.path.join(DATASET_EXPANDED, "categories", cf)) as f:
                categories[slug] = json.load(f)

    merchants = {}
    for mf in os.listdir(os.path.join(DATASET_EXPANDED, "merchants")):
        if mf.endswith(".json"):
            with open(os.path.join(DATASET_EXPANDED, "merchants", mf)) as f:
                m = json.load(f)
                merchants[m["merchant_id"]] = m

    customers = {}
    for cuf in os.listdir(os.path.join(DATASET_EXPANDED, "customers")):
        if cuf.endswith(".json"):
            with open(os.path.join(DATASET_EXPANDED, "customers", cuf)) as f:
                c = json.load(f)
                customers[c["customer_id"]] = c

    triggers = []
    for tf in os.listdir(os.path.join(DATASET_EXPANDED, "triggers")):
        if tf.endswith(".json"):
            with open(os.path.join(DATASET_EXPANDED, "triggers", tf)) as f:
                triggers.append(json.load(f))

    print(f"[TEST 1] Composing across {len(triggers)} expanded triggers (50 merchants, 5 categories)...")
    assert len(triggers) == 100, f"Expected 100 triggers, found {len(triggers)}"

    by_category = {}
    by_kind = {}

    for idx, t in enumerate(triggers):
        m = merchants[t["merchant_id"]]
        cat_slug = m.get("category_slug", "dentists")
        cat = categories.get(cat_slug, {})
        cust = customers.get(t.get("customer_id")) if t.get("customer_id") else None

        action = bot.compose(category=cat, merchant=m, trigger=t, customer=cust)
        body = action.get("body", "")

        # Category and Kind tracking
        by_category[cat_slug] = by_category.get(cat_slug, 0) + 1
        by_kind[t["kind"]] = by_kind.get(t["kind"], 0) + 1

        # 1. Zero URL Leakage
        assert "http://" not in body and "https://" not in body, f"URL leaked in trigger {t['id']}: {body}"

        # 2. Zero Internal Jargon
        for term in JARGON_TERMS:
            assert term.lower() not in body.lower(), f"Jargon '{term}' leaked in trigger {t['id']}: {body}"

        # 3. Action Schema Completeness
        for field in REQUIRED_ACTION_FIELDS:
            assert field in action and action[field] is not None, f"Missing field '{field}' in trigger {t['id']}"

        # 4. Message Length Bounds
        assert 30 <= len(body) <= 480, f"Body length out of bounds ({len(body)}) in trigger {t['id']}"

    print(f"  ✓ All 100 triggers passed with 0 errors!")
    print(f"  ✓ Category coverage: {by_category}")
    print(f"  ✓ Trigger kinds covered: {len(by_kind)} kinds")


def test_submission_file():
    print("[TEST 2] Validating submission.jsonl (30 canonical test pairs)...")
    with open("submission.jsonl") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 30, f"Expected 30 submission rows, found {len(lines)}"

    for row in lines:
        test_id = row.get("test_id")
        body = row.get("body", "")

        # 1. Zero URL Leakage
        assert "http://" not in body and "https://" not in body, f"URL leaked in {test_id}: {body}"

        # 2. Zero Internal Jargon
        for term in JARGON_TERMS:
            assert term.lower() not in body.lower(), f"Jargon '{term}' leaked in {test_id}: {body}"

        # 3. Schema Completeness
        for field in SUB_REQUIRED_FIELDS:
            assert field in row and row[field] is not None, f"Missing field '{field}' in {test_id}"

        # 4. Message Length Bounds
        assert 30 <= len(body) <= 480, f"Body length out of bounds ({len(body)}) in {test_id}"

    print(f"  ✓ All 30 submission.jsonl lines passed with 0 errors!\n")


if __name__ == "__main__":
    test_expanded_dataset()
    test_submission_file()
    print("ALL TESTS PASSED SUCCESSFULLY.")
