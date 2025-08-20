#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : generate_q1_eval.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import json
import random
from pathlib import Path

def generate_triplet(anchor, prompt_pool, label):
    # Simple positive/negative sampling
    positive = random.choice([p for p in prompt_pool if p != anchor])
    negative = random.choice([p for p in prompt_pool if p != anchor and p != positive])
    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "label": label
    }

def main():
    # === Replace with your real prompt logic ===
    anchors = [
        "Is the payment amount above threshold?",
        "Does the invoice match the PO?",
        "Is the vendor verified?",
        "Was the transaction flagged for review?"
    ]
    prompt_pool = anchors + [
        "Is the PO number missing?",
        "Is the payment delayed?",
        "Is the vendor blacklisted?",
        "Is the invoice total mismatched?"
    ]

    # === Rule-based label assignment (stub) ===
    def assign_label(anchor):
        return "Yes" if "verified" in anchor or "match" in anchor else "No"

    triplets = []
    for anchor in anchors:
        label = assign_label(anchor)
        triplet = generate_triplet(anchor, prompt_pool, label)
        triplets.append(triplet)

    # === Save to JSONL ===
    out_path = Path("data/q1_eval.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    print(f"✅ Generated {len(triplets)} triplets to {out_path}")

if __name__ == "__main__":
    main()
