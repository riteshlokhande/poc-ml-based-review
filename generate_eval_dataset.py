#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : generate_eval_dataset.py
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

def generate_triplet(anchor, prompt_pool, label, inject_adversarial=False):
    positive = random.choice([p for p in prompt_pool if p != anchor])
    negative = random.choice([p for p in prompt_pool if p != anchor and p != positive])

    if inject_adversarial:
        negative = inject_negation(negative)

    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "label": label
    }

def inject_negation(prompt):
    if " is " in prompt:
        return prompt.replace(" is ", " is not ", 1)
    elif " does " in prompt:
        return prompt.replace(" does ", " does not ", 1)
    return "NOT " + prompt

def assign_label_q1(anchor):
    return "Yes" if "verified" in anchor or "match" in anchor else "No"

def assign_label_q2(anchor):
    return "Yes" if "threshold" in anchor or "flagged" in anchor else "No"

def generate_bio_labels(prompt):
    tokens = prompt.split()
    labels = []
    for token in tokens:
        if token.lower() in ["payment", "invoice", "vendor", "po", "transaction"]:
            labels.append("B-ENTITY")
        else:
            labels.append("O")
    return list(zip(tokens, labels))

def validate_bio_spans(samples):
    for i, tokens in enumerate(samples):
        labels = [l for _, l in tokens]
        for j in range(len(labels)):
            if labels[j].startswith("I-") and (j == 0 or labels[j-1] == "O"):
                print(f"⚠️ Span misalignment at sample {i+1}, token '{tokens[j][0]}'")

def main():
    anchors_q1 = [
        "Does the invoice match the PO?",
        "Is the vendor verified?",
        "Is the PO number missing?",
        "Is the invoice total mismatched?"
    ]
    anchors_q2 = [
        "Is the payment amount above threshold?",
        "Was the transaction flagged for review?",
        "Is the payment delayed?",
        "Is the vendor blacklisted?"
    ]
    prompt_pool = anchors_q1 + anchors_q2

    triplets_q1, triplets_q2, ner_samples = [], [], []

    for anchor in anchors_q1:
        label = assign_label_q1(anchor)
        triplets_q1.append(generate_triplet(anchor, prompt_pool, label, inject_adversarial=True))
        ner_samples.append(generate_bio_labels(anchor))

    for anchor in anchors_q2:
        label = assign_label_q2(anchor)
        triplets_q2.append(generate_triplet(anchor, prompt_pool, label, inject_adversarial=True))
        ner_samples.append(generate_bio_labels(anchor))

    # Validate BIO span alignment
    validate_bio_spans(ner_samples)

    # Save outputs
    Path("data").mkdir(exist_ok=True)

    with open("data/q1_eval.jsonl", "w") as f:
        for t in triplets_q1:
            f.write(json.dumps(t) + "\n")

    with open("data/q2_eval.jsonl", "w") as f:
        for t in triplets_q2:
            f.write(json.dumps(t) + "\n")

    with open("data/ner_labels.jsonl", "w") as f:
        for tokens in ner_samples:
            f.write(json.dumps({
                "tokens": [t for t, _ in tokens],
                "labels": [l for _, l in tokens]
            }) + "\n")

    print(f"✅ Generated {len(triplets_q1)} Q1 triplets")
    print(f"✅ Generated {len(triplets_q2)} Q2 triplets")
    print(f"🧠 Generated {len(ner_samples)} NER samples")

if __name__ == "__main__":
    main()
