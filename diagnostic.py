#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : diagnostic.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import Counter
import pandas as pd

# Load model and tokenizer
model_path = "models/ner"
tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

# Sample prompts (replace with your actual dataset)
prompts = [
    "Validate payment of ₹5000 to Acme Corp on 12 Aug 2025",
    "Check if invoice #INV123 from Globex Ltd is approved",
    "Was ₹7500 refunded to John Doe on 15 Aug?"
]

# Diagnostic results
results = []

for text in prompts:
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding["offset_mapping"][0].tolist()

    with torch.no_grad():
        outputs = model(**{k: v for k, v in encoding.items() if k in ["input_ids", "attention_mask"]})
        preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        labels = [model.config.id2label[p] for p in preds]

    entity_spans = []
    current_entity = None
    for token, label, offset in zip(tokens, labels, offsets):
        if label.startswith("B-"):
            if current_entity:
                entity_spans.append(current_entity)
            current_entity = {"label": label[2:], "start": offset[0], "end": offset[1], "text": text[offset[0]:offset[1]]}
        elif label.startswith("I-") and current_entity:
            current_entity["end"] = offset[1]
            current_entity["text"] += text[offset[0]:offset[1]]
        else:
            if current_entity:
                entity_spans.append(current_entity)
                current_entity = None

    if current_entity:
        entity_spans.append(current_entity)

    results.append({
        "text": text,
        "tokens": tokens,
        "labels": labels,
        "entities": entity_spans
    })

# Display summary
for r in results:
    print(f"\n📝 Prompt: {r['text']}")
    print(f"🔍 Entities Found: {len(r['entities'])}")
    for ent in r["entities"]:
        print(f"  - {ent['label']}: '{ent['text']}'")

# Optional: Entity label distribution
label_counts = Counter()
for r in results:
    label_counts.update(r["labels"])
print("\n📊 Label Distribution:", dict(label_counts))
