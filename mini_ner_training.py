#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : mini_ner_training.py
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
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np

# Define label list
label_list = ["O", "B-AMOUNT", "I-AMOUNT", "B-ORG", "I-ORG", "B-DATE", "I-DATE"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# Instead of passing full examples with entities
examples = [
    {"text": "Validate payment of ₹5000 to Acme Corp on 12 Aug 2025", "entities": [(21, 26, "AMOUNT"), (30, 39, "ORG"), (43, 54, "DATE")]},
    {"text": "Was ₹7500 refunded to John Doe on 15 Aug?", "entities": [(4, 9, "AMOUNT"), (23, 31, "ORG"), (35, 42, "DATE")]}
]

# Split into two lists
texts = [
    "Validate payment of ₹5000 to Acme Corp on 12 Aug 2025",
    "Was ₹7500 refunded to John Doe on 15 Aug?"
]

entities = [
    [(21, 26, "AMOUNT"), (30, 39, "ORG"), (43, 54, "DATE")],
    [(4, 9, "AMOUNT"), (23, 31, "ORG"), (35, 42, "DATE")]
]

# Create dataset from texts only
ds = Dataset.from_dict({"text": texts})

# Tokenize and align labels
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.add_special_tokens({'additional_special_tokens': ['₹']})  # Optional: preserve ₹

def tokenize_and_align(example, idx):
    encoding = tokenizer(example["text"], return_offsets_mapping=True, truncation=True)
    labels = ["O"] * len(encoding["offset_mapping"])
    for start, end, label in entities[idx]:
        for i, (s, e) in enumerate(encoding["offset_mapping"]):
            if s >= start and e <= end:
                labels[i] = "B-" + label if s == start else "I-" + label
    encoding["labels"] = [label2id[l] for l in labels]
    return encoding

tokenized_ds = ds.map(lambda ex, idx: tokenize_and_align(ex, idx), with_indices=True)

# Load model
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))
model.config.id2label = id2label
model.config.label2id = label2id

# Training setup
args = TrainingArguments(
    output_dir="./ner_debug",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

# Train
trainer.train()

# 🔍 Debug: Show token-label pairs
for ex in tokenized_ds:
    tokens = tokenizer.convert_ids_to_tokens(ex["input_ids"])
    labels = [label_list[i] for i in ex["labels"]]
    print("\n🧾 Tokens & Labels:")
    for t, l in zip(tokens, labels):
        print(f"{t:15} {l}")
