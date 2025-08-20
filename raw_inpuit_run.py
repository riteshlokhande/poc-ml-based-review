#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : raw_inpuit_run.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("models/ner")
tokenizer = AutoTokenizer.from_pretrained("models/ner")

ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

tokens = ["Payment", "PAYMENT-000996", "of", "₹165", "from", "account", "XXXXXXXX5970", "was", "processed", "on", "2025-06-12"]
text = " ".join(tokens)



results = ner_pipe(text)
for r in results:
    print(f"{r['word']} → {r['entity_group']} ({r['score']:.2f})")
