#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : generate_label_list.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

# ===============================
# Shell Command: python3 generate_label_list.py
# Script Name: generate_label_list.py
# Description: Dynamically generate BIO label list from input text using semantic heuristics
# ===============================

import re
import spacy

# Load spaCy's small English model for semantic chunking
nlp = spacy.load("en_core_web_sm")

def extract_entity_types(text):
    entity_types = set()
    doc = nlp(text)

    # Step 1: Named entities from spaCy
    for ent in doc.ents:
        if ent.label_ in ["DATE"]:
            entity_types.add("DATE")
        elif ent.label_ in ["MONEY"]:
            entity_types.add("AMOUNT")
        elif ent.label_ in ["ORG", "CARDINAL"] and "account" in ent.text.lower():
            entity_types.add("ACCOUNT")

    # Step 2: Regex-based detection
    if re.search(r'\bPAYMENT[-_]\d+\b', text, re.IGNORECASE):
        entity_types.add("PAYMENT_ID")
    if re.search(r'[₹$€£]\s?\d+', text):
        entity_types.add("AMOUNT")
    if re.search(r'\bX{4,}\d{2,}\b', text):
        entity_types.add("ACCOUNT")
    if re.search(r'\b\d{4}-\d{2}-\d{2}\b', text) or re.search(r'\b\d{8}\b', text):
        entity_types.add("DATE")
    if re.search(r'\b(Single|Dual)\s+validation\b', text, re.IGNORECASE):
        entity_types.add("VALIDATION_TYPE")
    if re.search(r'\b(PASSED|FAILED|COMPLETED|PENDING)\b', text, re.IGNORECASE):
        entity_types.add("VALIDATION_STATUS")
    if re.search(r'\boutcome\b', text, re.IGNORECASE):
        entity_types.add("OUTCOME")
    if re.search(r'\bQ\d+:', text):
        entity_types.add("QUESTION")

    # Step 3: Noun phrase analysis
    for chunk in doc.noun_chunks:
        if "validation" in chunk.text.lower():
            entity_types.add("VALIDATION_TYPE")
        if "payment" in chunk.text.lower() and re.search(r'\d+', chunk.text):
            entity_types.add("PAYMENT_ID")

    return sorted(entity_types)

def generate_bio_labels(text):
    entity_types = extract_entity_types(text)
    label_list = ["O"]
    for etype in entity_types:
        label_list.append(f"B-{etype}")
        label_list.append(f"I-{etype}")
    return label_list

if __name__ == "__main__":
    # Example input
    input_text = """Payment PAYMENT-000996 of ₹165 from account XXXXXXXX5970 was processed on 2025-06-12 with Single validation.
    First validation completed on 20250607 with outcome: PASSED.
    Q1: Is Dual validation applied when the payment amount is above ₹5000?
    Q2: Did all applicable validations pass for this payment?"""

    labels = generate_bio_labels(input_text)
    print("✅ Generated BIO Label List:")
    print(labels)
