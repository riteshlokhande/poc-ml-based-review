#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : semantic_drift_tracker.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def load_prompts(jsonl_path):
    with open(jsonl_path) as f:
        return [json.loads(line)["prompt"] for line in f]

def compare_embeddings(prompts, tokenizer_a, model_a, tokenizer_b, model_b):
    drift_scores = []
    for prompt in prompts:
        emb_a = get_embedding(prompt, tokenizer_a, model_a)
        emb_b = get_embedding(prompt, tokenizer_b, model_b)
        score = cosine_similarity([emb_a], [emb_b])[0][0]
        drift_scores.append((prompt, round(score, 4)))
    return drift_scores

def main():
    prompts = load_prompts("data/prompts.jsonl")

    # Tokenizer + model A
    tokenizer_a = AutoTokenizer.from_pretrained(config["tokenizers"]["base"])
    model_a = AutoModel.from_pretrained(config["tokenizers"]["base"])

    # Tokenizer + model B (custom or updated)
    tokenizer_b = AutoTokenizer.from_pretrained(config["tokenizers"]["custom_hf"])
    model_b = AutoModel.from_pretrained(config["tokenizers"]["custom_hf"])

    drift_scores = compare_embeddings(prompts, tokenizer_a, model_a, tokenizer_b, model_b)

    for prompt, score in drift_scores:
        print(f"Drift Score: {score} | Prompt: {prompt}")

if __name__ == "__main__":
    main()
