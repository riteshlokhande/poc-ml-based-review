#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : prepare_contrastive_dataset.py
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
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def load_contrastive_pairs(path="./data/contrastive_pairs.jsonl"):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def embed_prompts(pairs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embedded = []
    for p in pairs:
        anchor = model.encode(p["anchor"])
        neg_q1 = model.encode(p["hard_negative_q1"])
        neg_q2 = model.encode(p["hard_negative_q2"])
        embedded.append({
            "Payment ID": p["Payment ID"],
            "anchor_emb": anchor,
            "neg_q1_emb": neg_q1,
            "neg_q2_emb": neg_q2
        })
    return embedded

def save_embeddings(embedded, out_dir="./data/embeddings"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "anchor.npy"), np.array([e["anchor_emb"] for e in embedded]))
    np.save(os.path.join(out_dir, "neg_q1.npy"), np.array([e["neg_q1_emb"] for e in embedded]))
    np.save(os.path.join(out_dir, "neg_q2.npy"), np.array([e["neg_q2_emb"] for e in embedded]))

def build_triplets(pairs, labels_df):
    triplets_q1, triplets_q2 = [], []
    label_map = labels_df.set_index("Payment ID").to_dict()

    for p in pairs:
        pid = p["Payment ID"]
        q1_label = label_map["Q1 Label"][pid]
        q2_label = label_map["Q2 Label"][pid]

        triplets_q1.append({
            "anchor": p["anchor"],
            "positive": p["anchor"],  # same label
            "negative": p["hard_negative_q1"],
            "label": q1_label
        })
        triplets_q2.append({
            "anchor": p["anchor"],
            "positive": p["anchor"],
            "negative": p["hard_negative_q2"],
            "label": q2_label
        })

    return triplets_q1, triplets_q2

def save_triplets(triplets, path):
    with open(path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

def main():
    pairs = load_contrastive_pairs()
    labels_df = pd.read_csv("./data/qa_labels.csv")

    # Embed prompts
    embedded = embed_prompts(pairs)
    save_embeddings(embedded)

    # Build triplets
    triplets_q1, triplets_q2 = build_triplets(pairs, labels_df)
    save_triplets(triplets_q1, "./data/triplets_q1.jsonl")
    save_triplets(triplets_q2, "./data/triplets_q2.jsonl")

    print("📦 Contrastive training dataset ready: embeddings + triplets")

if __name__ == "__main__":
    main()
