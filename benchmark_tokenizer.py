#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : benchmark_tokenizer.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

from transformers import RobertaTokenizer
import json
import argparse
from collections import Counter

def benchmark_fragmentation(jsonl_path, tokenizer_name=config["tokenizers"]["base"], max_len=128):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    token_counts = Counter()
    fragment_stats = []

    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            tokens = sample.get("tokens", [])
            labels = sample.get("labels", [])
            assert len(tokens) == len(labels)

            for token, label in zip(tokens, labels):
                if label.startswith("B-"):
                    sub_tokens = tokenizer.tokenize(token)
                    token_counts[token] += 1
                    fragment_stats.append((token, len(sub_tokens)))

    print(f"🔍 Fragmentation Report for {jsonl_path}")
    for token, count in token_counts.items():
        avg_frag = sum(l for t, l in fragment_stats if t == token) / count
        print(f"🧠 Token: '{token}' | Avg Fragments: {avg_frag:.2f} | Count: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=config["tokenizers"]["base"])
    args = parser.parse_args()

    benchmark_fragmentation(args.jsonl_path, args.tokenizer)
