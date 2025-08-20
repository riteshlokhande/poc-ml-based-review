#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : tokenizer_diagnostics.py
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
from transformers import RobertaTokenizer
from collections import defaultdict
import argparse

def analyze_fragmentation(jsonl_path, tokenizer_name=config["tokenizers"]["base"]):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    token_stats = defaultdict(list)

    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            tokens, labels = sample["tokens"], sample["labels"]
            for token, label in zip(tokens, labels):
                if label.startswith("B-"):
                    sub_tokens = tokenizer.tokenize(token)
                    token_stats[token].append(len(sub_tokens))

    print(f"\n🔍 Fragmentation Report for {jsonl_path}")
    for token, fragments in sorted(token_stats.items(), key=lambda x: -len(x[1])):
        avg_frag = sum(fragments) / len(fragments)
        if avg_frag > 1.5:
            print(f"⚠️ Token: '{token}' | Avg Fragments: {avg_frag:.2f} | Count: {len(fragments)}")

def suggest_merge_rules(jsonl_path, tokenizer_name=config["tokenizers"]["base"]):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    merge_candidates = set()

    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            tokens = sample["tokens"]
            for token in tokens:
                sub_tokens = tokenizer.tokenize(token)
                if len(sub_tokens) > 2:
                    merge_candidates.add(token)

    print("\n🧠 Suggested Merge Rules for Custom Tokenizer:")
    for token in sorted(merge_candidates):
        print(f"🔧 Consider merging: '{token}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=config["tokenizers"]["base"])
    args = parser.parse_args()

    analyze_fragmentation(args.jsonl_path, args.tokenizer)
    suggest_merge_rules(args.jsonl_path, args.tokenizer)
