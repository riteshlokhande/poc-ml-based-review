#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : benchmark_custom_tokenizer.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

from tokenizers import Tokenizer
import json
import argparse
from collections import defaultdict

def benchmark(jsonl_path, tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    stats = defaultdict(list)

    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            tokens = sample["tokens"]
            for token in tokens:
                sub_tokens = tokenizer.encode(token).tokens
                stats[token].append(len(sub_tokens))

    print(f"\n📊 Fragmentation Report for {tokenizer_path}")
    for token, frags in sorted(stats.items(), key=lambda x: -len(x[1])):
        avg_frag = sum(frags) / len(frags)
        if avg_frag > 1.5:
            print(f"⚠️ Token: '{token}' | Avg Fragments: {avg_frag:.2f} | Count: {len(frags)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    args = parser.parse_args()

    benchmark(args.jsonl_path, args.tokenizer_path)
