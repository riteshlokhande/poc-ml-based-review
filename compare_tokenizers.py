#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : compare_tokenizers.py
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
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt

def load_samples(jsonl_path):
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]

def get_fragmentation_stats(samples, tokenizer, name):
    stats = defaultdict(list)
    for sample in samples:
        tokens = sample["tokens"]
        for token in tokens:
            sub_tokens = tokenizer.tokenize(token) if hasattr(tokenizer, "tokenize") else tokenizer.encode(token).tokens
            stats[token].append(len(sub_tokens))
    return stats

def plot_fragmentation_comparison(stats_dict):
    plt.figure(figsize=(10, 6))
    for name, stats in stats_dict.items():
        avg_frags = [sum(v)/len(v) for k, v in stats.items()]
        plt.plot(sorted(avg_frags), label=name)
    plt.title("Tokenizer Fragmentation Comparison")
    plt.xlabel("Token Index (sorted by fragmentation)")
    plt.ylabel("Avg Fragments per Token")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    jsonl_path = "data/ner_labels.jsonl"
    samples = load_samples(jsonl_path)

    # Load tokenizers
    hf_tokenizer = AutoTokenizer.from_pretrained(config["tokenizers"]["base"])
    custom_tokenizer = Tokenizer.from_file("custom_tokenizer/tokenizer.json")

    # Benchmark
    stats_hf = get_fragmentation_stats(samples, hf_tokenizer, config["tokenizers"]["base"])
    stats_custom = get_fragmentation_stats(samples, custom_tokenizer, "custom-BPE")

    # Plot
    plot_fragmentation_comparison({
        config["tokenizers"]["base"]: stats_hf,
        "custom-BPE": stats_custom
    })

if __name__ == "__main__":
    main()
