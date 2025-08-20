#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : stage0_build_tokenizer.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import os
import json
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from collections import Counter

# === Config ===
CORPUS_PATH = "tokenizer_pipeline/data/corpus.txt"
TOKENIZER_DIR = "tokenizer"
VOCAB_SIZE = 5000
MIN_FREQ = 2
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

# === Step 1: Train Tokenizer ===
def train_tokenizer():
    print("🔧 Training Byte-Level BPE Tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[CORPUS_PATH],
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS
    )
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_model(TOKENIZER_DIR)

    # Save tokenizer config for HF compatibility
    config = {
        "model_type": "roberta",
        "tokenizer_class": "RobertaTokenizerFast",
        "add_prefix_space": True
    }
    with open(os.path.join(TOKENIZER_DIR, "tokenizer_config.json"), "w") as f:
        json.dump(config, f)

    print(f"✅ Tokenizer trained and saved to '{TOKENIZER_DIR}'")

# === Step 2: Load HF-Compatible Tokenizer ===
def load_tokenizer():
    print("📦 Loading tokenizer with RobertaTokenizerFast...")
    return RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR, add_prefix_space=True)

# === Step 3: Run Tokenization Diagnostics ===
def diagnostic_tokenization(tokenizer, sample_lines):
    print("\n📊 Tokenization Diagnostics:")
    all_tokens = []
    fragmentation_count = 0

    for line in sample_lines:
        tokens = tokenizer.tokenize(line)
        all_tokens.extend(tokens)
        print(f"\n🔍 Line: {line}")
        print(f"🧩 Tokens: {tokens}")
        for word in line.split():
            if not any(word in tok or tok in word for tok in tokens):
                fragmentation_count += 1
                print(f"⚠️ Fragmented: '{word}' → {tokens}")

    token_freq = Counter(all_tokens)
    print(f"\n🔢 Total unique tokens: {len(token_freq)}")
    print(f"⚠️ Fragmentation rate: {fragmentation_count}/{len(sample_lines)} lines")

    top_tokens = token_freq.most_common(10)
    print("\n🔥 Top tokens:")
    for tok, freq in top_tokens:
        print(f"{tok}: {freq}")

# === Step 4: Sample Usage ===
def sample_usage(tokenizer):
    print("\n🧪 Sample Encoding:")
    sample_text = "Outcome: PASSED for txn_id_ABC123"
    encoded = tokenizer(sample_text, return_tensors="pt")
    print(f"Input IDs: {encoded['input_ids']}")
    print(f"Tokens: {tokenizer.tokenize(sample_text)}")

# === Main Execution ===
if __name__ == "__main__":
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"❌ Corpus file not found at '{CORPUS_PATH}'")

    train_tokenizer()
    tokenizer = load_tokenizer()

    # Diagnostics
    sample_lines = [
        "₹5000 validation passed for PAYMENT-000996",
        "Dual validation applied when amount exceeds ₹10000",
        "Outcome: PASSED",
        "txn_id_ABC123 flagged for manual review"
    ]
    diagnostic_tokenization(tokenizer, sample_lines)

    # Sample usage
    sample_usage(tokenizer)
