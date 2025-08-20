#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : diagnostics.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

from tokenizers import Tokenizer
from utils import load_lines

tokenizer = Tokenizer.from_file("output/tokenizer/tokenizer.json")
lines = load_lines("data/corpus.txt")

for line in lines:
    encoding = tokenizer.encode(line)
    tokens = encoding.tokens
    fragments = [t for t in tokens if len(t) == 1 or t.startswith("##")]
    rate = round(len(fragments) / len(tokens), 3)
    if rate > 0.1:
        print(f"⚠️ Fragmented: '{line}' → {tokens} (Rate: {rate})")
