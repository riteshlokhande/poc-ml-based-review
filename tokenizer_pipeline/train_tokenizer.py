#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : train_tokenizer.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import yaml
from tokenizers import BertWordPieceTokenizer
from utils import load_lines

with open("./tokenizer_pipeline/config.yaml") as f:
    cfg = yaml.safe_load(f)

tokenizer = BertWordPieceTokenizer(lowercase=cfg["lowercase"])
tokenizer.train(
    files=[cfg["data_path"]],
    vocab_size=cfg["vocab_size"],
    special_tokens=cfg["special_tokens"]
)

# Inject domain tokens
domain_tokens = load_lines(cfg["domain_tokens_path"])
tokenizer.add_tokens(domain_tokens)

tokenizer.save_model(cfg["output_dir"])
print(f"✅ Tokenizer saved to {cfg['output_dir']}")
