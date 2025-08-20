#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : utils.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import unicodedata

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip())

def load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [normalize_text(line) for line in f if line.strip()]
