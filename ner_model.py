#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : ner_model.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

from transformers import RobertaForTokenClassification

def build_ner_model(label_map, base_model=config["tokenizers"]["base"]):
    return RobertaForTokenClassification.from_pretrained(base_model, num_labels=len(label_map))
