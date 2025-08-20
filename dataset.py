#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : dataset.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import torch
import json
from torch.utils.data import Dataset

class ContrastiveQADataset(Dataset):
    def __init__(self, path, tokenizer, label_map={"Yes": 1, "No": 0}, max_len=128):
        self.samples = [json.loads(line) for line in open(path)]
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __getitem__(self, idx):
        s = self.samples[idx]
        anchor = self.tokenizer(s["anchor"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        positive = self.tokenizer(s["positive"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        negative = self.tokenizer(s["negative"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        label = torch.tensor(self.label_map[s["label"]])
        return anchor, positive, negative, label

    def __len__(self):
        return len(self.samples)
