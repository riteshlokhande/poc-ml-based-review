#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : ner_dataset.py
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
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, path, tokenizer, label_map, max_len=128):
        self.samples = [json.loads(line) for line in open(path)]
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens, labels = sample["tokens"], sample["labels"]
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                                  padding="max_length", max_length=self.max_len, return_tensors="pt")

        word_ids = encoding.word_ids()
        label_ids = []
        for i in range(len(word_ids)):
            if word_ids[i] is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.label_map[labels[word_ids[i]]])

        return {**{k: v.squeeze(0) for k, v in encoding.items()}, "labels": torch.tensor(label_ids)}

    def __len__(self):
        return len(self.samples)
