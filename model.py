#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : model.py
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
import torch.nn as nn
from transformers import RobertaModel

class ContrastiveClassifier(nn.Module):
    def __init__(self, base_model=config["tokenizers"]["base"], emb_dim=768, margin=0.3):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(base_model)
        self.classifier = nn.Linear(emb_dim, 2)
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        a_out = self.encoder(**anchor).last_hidden_state[:, 0, :]
        p_out = self.encoder(**positive).last_hidden_state[:, 0, :]
        n_out = self.encoder(**negative).last_hidden_state[:, 0, :]

        pos_sim = self.cos(a_out, p_out)
        neg_sim = self.cos(a_out, n_out)
        contrastive_loss = torch.relu(self.margin - pos_sim + neg_sim).mean()

        logits = self.classifier(a_out)
        return logits, contrastive_loss
