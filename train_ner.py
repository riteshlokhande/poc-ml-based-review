#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : train_ner.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AdamW
from ner_dataset import NERDataset
from ner_model import build_ner_model

def train(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = {k: batch[k] for k in batch if k != "labels"}
            labels = batch["labels"]
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} ✅ Loss: {total_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    label_list = ["O", "B-ENTITY"]
    label_map = {label: i for i, label in enumerate(label_list)}

    tokenizer = RobertaTokenizer.from_pretrained(config["tokenizers"]["base"])
    dataset = NERDataset(args.data_path, tokenizer, label_map)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = build_ner_model(label_map)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train(model, loader, optimizer, args.epochs)
