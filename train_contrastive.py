#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : train_contrastive.py
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
from model import ContrastiveClassifier
from dataset import ContrastiveQADataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

def train(model, dataloader, optimizer, criterion, task_name, epochs=3):
    model.train()
    for epoch in range(epochs):
        correct, total = 0, 0
        for anchor, positive, negative, label in dataloader:
            for k in anchor: anchor[k] = anchor[k].squeeze(0)
            for k in positive: positive[k] = positive[k].squeeze(0)
            for k in negative: negative[k] = negative[k].squeeze(0)

            logits, contrastive_loss = model(anchor, positive, negative)
            clf_loss = criterion(logits, label)
            loss = clf_loss + contrastive_loss

            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc = correct / total
        print(f"[{task_name}] Epoch {epoch+1} ✅ Loss: {loss.item():.4f} | Accuracy: {acc:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["q1", "q2"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config["tokenizers"]["base"])
    dataset = ContrastiveQADataset(args.data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ContrastiveClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    train(model, loader, optimizer, criterion, args.task, args.epochs)
