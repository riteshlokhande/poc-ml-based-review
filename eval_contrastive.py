#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : eval_contrastive.py
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
import numpy as np
import json
import argparse
from model import ContrastiveClassifier
from dataset import ContrastiveQADataset
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader

def evaluate(model, dataloader, task_name, margin=0.3, log_path=None):
    model.eval()
    correct, total = 0, 0
    violations, misclassified = []

    with torch.no_grad():
        for anchor, positive, negative, label in dataloader:
            for k in anchor: anchor[k] = anchor[k].squeeze(0)
            for k in positive: positive[k] = positive[k].squeeze(0)
            for k in negative: negative[k] = negative[k].squeeze(0)

            logits, contrastive_loss = model(anchor, positive, negative)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            # Embeddings for margin check
            a_out = model.encoder(**anchor).last_hidden_state[:, 0, :]
            p_out = model.encoder(**positive).last_hidden_state[:, 0, :]
            n_out = model.encoder(**negative).last_hidden_state[:, 0, :]

            pos_sim = model.cos(a_out, p_out)
            neg_sim = model.cos(a_out, n_out)
            violation_mask = (pos_sim - neg_sim < margin)

            for i in range(len(label)):
                if preds[i] != label[i] or violation_mask[i]:
                    misclassified.append({
                        "anchor_ids": anchor["input_ids"][i].tolist(),
                        "pred": preds[i].item(),
                        "true": label[i].item(),
                        "pos_sim": round(pos_sim[i].item(), 4),
                        "neg_sim": round(neg_sim[i].item(), 4),
                        "violation": bool(violation_mask[i].item())
                    })

    acc = correct / total
    print(f"[{task_name}] ✅ Accuracy: {acc:.2%} | Margin Violations: {sum(v['violation'] for v in misclassified)}")

    if log_path:
        with open(log_path, "w") as f:
            json.dump(misclassified, f, indent=2)
        print(f"🔍 Misclassified samples saved to {log_path}")

def export_embeddings(model, dataloader, out_path="embeddings.npy"):
    model.eval()
    all_embeds = []

    with torch.no_grad():
        for anchor, _, _, _ in dataloader:
            for k in anchor: anchor[k] = anchor[k].squeeze(0)
            a_out = model.encoder(**anchor).last_hidden_state[:, 0, :]
            all_embeds.append(a_out.cpu().numpy())

    all_embeds = np.vstack(all_embeds)
    np.save(out_path, all_embeds)
    print(f"🧠 Embeddings exported to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["q1", "q2"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="misclassified.json")
    parser.add_argument("--export_embeds", action="store_true")
    parser.add_argument("--embed_path", type=str, default="embeddings.npy")
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(config["tokenizers"]["base"])
    dataset = ContrastiveQADataset(args.data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=8)

    model = ContrastiveClassifier()
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))

    evaluate(model, loader, args.task, log_path=args.log_path)

    if args.export_embeds:
        export_embeddings(model, loader, out_path=args.embed_path)
