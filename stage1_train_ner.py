#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : stage1_train_ner.py
Author     : Ritesh
Created    : 2025-08-18
Description: Train a NER model with Roberta, using defaults from config/config.yaml
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

# === PROJECT ROOT & CONFIG LOADING ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR         # adjust if script is nested, e.g. SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

def resolve_path(p: str) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)

# === DATA LOADING & TOKENIZATION ===
def load_data(prompt_path: str, span_path: str):
    df = pd.read_csv(prompt_path)
    with open(span_path, "r") as f:
        spans = [json.loads(line) for line in f]
    return df["prompt"].tolist(), spans

def build_label_maps():
    label_list = [
        "O",
        "B-AMOUNT", "I-AMOUNT",
        "B-ORG",    "I-ORG",
        "B-DATE",   "I-DATE",
        "B-PAYMENT_ID", "I-PAYMENT_ID",
        "B-ACCOUNT",    "I-ACCOUNT",
    ]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label_list, label2id, id2label

def tokenize_and_align(example, idx, tokenizer, spans, label2id):
    encoding = tokenizer(
        example["prompt"],
        return_offsets_mapping=True,
        truncation=True,
    )
    labels = ["O"] * len(encoding["offset_mapping"])
    for ent in spans[idx]:
        for i, (s, e) in enumerate(encoding["offset_mapping"]):
            if s >= ent["start"] and e <= ent["end"]:
                labels[i] = (
                    f"B-{ent['label']}"
                    if s == ent["start"]
                    else f"I-{ent['label']}"
                )
    encoding["labels"] = [label2id.get(l, 0) for l in labels]
    return encoding

# === TRAINING FUNCTION ===
def train_ner(
    tokenizer_dir: str,
    output_dir: str,
    prompt_path: str,
    span_path: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    max_len: int = 128,
):
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(
        tokenizer_dir, add_prefix_space=True
    )

    # Load data & build labels
    prompts, spans = load_data(prompt_path, span_path)
    label_list, label2id, id2label = build_label_maps()

    # Build Dataset
    raw_ds = Dataset.from_dict({"prompt": prompts})
    tokenized_ds = raw_ds.map(
        lambda ex, idx: tokenize_and_align(ex, idx, tokenizer, spans, label2id),
        with_indices=True,
    )
    tokenized_ds.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Load / fine-tune model
    model = RobertaForTokenClassification.from_pretrained(
        config["tokenizers"]["base"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    print("🚀 Starting NER training...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ NER model saved to '{output_dir}'")

# === MAIN ENTRYPOINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER with Roberta")

    # pull defaults from config.yaml if flags not passed
    parser.add_argument(
        "--tokenizer_dir",
        default=config["paths"]["tokenizer_dir"],
        help="Path to custom tokenizer directory",
    )
    parser.add_argument(
        "--output_dir",
        default=config["paths"]["ner_output"],
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--prompt_path",
        default=config["paths"]["prompts"],
        help="Path to input prompts CSV",
    )
    parser.add_argument(
        "--span_path",
        default=config["paths"]["span_path"],
        help="Path to entity spans JSONL",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size",
    )

    args = parser.parse_args()

    # resolve everything against project root
    tokenizer_dir = resolve_path(args.tokenizer_dir)
    output_dir    = resolve_path(args.output_dir)
    prompt_path   = resolve_path(args.prompt_path)
    span_path     = resolve_path(args.span_path)

    train_ner(
        tokenizer_dir=tokenizer_dir,
        output_dir=output_dir,
        prompt_path=prompt_path,
        span_path=span_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
