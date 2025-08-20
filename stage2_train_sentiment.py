#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : stage2_train_sentiment.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import argparse, os, torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast, RobertaForSequenceClassification,
    Trainer, TrainingArguments
)

def load_tokenizer(tokenizer_dir):
    print(f"📦 Loading custom tokenizer from '{tokenizer_dir}'...")
    return RobertaTokenizerFast.from_pretrained(tokenizer_dir, add_prefix_space=True)

def preprocess_function(examples, tokenizer, max_len):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_len
    )

def train_sentiment(tokenizer_dir, output_dir, num_epochs=3, batch_size=4, max_len=128):
    # === Step 1: Load SST-2 dataset ===
    dataset = load_dataset("glue", "sst2")
    train_ds = dataset["train"].shuffle(seed=42).select(range(5000))
    val_ds = dataset["validation"]

    # === Step 2: Load tokenizer ===
    tokenizer = load_tokenizer(tokenizer_dir)

    # === Step 3: Load model ===
    model = RobertaForSequenceClassification.from_pretrained(config["tokenizers"]["base"], num_labels=2)

    # === Step 4: Tokenize datasets ===
    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer, max_len), batched=True)
    val_ds = val_ds.map(lambda x: preprocess_function(x, tokenizer, max_len), batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # === Step 5: Training setup ===
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        no_cuda=not torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    # === Step 6: Train and save ===
    print("🚀 Starting sentiment training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Sentiment model saved to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", default="tokenizer", help="Path to custom tokenizer directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save trained model")
    args = parser.parse_args()
    train_sentiment(args.tokenizer_dir, args.output_dir)
