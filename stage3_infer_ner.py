#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : stage3_infer_ner.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import pandas as pd
import argparse
import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, pipeline

def visualize_entities(text, ents):
    spans = sorted(ents, key=lambda x: x["start"])
    result = ""
    last_idx = 0
    for ent in spans:
        label = ent.get("entity_group", ent.get("entity", ""))
        result += text[last_idx:ent["start"]]
        result += f"[{text[ent['start']:ent['end']]}|{label}]"
        last_idx = ent["end"]
    result += text[last_idx:]
    return result

def run_dynamic_ner(prompt_csv, model_dir, relevance_threshold=0.5, mode="simple"):
    df = pd.read_csv(prompt_csv)

    # === Load custom tokenizer and model ===
    print(f"📦 Loading model and tokenizer from '{model_dir}'...")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(model_dir)

    # === Mode: Aggregated entity spans ===
    if mode == "simple":
        ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        def extract_entities(txt):
            ents = ner_pipe(txt)
            return [
                {
                    "text": txt[ent["start"]:ent["end"]],
                    "label": ent["entity_group"],
                    "score": round(ent["score"], 3),
                    "start": ent["start"],
                    "end": ent["end"]
                }
                for ent in ents if ent["score"] >= relevance_threshold
            ]

        df["NER_Entities"] = df["prompt"].apply(extract_entities)
        df["NER_Overlay"] = df.apply(lambda row: visualize_entities(row["prompt"], row["NER_Entities"]), axis=1)

    # === Mode: Raw token-level predictions ===
    elif mode == "none":
        ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none")

        def extract_raw_entities(text):
            ents = ner_pipe(text)
            return [
                {
                    "text": ent["word"],
                    "label": ent["entity"],
                    "score": round(ent["score"], 3),
                    "start": ent["start"],
                    "end": ent["end"]
                }
                for ent in ents if ent["score"] >= relevance_threshold
            ]

        df["NER_Entities"] = df["prompt"].apply(extract_raw_entities)
        df["NER_Overlay"] = df.apply(lambda row: visualize_entities(row["prompt"], row["NER_Entities"]), axis=1)

    # === Mode: Manual token alignment and logits inspection ===
    elif mode == "manual":
        for text in df["prompt"]:
            encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            offsets = encoding["offset_mapping"]

            inputs = {k: torch.tensor([v]) for k, v in encoding.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)[0].tolist()
            labels = [model.config.id2label[p] for p in preds]

            print(f"\n📝 Prompt: {text}")
            print("🔍 Token Predictions:")
            for tok, lab, (start, end) in zip(tokens, labels, offsets):
                span = text[start:end]
                print(f"{tok:15} → {lab:10} | '{span}'")

        return

    # === Benchmarking: Entity coverage across prompts ===
    def benchmark_coverage(df):
        total = len(df)
        with_entity = df["NER_Entities"].apply(lambda x: len(x) > 0).sum()
        print(f"\n📊 Entity Coverage: {with_entity}/{total} prompts ({(with_entity/total)*100:.2f}%)")

    benchmark_coverage(df)
    df.to_csv("./data/ner_output_dynamic.csv", index=False)
    print("✅ Dynamic NER complete. Saved to ./data/ner_output_dynamic.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_csv", default="data/input_prompts.csv")
    parser.add_argument("--model_dir", default="models/ner")
    parser.add_argument("--mode", choices=["simple", "none", "manual"], default="simple")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_dynamic_ner(args.prompt_csv, args.model_dir, relevance_threshold=args.threshold, mode=args.mode)
