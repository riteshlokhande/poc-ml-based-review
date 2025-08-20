#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : stage4_infer_sentiment.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline
import pandas as pd

def run_sentiment_inference(prompt_csv, model_dir, max_len=512):
    df = pd.read_csv(prompt_csv)

    # === Load model and tokenizer ===
    print(f"📦 Loading sentiment model from '{model_dir}'...")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, add_prefix_space=True)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # === Inference function ===
    def get_sentiment(text):
        result = sentiment_pipe(text[:max_len])[0]
        return {
            "label": result["label"],
            "score": round(result["score"], 3)
        }

    # === Apply inference ===
    df["Sentiment_Result"] = df["prompt"].apply(get_sentiment)
    df["Sentiment_Label"] = df["Sentiment_Result"].apply(lambda x: x["label"])
    df["Sentiment_Score"] = df["Sentiment_Result"].apply(lambda x: x["score"])

    # === Optional: Distribution diagnostics ===
    sentiment_counts = df["Sentiment_Label"].value_counts()
    print("\n📊 Sentiment Distribution:")
    for label, count in sentiment_counts.items():
        print(f"{label}: {count} ({(count/len(df))*100:.2f}%)")

    # === Save output ===
    df.to_csv("./data/sentiment_output.csv", index=False)
    print("✅ Sentiment inference complete. Saved to ./data/sentiment_output.csv")

if __name__ == "__main__":
    run_sentiment_inference("data/input_prompts.csv", "models/sentiment")
