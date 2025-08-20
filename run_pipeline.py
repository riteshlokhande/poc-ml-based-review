#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : run_pipeline.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
import os

# === Load config ===
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

# === Import modules ===
from stage0_generate_inputs import main as generate_inputs
from stage1_train_ner import train_ner
from stage2_train_sentiment import train_sentiment
from stage3_infer_ner import run_dynamic_ner
from stage4_infer_sentiment import run_sentiment_inference
from composite_model import run_composite_inference  # NEW
from stage5_model_logic_mapper import apply_model_logic

def run_pipeline():
    # === Setup directories ===
    for key in ["data_dir", "model_dir", "tokenizer_dir", "output_dir"]:
        os.makedirs(config["paths"].get(key, "data"), exist_ok=True)

    # === Step 1: Generate synthetic prompts ===
    print("\n🔄 Step 1: Generating synthetic inputs...")
    generate_inputs()

    # === Step 2: Train NER model ===
    print("\n🧠 Step 2: Training NER model...")
    train_ner(
        tokenizer_dir=config["paths"]["tokenizer_dir"],
        output_dir=config["models"]["ner_model"],
        prompt_path=config["paths"]["prompts"],
        span_path=config["paths"]["span_path"]
    )

    # === Step 3: Train Sentiment model ===
    print("\n💬 Step 3: Training Sentiment model...")
    train_sentiment(
        tokenizer_dir=config["paths"]["tokenizer_dir"],
        output_dir=config["models"]["sentiment_model"]
    )

    # === Step 4: Run NER inference ===
    print("\n🔍 Step 4: Running NER inference...")
    run_dynamic_ner(
        prompt_csv=config["paths"]["prompts"],
        model_dir=config["models"]["ner_model"],
        relevance_threshold=config["thresholds"].get("ner_relevance", 0.0),
        mode="simple"
    )

    # === Step 5: Run Sentiment inference ===
    print("\n📈 Step 5: Running Sentiment inference...")
    run_sentiment_inference(
        prompt_csv=config["paths"]["prompts"],
        model_dir=config["models"]["sentiment_model"]
    )

    # === Step 6: Composite model inference ===
    print("\n🧩 Step 6: Running Composite Model Inference...")
    run_composite_inference(
        ner_csv=config["paths"]["ner_output"],
        sentiment_csv=config["paths"]["sentiment_output"],
        output_csv=config["paths"]["composite_output"]
    )

    # === Step 7: Apply business logic ===
    print("\n🧮 Step 7: Applying model-based business logic...")
    apply_model_logic(
        ner_csv=config["paths"]["ner_output"],
        sentiment_csv=config["paths"]["sentiment_output"],
        output_csv=config["paths"]["final_output"]
    )

    print("\n✅ Pipeline complete. Final predictions saved to", config["paths"]["final_output"])

if __name__ == "__main__":
    run_pipeline()
