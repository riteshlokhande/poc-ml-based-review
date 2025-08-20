#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : composite_model.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import pandas as pd

def run_composite_inference(ner_csv, sentiment_csv, output_csv):
    print("🔗 Merging NER and Sentiment outputs...")

    # Load predictions
    ner_df = pd.read_csv(ner_csv)
    sentiment_df = pd.read_csv(sentiment_csv)

    # Merge on prompt ID or text
    merged = pd.merge(ner_df, sentiment_df, on="prompt", how="inner")

    # Apply composite logic
    def apply_logic(row):
        entities = row.get("entities", "")
        sentiment = row.get("sentiment", "")
        if "payment" in entities and sentiment == "negative":
            return "flag_for_review"
        elif "approval" in entities and sentiment == "positive":
            return "auto_approve"
        else:
            return "manual_check"

    merged["composite_decision"] = merged.apply(apply_logic, axis=1)

    # Save output
    merged.to_csv(output_csv, index=False)
    print(f"✅ Composite predictions saved to {output_csv}")
