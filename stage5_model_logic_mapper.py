#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : stage5_model_logic_mapper.py
Author     : Ritesh
Created    : 2025-08-18
Description: Applies ML models for Q1 (approval) and Q2 (sentiment) predictions,
             with fallback to manual logic. Outputs raw predictions, a readable CSV,
             and an audit summary.
"""

import re
import yaml
import pandas as pd
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Load Config ===
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()
score_thresh  = config.get("thresholds", {}).get("sentiment_score", 0.5)
amount_thresh = config.get("thresholds", {}).get("amount_limit", 5000)

# === Model Setup ===
q1_model_name = config.get("models", {}).get("q1_classifier")
q2_model_name = config.get("models", {}).get("q2_classifier")

if q1_model_name:
    q1_tokenizer = AutoTokenizer.from_pretrained(q1_model_name)
    q1_model     = AutoModelForSequenceClassification.from_pretrained(q1_model_name)
    q1_mapping   = config.get("label_mapping", {}).get("q1", {"LABEL_0": "No", "LABEL_1": "Yes"})
else:
    q1_model = None

if q2_model_name:
    q2_tokenizer = AutoTokenizer.from_pretrained(q2_model_name)
    q2_model     = AutoModelForSequenceClassification.from_pretrained(q2_model_name)
    q2_mapping   = config.get("label_mapping", {}).get("q2", {"LABEL_NEG": "No", "LABEL_POS": "Yes"})
else:
    q2_model = None

label_map = {
    "LABEL_0": "POSITIVE",
    "LABEL_1": "NEGATIVE",
    "LABEL_2": "NEUTRAL"
}

# === Utilities ===
def sanitize_entity_string(entity_str):
    return re.sub(r'np\.float32\(([\d\.]+)\)', r'\1', entity_str)

def extract_metadata(entity_str):
    entity_str = sanitize_entity_string(entity_str)
    entities = ast.literal_eval(entity_str)
    meta = {"Payment ID": "", "Amount": "", "Account": "", "Date": ""}
    for ent in entities:
        label = ent["label"].upper()
        text = ent["text"]
        if "PAYMENT_ID" in label:
            meta["Payment ID"] = text
        elif "AMOUNT" in label:
            meta["Amount"] = text
        elif "ACCOUNT" in label:
            meta["Account"] = text
        elif "DATE" in label:
            meta["Date"] = text
    return pd.Series(meta)

# === Q1: Approval Prediction ===
def predict_q1_model(prompt_text):
    inputs  = q1_tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True)
    outputs = q1_model(**inputs)
    probs   = torch.softmax(outputs.logits, dim=1)
    idx     = torch.argmax(probs, dim=1).item()
    label   = q1_model.config.id2label[idx]
    return q1_mapping.get(label, "No")

def answer_q1(entity_list, prompt_text):
    # Model-based inference
    if q1_model:
        try:
            return predict_q1_model(prompt_text)
        except Exception as e:
            print(f"⚠️ Q1 model error: {e}")
    # Fallback manual logic
    try:
        entities = ast.literal_eval(sanitize_entity_string(entity_list))
        amount   = next(
            int(ent["text"].replace("₹", "").replace(",", ""))
            for ent in entities
            if ent.get("label", "").upper() == "AMOUNT" and ent["text"].replace("₹","").replace(",","").isdigit()
        )
        return "Yes" if amount > amount_thresh and "dual" in prompt_text.lower() else "No"
    except Exception as e:
        print(f"⚠️ Q1 fallback error: {e}")
        return "No"

# === Q2: Sentiment + Validation Prediction ===
def predict_q2_model(prompt_text):
    inputs  = q2_tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True)
    outputs = q2_model(**inputs)
    probs   = torch.softmax(outputs.logits, dim=1)
    idx     = torch.argmax(probs, dim=1).item()
    label   = q2_model.config.id2label[idx]
    return q2_mapping.get(label, "No")

def answer_q2(label, score, prompt_text):
    # Model-based inference
    if q2_model:
        try:
            return predict_q2_model(prompt_text)
        except Exception as e:
            print(f"⚠️ Q2 model error: {e}")
    # Fallback manual logic
    mapped   = label_map.get(label.upper(), "UNKNOWN")
    passed_v = "PASSED" in prompt_text.upper()
    return "Yes" if mapped in ["POSITIVE", "NEUTRAL"] and score >= score_thresh and passed_v else "No"

# === Audit Summary ===
def generate_audit_summary(df, path="data/summary_report.txt"):
    def prediction_counts(col):
        return df[col].value_counts().to_dict()

    def compute_accuracy(pred_col, true_col):
        return round((df[pred_col] == df[true_col]).mean() * 100, 2)

    def get_mismatches(pred_col, true_col):
        mm = df[df[pred_col] != df[true_col]]
        return mm[[pred_col, true_col]].reset_index(drop=True)

    with open(path, "w") as f:
        f.write("📊 Prediction Summary\n")
        f.write(f"Q1 Predictions → {prediction_counts('Model_Q1')}\n")
        f.write(f"Q2 Predictions → {prediction_counts('Model_Q2')}\n\n")

        if {"Expected_Q1","Expected_Q2"}.issubset(df.columns):
            f.write("✅ Accuracy Report\n")
            f.write(f"Q1 Accuracy → {compute_accuracy('Model_Q1','Expected_Q1')}%\n")
            f.write(f"Q2 Accuracy → {compute_accuracy('Model_Q2','Expected_Q2')}%\n\n")
            f.write("⚠️ Mismatched Predictions\n")
            f.write("Q1 Mismatches:\n")
            f.write(get_mismatches('Model_Q1','Expected_Q1').to_string(index=False))
            f.write("\n\nQ2 Mismatches:\n")
            f.write(get_mismatches('Model_Q2','Expected_Q2').to_string(index=False))
        else:
            f.write("⚠️ Ground truth columns not found. Skipping accuracy and mismatch report.\n")

    print(f"📝 Summary report saved to '{path}'")

# === Main Logic ===
def apply_model_logic(ner_csv, sentiment_csv, output_csv, readable_csv):
    print(f"📥 Loading NER output from '{ner_csv}'")
    ner_df = pd.read_csv(ner_csv)

    print(f"📥 Loading Sentiment output from '{sentiment_csv}'")
    sent_df = pd.read_csv(sentiment_csv)

    df = ner_df.copy()
    df["Sentiment_Label"] = sent_df["Sentiment_Label"]
    df["Sentiment_Score"] = sent_df["Sentiment_Score"]
    df["Model_Q1"] = df.apply(lambda r: answer_q1(r["NER_Entities"], r["prompt"]), axis=1)
    df["Model_Q2"] = df.apply(lambda r: answer_q2(r["Sentiment_Label"], r["Sentiment_Score"], r["prompt"]), axis=1)

    df["Q1_Reason"] = df["Model_Q1"].map(lambda x: "Model=Yes" if x=="Yes" else "Model=No")
    df["Q2_Reason"] = df["Model_Q2"].map(lambda x: "Model=Yes" if x=="Yes" else "Model=No")

    df.to_csv(output_csv, index=False)
    print(f"✅ Raw predictions saved to '{output_csv}'")

    # === Readable Output ===
    meta_df = df["NER_Entities"].apply(extract_metadata)
    readable_df = pd.concat([
        meta_df,
        df[["prompt","Model_Q1","Q1_Reason","Model_Q2","Q2_Reason"]]
    ], axis=1)
    readable_df["Validation Type"] = readable_df["prompt"].str.contains("dual", case=False).map({True:"Dual",False:"Single"})
    readable_df.to_csv(readable_csv, index=False)
    print(f"🧾 Readable predictions saved to '{readable_csv}'")

    # === Audit Summary ===
    generate_audit_summary(df)

# === CLI Entry Point ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner_csv",      default="data/ner_output_dynamic.csv")
    parser.add_argument("--sentiment_csv",default="data/sentiment_output.csv")
    parser.add_argument("--output_csv",   default="data/model_predictions.csv")
    parser.add_argument("--readable_csv", default="data/prediction_readable.csv")
    args = parser.parse_args()

    apply_model_logic(
        args.ner_csv,
        args.sentiment_csv,
        args.output_csv,
        args.readable_csv
    )
