#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : stage0_generate_inputs.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import os
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

# === Synthetic Generators ===
def generate_payment_id(i): return f"PAYMENT-{i:06d}"
def generate_account_number(): return f"XXXXXXXX{random.randint(1000, 9999)}"
def generate_payment_date(): return (datetime(2025, 6, 1) + timedelta(days=random.randint(0, 60))).strftime("%Y-%m-%d")
def generate_validation_dates(payment_date):
    dt = datetime.strptime(payment_date, "%Y-%m-%d")
    v1 = dt - timedelta(days=random.randint(1, 5))
    v2 = v1 - timedelta(days=random.randint(1, 3))
    return v1.strftime("%Y%m%d"), v2.strftime("%Y%m%d")
def generate_outcomes(level, accuracy=0.95):
    passed = lambda: np.random.rand() < accuracy
    if level == "Single": return "PASSED" if passed() else "FAILED", ""
    else: return ("PASSED" if passed() else "FAILED", "PASSED" if passed() else "FAILED")

# === Labeling Logic for Q1/Q2 ===
def label_q1(amount, level):
    return "Yes" if amount > 5000 and level == "Dual" else "No"

def label_q2(level, o1, o2):
    if level == "Single":
        return "Yes" if o1 == "PASSED" else "No"
    else:
        return "Yes" if o1 == "PASSED" and o2 == "PASSED" else "No"

# === Core DataFrame Builder ===
def make_payment_df(n=1000, edge_case_ratio=0.03):
    rows, high_value_indices = [], []
    for i in range(n):
        pid = generate_payment_id(i)
        acc = generate_account_number()
        amt = random.randint(100, 10000)
        lvl = "Dual" if amt > 5000 else "Single"
        pdate = generate_payment_date()
        v1date, v2date = generate_validation_dates(pdate)
        if amt > 5000: high_value_indices.append(i)
        o1, o2 = generate_outcomes(lvl)
        q1 = label_q1(amt, lvl)
        q2 = label_q2(lvl, o1, o2)
        rows.append({
            "Payment ID": pid, "Account Number": acc, "Amount": amt,
            "Payment Date": pdate, "Validation Level": lvl,
            "First Validation Completion Date": v1date,
            "Second Validation Completion Date": v2date if lvl == "Dual" else "",
            "First Validation Outcome": o1,
            "Second Validation Outcome": o2,
            "Q1 Label": q1,
            "Q2 Label": q2,
            "Edge Case": "NO"
        })
    # Inject edge cases
    for idx in random.sample(high_value_indices, int(len(high_value_indices) * edge_case_ratio)):
        rows[idx]["Validation Level"] = "Single"
        rows[idx]["Second Validation Completion Date"] = ""
        rows[idx]["Second Validation Outcome"] = ""
        rows[idx]["Edge Case"] = "YES"
        rows[idx]["Q1 Label"] = label_q1(rows[idx]["Amount"], "Single")
        rows[idx]["Q2 Label"] = label_q2("Single", rows[idx]["First Validation Outcome"], "")
    return pd.DataFrame(rows)

# === Prompt + Entity Span Builder ===
def build_prompt_df(payments_df):
    prompts, spans, labels = [], [], []
    for _, p in payments_df.iterrows():
        prompt = (
            f"Payment {p['Payment ID']} of ₹{p['Amount']} from account {p['Account Number']} "
            f"was processed on {p['Payment Date']} with {p['Validation Level']} validation.\n"
            f"First validation completed on {p['First Validation Completion Date']} with outcome: {p['First Validation Outcome']}.\n"
        )
        if p["Validation Level"] == "Dual":
            prompt += f"Second validation completed on {p['Second Validation Completion Date']} with outcome: {p['Second Validation Outcome']}.\n"
        prompt += (
            "\nQ1: Is Dual validation applied when the payment amount is above ₹5000?\n"
            "Q2: Did all applicable validations passed for this payment?"
        )
        prompts.append(prompt)

        # Entity spans
        span_list = []
        span_list.append({"start": prompt.find(p["Payment ID"]), "end": prompt.find(p["Payment ID"]) + len(p["Payment ID"]), "label": "PAYMENT_ID"})
        amt_str = f"₹{p['Amount']}"
        span_list.append({"start": prompt.find(amt_str), "end": prompt.find(amt_str) + len(amt_str), "label": "AMOUNT"})
        span_list.append({"start": prompt.find(p["Account Number"]), "end": prompt.find(p["Account Number"]) + len(p["Account Number"]), "label": "ACCOUNT"})
        span_list.append({"start": prompt.find(p["Payment Date"]), "end": prompt.find(p["Payment Date"]) + len(p["Payment Date"]), "label": "DATE"})
        spans.append(span_list)

        labels.append({
            "Payment ID": p["Payment ID"],
            "Q1 Label": p["Q1 Label"],
            "Q2 Label": p["Q2 Label"]
        })

    return pd.DataFrame({"prompt": prompts}), spans, pd.DataFrame(labels)

def generate_contrastive_pairs(df, prompt_df):
    pairs = []
    for i, row in df.iterrows():
        anchor = prompt_df.iloc[i]["prompt"]
        amt = row["Amount"]
        lvl = row["Validation Level"]
        o1, o2 = row["First Validation Outcome"], row["Second Validation Outcome"]
        pdate = row["Payment Date"]
        acc = row["Account Number"]
        pid = row["Payment ID"]

        # === Q1 Hard Negative ===
        flipped_lvl = "Single" if lvl == "Dual" else "Dual"
        q1_neg = anchor.replace(f"with {lvl} validation", f"with {flipped_lvl} validation")

        # === Q2 Hard Negative ===
        if lvl == "Single":
            flipped_o1 = "FAILED" if o1 == "PASSED" else "PASSED"
            q2_neg = anchor.replace(f"outcome: {o1}", f"outcome: {flipped_o1}")
        else:
            flipped_o2 = "FAILED" if o2 == "PASSED" else "PASSED"
            q2_neg = anchor.replace(
                f"Second validation completed on {row['Second Validation Completion Date']} with outcome: {o2}",
                f"Second validation completed on {row['Second Validation Completion Date']} with outcome: {flipped_o2}"
            )

        pairs.append({
            "Payment ID": pid,
            "anchor": anchor,
            "hard_negative_q1": q1_neg,
            "hard_negative_q2": q2_neg
        })

    with open("./data/contrastive_pairs.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print("🧪 Generated contrastive_pairs.jsonl with hard negatives for Q1 and Q2")


# === Main Execution ===
def main():
    os.makedirs("./data", exist_ok=True)
    df = make_payment_df()
    prompt_df, span_list, label_df = build_prompt_df(df)

    # Save prompts
    prompt_df["Payment ID"] = df["Payment ID"]
    prompt_df.to_csv("./data/input_prompts.csv", index=False)
    df.to_csv("./data/payment_metadata.csv", index=False)

    # Save entity spans
    with open("./data/entity_spans.jsonl", "w") as f:
        for spans in span_list:
            f.write(json.dumps(spans) + "\n")

    # Save Q1/Q2 labels
    label_df.to_csv("./data/qa_labels.csv", index=False)

    # Generate contrastive pairs
    generate_contrastive_pairs(df, prompt_df)

    print(
        "✅ Generated input_prompts.csv, payment_metadata.csv, entity_spans.jsonl, qa_labels.csv, and contrastive_pairs.jsonl")

if __name__ == "__main__":
    main()
