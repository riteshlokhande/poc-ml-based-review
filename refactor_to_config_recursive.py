#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : refactor_to_config_recursive.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import os
import re

CONFIG_BLOCK = '''import yaml
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()\n\n'''

REPLACEMENTS = {
    r'"data/"': 'config["paths"]["data_dir"]',
    r'"models/ner_model.pt"': 'config["models"]["ner_model"]',
    r'"roberta-base"': 'config["tokenizers"]["base"]',
    r'"custom-tokenizer-v2"': 'config["tokenizers"]["custom_hf"]',
    r'0\.90': 'config["thresholds"]["drift_score"]'
}

CHANGE_LOG = []

def refactor_script(script_path):
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(script_path, "r", encoding="latin-1") as f:
            content = f.read()

    original_content = content
    injected = False

    # Inject config block if missing
    if "load_config" not in content:
        content = CONFIG_BLOCK + content
        injected = True

    # Replace hardcoded values
    for pattern, replacement in REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)
        CHANGE_LOG.append({
            "file": script_path,
            "config_injected": injected,
            "replacements": [k for k in REPLACEMENTS if re.search(k, original_content)]
        })


def scan_directory(root="."):
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".py") and file != "refactor_to_config_recursive.py":
                refactor_script(os.path.join(dirpath, file))

def write_report():
    with open("refactor_report.txt", "w") as f:
        for entry in CHANGE_LOG:
            f.write(f"✅ {entry['file']}\n")
            if entry["config_injected"]:
                f.write("  - Injected config block\n")
            for r in entry["replacements"]:
                f.write(f"  - Replaced: {r}\n")
            f.write("\n")

def main():
    scan_directory()
    write_report()
    print(f"\n📝 Refactor complete. See 'refactor_report.txt' for details.")

if __name__ == "__main__":
    main()
