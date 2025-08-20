#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : refactor_to_config.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import os
import re

CONFIG_BLOCK = '''import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()\n\n'''

REPLACEMENTS = {
    r'config["paths"]["data_dir"]': 'config["paths"]["data_dir"]',
    r'config["models"]["ner_model"]': 'config["models"]["ner_model"]',
    r'config["tokenizers"]["base"]': 'config["tokenizers"]["base"]',
    r'config["tokenizers"]["custom_hf"]': 'config["tokenizers"]["custom_hf"]',
    r'0\.90': 'config["thresholds"]["drift_score"]'
}

def refactor_script(script_path):
    with open(script_path, "r") as f:
        content = f.read()

    # Inject config block if missing
    if "load_config" not in content:
        content = CONFIG_BLOCK + content

    # Replace hardcoded values
    for pattern, replacement in REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content)

    with open(script_path, "w") as f:
        f.write(content)
    print(f"✅ Refactored: {script_path}")

def main():
    for file in os.listdir("."):
        if file.endswith(".py") and file != "refactor_to_config.py":
            refactor_script(file)

if __name__ == "__main__":
    main()
