#!/usr/bin/env python3
# add_header.py
# Utility to batch-prepend a header to all .py files in a directory tree

import argparse
import sys
from pathlib import Path
from datetime import datetime
from dateutil import tz

# === CONFIGURE YOUR HEADER HERE ===
# You can embed placeholders like {filename}, {date}, {author}.
HEADER_TEMPLATE = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
File       : {filename}
Author     : Ritesh
Created    : {date}
Description: <SHORT DESCRIPTION HERE>
\"\"\"

"""

def build_header(path: Path, author="Ritesh"):
    """
    Fill in the header template for a given file.
    """
    today = datetime.now(tz=tz.tzlocal()).strftime("%Y-%m-%d")
    return HEADER_TEMPLATE.format(
        filename=path.name,
        date=today,
        author=author
    )

def has_header(text: str, header_start: str) -> bool:
    """
    Simple check: does the file already start with the header?
    """
    return text.startswith(header_start)

def prepend_header(py_file: Path, author: str) -> bool:
    """
    Read the file, and if the header isn't present, prepend it.
    Returns True if header was added.
    """
    text = py_file.read_text(encoding="utf-8")
    header = build_header(py_file, author=author)
    if has_header(text, header.splitlines()[0]):
        return False
    py_file.write_text(header + text, encoding="utf-8")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Batch-add a script header to all .py files."
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        required=True,
        help="Root directory to scan for .py files"
    )
    parser.add_argument(
        "--author", "-a",
        default="Ritesh",
        help="Author name to inject into the header"
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"❌ Error: {args.dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    added, skipped = 0, 0
    for py_file in args.dir.rglob("*.py"):
        if prepend_header(py_file, author=args.author):
            print(f"➕ Header added to {py_file}")
            added += 1
        else:
            skipped += 1

    print(f"\n✅ Done. Headers added: {added}, skipped (already present): {skipped}")

if __name__ == "__main__":
    main()
