#!/usr/bin/env python3
"""Command-line interface for asking questions about a CSV dataset."""

import argparse
import sys
from pathlib import Path

# Allow both `python utils/ask_csv.py` and `python -m utils.ask_csv` from the project root.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.analysis import analyze  # noqa: E402
from utils.csv_profile import build_profile
from utils.llm_client import DEFAULT_MODEL, LLM, LLMError
from utils.query_engine import Engine
from utils.report import format_result

CSV_PATH = Path(__file__).resolve().parent.parent / "data.csv"


def print_result(result):
    """Print an analyze() result in the shared report layout."""
    print("\n" + format_result(result))


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a CSV with a local LLM.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="CSV file to analyze (default: data.csv)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-prompts", action="store_true", help="print the prompts sent to the model")
    args = parser.parse_args()

    try:
        profile_text, missing, duplicates = build_profile(args.csv)
    except (OSError, ValueError) as exc:
        sys.exit(f"Could not read {args.csv}: {exc}")
    engine = Engine(args.csv)
    print(f"Loaded {args.csv}. Model: {args.model}")
    print("Enter a question, or type 'exit' to quit.")
    try:
        while True:
            try:
                question = input("\nEnter query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() == "exit":
                break
            if not question:
                continue
            # A fresh client per question keeps the speed report per question, like the web app.
            llm = LLM(args.model, args.show_prompts)
            try:
                result = analyze(
                    question, llm, engine, profile_text, missing, duplicates,
                    on_status=lambda message: print(f"  ({message})"),
                )
            except LLMError as exc:
                sys.exit(str(exc))
            print_result(result)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
