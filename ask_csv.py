#!/usr/bin/env python3
"""Command-line interface for asking questions about a CSV dataset."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from analysis import analyze
from csv_profile import build_profile
from llm_client import DEFAULT_MODEL, LLM, LLMError
from query_engine import Engine

CSV_PATH = Path(__file__).resolve().with_name("data.csv")


def print_table(table, limit=20):
    if not table or not table["rows"]:
        print("(no rows)")
        return
    frame = pd.DataFrame(table["rows"], columns=table["columns"])
    print(frame.head(limit).to_string(index=False))


def print_result(result):
    """Print an analyze() result in the same layout the CLI has always used."""
    kind = result["kind"]
    if kind == "profile":
        print("\nDATASET PROFILE:")
        print(result["profile"])
        print("\nDATA QUALITY:")
        for note in result["notes"][:2]:
            print(f"- {note}")
        print(f"\n{result['notes'][2]}")
        return
    if kind == "churn":
        print("\nCUSTOMER CHURN: ASSOCIATIONS, NOT CAUSES")
        print(result["answer"])
        for table in result["tables"]:
            print(f"\n{table['title'].upper()}:")
            print_table(table, limit=None)
        for note in result["notes"]:
            print(f"\n{note}")
        return
    if kind == "unanswerable":
        print(f"\nCANNOT ANSWER: {result['answer']}")
        return
    if kind == "failed":
        print(f"\nFAILED: {result['answer']}")
        return

    print(f"\nANSWER:\n{result['answer']}")
    for note in result["notes"]:
        print(note)
    if result["chart_path"]:
        print(f"\nBAR CHART: saved to {result['chart_path']}")
    elif result["chart_error"]:
        print(f"\nBAR CHART: {result['chart_error']}")
    print(f"\nSQL:\n{result['sql']}")
    print(f"\nRESULT ({result['row_count']} rows):")
    print_table(result["table"])
    for title, items in (("ASSUMPTIONS", result["assumptions"]), ("WARNINGS", result["warnings"])):
        if items:
            print(f"\n{title}:\n- " + "\n- ".join(items))
    print(
        "\nNUMERIC GROUNDING: "
        + ("answer numbers match the result table" if result["grounded"] else "UNVERIFIED - check the table")
    )
    print("NOTE: Numeric grounding checks the answer against the query result; it does not validate the SQL logic.")
    print(f"TIME: {result['elapsed_s']:.1f}s | {result['llm_report']}")


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
