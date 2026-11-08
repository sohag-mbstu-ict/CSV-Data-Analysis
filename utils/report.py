"""Plain-text report of an analyze() result, shared by the CLI and the Django server log."""

import pandas as pd


def table_text(table, limit=20):
    """Format a {"columns", "rows"} table as aligned text."""
    if not table or not table["rows"]:
        return "(no rows)"
    frame = pd.DataFrame(table["rows"], columns=table["columns"])
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_string(index=False)


def format_result(result):
    """Return an analyze() result as the multi-line text the CLI prints."""
    kind = result.get("kind")
    lines = []

    if kind == "profile":
        lines += ["DATASET PROFILE:", result["profile"], "", "DATA QUALITY:"]
        lines += [f"- {note}" for note in result["notes"][:2]]
        lines += ["", result["notes"][2]]
        return "\n".join(lines)
    if kind == "churn":
        lines += ["CUSTOMER CHURN: ASSOCIATIONS, NOT CAUSES", result["answer"]]
        for table in result["tables"]:
            lines += ["", f"{table['title'].upper()}:", table_text(table, limit=None)]
        for note in result["notes"]:
            lines += ["", note]
        return "\n".join(lines)
    if kind == "unanswerable":
        return f"CANNOT ANSWER: {result['answer']}"
    if kind == "failed":
        return f"FAILED: {result['answer']}"
    if kind == "error":
        return f"ERROR: {result['answer']}"
    if kind in ("chat", "clarify"):
        lines += [result["answer"]]
        lines += [f"  - {example}" for example in result.get("suggestions") or []]
        return "\n".join(lines)

    lines += ["ANSWER:", result["answer"]]
    lines += result["notes"]
    if result["chart_path"]:
        lines += ["", f"BAR CHART: saved to {result['chart_path']}"]
    elif result["chart_error"]:
        lines += ["", f"BAR CHART: {result['chart_error']}"]
    lines += ["", "SQL:", result["sql"]]
    lines += ["", f"RESULT ({result['row_count']} rows):", table_text(result["table"])]
    for title, items in (("ASSUMPTIONS", result["assumptions"]), ("WARNINGS", result["warnings"])):
        if items:
            lines += ["", f"{title}:"] + [f"- {item}" for item in items]
    lines += [
        "",
        "VERIFIED: " + ("YES - answer numbers match the result table" if result["grounded"]
                        else "NO - check the table"),
        f"TIME: {result['elapsed_s']:.1f}s | {result['llm_report']}",
    ]
    return "\n".join(lines)
