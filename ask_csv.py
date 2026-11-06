#!/usr/bin/env python3
"""Command-line interface for asking questions about a CSV dataset."""

import argparse
import re
import time
from pathlib import Path

from csv_profile import build_profile, is_dataset_description_request
from llm_client import DEFAULT_MODEL, LLM, question_to_sql
from query_engine import Engine, check_sql
from result_grounding import write_answer
from visualization import save_bar_chart, wants_bar_chart

CSV_PATH = Path(__file__).resolve().with_name("data.csv")
MAX_REPAIRS = 2
SHARE_REQUEST = re.compile(
    r"\b(share|portion|percentage|percent)\b.*\b(total|overall)\b|"
    r"\b(total|overall)\b.*\b(share|portion|percentage|percent)\b",
    re.I,
)
CHURN_WHY_REQUEST = re.compile(
    r"\bwhy\b.*\b(churn|churned|churning)\b|"
    r"\b(reason|cause|factor|driver)s?\b.*\b(churn|churned|churning)\b",
    re.I,
)


def add_share_context(question, sql, result, engine):
    """Add verified amounts and missing-category impact to share-of-total results."""
    if not SHARE_REQUEST.search(question) or len(result) != 1:
        return None

    types = engine.column_types()
    numeric_types = ("INT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "HUGEINT")
    numeric_columns = [
        column for column, type_name in types.items()
        if any(type_name.upper().startswith(numeric_type) for numeric_type in numeric_types)
    ]

    sum_expressions = re.findall(r"\bSUM\s*\(([^()]*)\)", sql, re.I | re.S)
    measures = []
    for expression in sum_expressions:
        for column in numeric_columns:
            if re.search(rf'(?<![\w])"?{re.escape(column)}"?(?![\w])', expression, re.I):
                if column not in measures:
                    measures.append(column)
    question_words = set(re.findall(r"[a-z0-9_]+", question.lower()))
    measure_column = next(
        (column for column in measures if set(re.findall(r"[a-z0-9_]+", column.lower())) & question_words),
        measures[0] if measures else None,
    )
    if measure_column is None:
        return None

    # Find the quoted category comparison in the generated WHERE/CASE expression.
    predicates = re.findall(
        r'(?:(?:"([^"]+)"|([A-Za-z_]\w*))\s*=\s*\'((?:[^\']|\'\')*)\')',
        sql,
        re.I,
    )
    category_column = category_value = None
    for quoted_column, bare_column, literal in predicates:
        column = quoted_column or bare_column
        if column not in types or column in numeric_columns:
            continue
        value = literal.replace("''", "'")
        identifier = '"' + column.replace('"', '""') + '"'
        exists = engine.con.execute(
            f"SELECT 1 FROM data WHERE {identifier} = ? LIMIT 1",
            [value],
        ).fetchone()
        if exists:
            category_column, category_value = column, value
            break
    if category_column is None:
        return None

    summary = engine.share_summary(category_column, category_value, measure_column)
    if summary["share_pct"] is None:
        return None

    # Replace the model's share value with a calculation whose denominator includes NULL categories.
    share_columns = [
        column for column in result.columns
        if re.search(r"share|percent|pct", str(column), re.I)
    ]
    for column in share_columns:
        result.loc[result.index[0], column] = summary["share_pct"]
    if not share_columns:
        result["verified_share_pct"] = summary["share_pct"]

    result["share_pct"] = summary["share_pct"]
    result["category_amount"] = summary["category_amount"]
    result["overall_amount"] = summary["overall_amount"]
    result["unassigned_category_amount"] = summary["unassigned_amount"]
    result["unassigned_category_rows"] = summary["unassigned_rows"]
    return {
        "category_column": category_column,
        "category_value": category_value,
        "measure_column": measure_column,
        **summary,
    }


def show_churn_associations(engine):
    """Print customer-level churn associations with a clear non-causal caveat."""
    cohorts, segments = engine.churn_associations()
    by_status = {bool(row.is_churned): row for row in cohorts.itertuples(index=False)}
    active = by_status.get(False)
    churned = by_status.get(True)
    print("\nCUSTOMER CHURN: ASSOCIATIONS, NOT CAUSES")
    if active is not None and churned is not None:
        print(
            f"Compared across {churned.customers} churned and {active.customers} active customers, "
            f"average satisfaction was {churned.avg_satisfaction_per_customer:.2f} for churned "
            f"customers and {active.avg_satisfaction_per_customer:.2f} for active customers. "
            f"Average support tickets per order per customer were "
            f"{churned.avg_tickets_per_order_per_customer:.2f} versus "
            f"{active.avg_tickets_per_order_per_customer:.2f}. "
            f"Average ages were {churned.avg_customer_age:.2f} and {active.avg_customer_age:.2f}, respectively."
        )
    print("\nCUSTOMER SEGMENT CHURN RATES:")
    print(segments.to_string(index=False) if not segments.empty else "(no segment data)")
    print(
        "\nThese are patterns associated with churn in this CSV, not proof of why customers churn. "
        "The dataset is observational and cannot establish causation."
    )


def ask(question, llm, engine, profile_text, missing, duplicates):
    """Run the SQL, explanation, numeric verification, and optional chart pipeline."""
    start = time.time()
    if is_dataset_description_request(question):
        missing_cells = sum(missing.values())
        print("\nDATASET PROFILE:")
        print(profile_text)
        print("\nDATA QUALITY:")
        print(f"- Exact duplicate rows: {duplicates}")
        print(f"- Missing cells across all columns: {missing_cells}")
        print("\nThis profile is computed directly from the CSV; no SQL or model interpretation is needed.")
        return
    if CHURN_WHY_REQUEST.search(question):
        show_churn_associations(engine)
        return

    plan = question_to_sql(llm, profile_text, question)

    if plan.get("answerable") is False:
        print(f"\nCANNOT ANSWER: {plan.get('reason') or 'this dataset does not contain the needed data.'}")
        return

    sql, result, truncated = plan["sql"], None, False
    error = plan.get("error")
    for attempt in range(MAX_REPAIRS + 1):
        if sql:
            try:
                check_sql(sql)
                result, truncated = engine.run(sql)
                break
            except Exception as exc:
                error = str(exc).split("\n")[0]
        else:
            error = error or "no query was produced"

        if attempt < MAX_REPAIRS:
            print(f"  (query failed: {error[:120]} - asking the model to fix it)")
            plan = question_to_sql(
                llm,
                profile_text,
                question,
                failed_sql=sql or "(none)",
                error=error,
            )
            if plan.get("answerable") is False:
                print(f"\nCANNOT ANSWER: {plan.get('reason')}")
                return
            sql, error = plan["sql"], plan.get("error")

    if result is None:
        print(f"\nFAILED: could not compute a reliable answer (last error: {error}). No numbers are reported.")
        return

    share_context = add_share_context(question, sql, result, engine)

    assumptions = list(plan.get("assumptions") or [])
    if duplicates and not re.search(r"\bdata_raw\b", sql, re.I):
        assumptions.append(f"{duplicates} exact duplicate rows were removed before analysis.")
    warnings = [
        f'Column "{column}" has {count} missing values (excluded where used).'
        for column, count in missing.items()
        if count and re.search(rf'"{re.escape(column)}"|\b{re.escape(column)}\b', sql)
    ]
    if truncated:
        warnings.append("Result truncated to 200 rows.")
    if share_context and share_context["unassigned_rows"]:
        warnings.append(
            f"{share_context['unassigned_rows']} rows have no "
            f'"{share_context["category_column"]}" value; their '
            f'{share_context["unassigned_amount"]:,.2f} in '
            f'"{share_context["measure_column"]}" is included in the overall total but cannot be attributed to a category.'
        )

    answer, grounded = write_answer(llm, question, result, assumptions)
    chart_path, chart_error = None, None
    if wants_bar_chart(question):
        try:
            chart_path, chart_error = save_bar_chart(result, question)
        except Exception as exc:
            chart_error = f"Could not create the chart: {exc}"

    print(f"\nANSWER:\n{answer}")
    if share_context:
        print(
            "The share percentage above was independently recalculated using the full dataset total; "
            "the SQL shown below is the model-generated query."
        )
        print(
            "\nVERIFIED SHARE DETAILS: "
            f'{share_context["category_value"]} '
            f'{share_context["measure_column"]} = {share_context["category_amount"]:,.2f}; '
            f'overall {share_context["measure_column"]} = {share_context["overall_amount"]:,.2f}; '
            f'share = {share_context["share_pct"]:.2f}%.'
        )
        if share_context["unassigned_rows"]:
            print(
                f'{share_context["unassigned_rows"]} rows have no '
                f'{share_context["category_column"]} value '
                f'({share_context["unassigned_amount"]:,.2f} '
                f'{share_context["measure_column"]}); this amount is included in the denominator '
                "but cannot be attributed to a category."
            )
    if chart_path:
        print(f"\nBAR CHART: saved to {chart_path}")
    elif chart_error:
        print(f"\nBAR CHART: {chart_error}")
    print(f"\nSQL:\n{sql}")
    print(f"\nRESULT ({len(result)} rows):")
    print(result.head(20).to_string(index=False) if len(result) else "(no rows)")
    for title, items in (("ASSUMPTIONS", assumptions), ("WARNINGS", warnings)):
        if items:
            print(f"\n{title}:\n- " + "\n- ".join(items))
    print(
        f"\nNUMERIC GROUNDING: {'answer numbers match the result table' if grounded else 'UNVERIFIED - check the table'}"
    )
    print("NOTE: Numeric grounding checks the answer against the query result; it does not validate the SQL logic.")
    print(f"TIME: {time.time() - start:.1f}s | {llm.speed_report()}")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a CSV with a local LLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-prompts", action="store_true", help="print the prompts sent to the model")
    args = parser.parse_args()

    profile_text, missing, duplicates = build_profile(CSV_PATH)
    engine = Engine(CSV_PATH)
    llm = LLM(args.model, args.show_prompts)
    print(f"Loaded {CSV_PATH}. Model: {args.model}")
    print("Enter a question, or type 'exit' to quit.")
    try:
        while True:
            try:
                question = input("\nEnter query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() == "exit":
                break
            if question:
                ask(question, llm, engine, profile_text, missing, duplicates)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
