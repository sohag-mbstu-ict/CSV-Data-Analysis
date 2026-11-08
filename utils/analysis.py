"""Question-answering pipeline shared by the command line and the web API.

`analyze()` runs one question end to end and returns a plain dictionary that is
safe to serialize as JSON. It never prints, so callers decide how to show it.
"""

import datetime
import decimal
import math
import re
import time

import numpy as np
import pandas as pd

from utils.csv_profile import is_dataset_description_request
from utils.llm_client import question_to_sql
from utils.query_engine import check_sql
from utils.result_grounding import write_answer
from utils.visualization import save_bar_chart, wants_bar_chart

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
# Whole messages that are greetings, thanks or "what can you do" rather than data questions.
SMALL_TALK = re.compile(
    r"^\s*(?:"
    r"h+i+|hello+|hey+|hiya|yo|howdy|greetings|good\s+(?:morning|afternoon|evening|day)|"
    r"(?:as+alamu\s*alaikum|salam|salaam)|"
    r"thanks?(?:\s+you)?(?:\s+(?:so\s+much|a\s+lot))?|thx|ty|"
    r"bye|goodbye|see\s+you|ok(?:ay)?|cool|nice|great|"
    r"help|what\s+can\s+you\s+do|how\s+does\s+(?:this|it)\s+work|who\s+are\s+you|"
    r"how\s+are\s+you(?:\s+doing)?"
    r")(?:\s+(?:there|all|everyone|again|claude|bot))?\s*[!.?,:)\s]*$",
    re.I,
)
NO_QUERY = "no query was produced"
NUMERIC_TYPE_PREFIXES = (
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
    "UINTEGER", "UBIGINT", "DECIMAL", "DOUBLE", "FLOAT", "REAL",
)
ID_COLUMN = re.compile(r"(?i:(^|[_\s])id)$|[a-z]Id$")
MEASURE_HINTS = ["revenue", "sales", "amount", "profit", "income", "total", "value", "price", "cost"]

# The built-in churn report only works on datasets that have these columns.
CHURN_COLUMNS = {
    "customer_id", "churned", "customer_segment", "customer_age",
    "satisfaction_score", "support_tickets",
}


def to_json_value(value):
    """Convert DuckDB/pandas/numpy values into JSON-friendly Python values."""
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if value is None:
        return None
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


def frame_to_table(df, title=None):
    """Turn a DataFrame into {"title", "columns", "rows"} for JSON responses."""
    return {
        "title": title,
        "columns": [str(column) for column in df.columns],
        "rows": [[to_json_value(value) for value in row] for row in df.itertuples(index=False)],
    }


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


def _empty_result(question, kind):
    return {
        "kind": kind,             # profile | churn | answer | unanswerable | failed
        "question": question,
        "answer": "",
        "profile": None,          # dataset profile text (kind == "profile")
        "notes": [],              # extra verified facts shown under the answer
        "sql": None,
        "table": None,            # main query result {"title", "columns", "rows"}
        "row_count": 0,
        "truncated": False,
        "tables": [],             # additional tables (e.g. churn by segment)
        "assumptions": [],
        "warnings": [],
        "repairs": [],            # SQL errors the model was asked to fix
        "grounded": None,         # True/False once numbers were checked
        "chart_path": None,
        "chart_error": None,
        "elapsed_s": 0.0,
        "llm_report": None,
        "suggestions": [],        # example questions the user can click (chat / clarify)
    }


def _churn_report(question, engine):
    result = _empty_result(question, "churn")
    cohorts, segments = engine.churn_associations()
    by_status = {bool(row.is_churned): row for row in cohorts.itertuples(index=False)}
    active, churned = by_status.get(False), by_status.get(True)
    if active is not None and churned is not None:
        result["answer"] = (
            f"Compared across {churned.customers} churned and {active.customers} active customers, "
            f"average satisfaction was {churned.avg_satisfaction_per_customer:.2f} for churned "
            f"customers and {active.avg_satisfaction_per_customer:.2f} for active customers. "
            f"Average support tickets per order per customer were "
            f"{churned.avg_tickets_per_order_per_customer:.2f} versus "
            f"{active.avg_tickets_per_order_per_customer:.2f}. "
            f"Average ages were {churned.avg_customer_age:.2f} and {active.avg_customer_age:.2f}, respectively."
        )
    else:
        result["answer"] = "The dataset does not contain both churned and active customers to compare."
    result["tables"].append(frame_to_table(segments, "Customer segment churn rates"))
    result["notes"].append(
        "These are patterns associated with churn in this CSV, not proof of why customers churn. "
        "The dataset is observational and cannot establish causation."
    )
    return result


def _example_questions(engine):
    """Build a few example questions from the dataset's real column names."""
    types = engine.column_types()
    numeric = [
        name for name, type_name in types.items()
        if type_name.upper().startswith(NUMERIC_TYPE_PREFIXES) and not ID_COLUMN.search(name)
    ]
    measure = next(
        (name for hint in MEASURE_HINTS for name in numeric if hint in name.lower()),
        numeric[0] if numeric else None,
    )
    text_columns = [
        name for name, type_name in types.items()
        if type_name.upper() == "VARCHAR" and not ID_COLUMN.search(name)
    ]
    categories = []
    if text_columns:
        expressions = ", ".join(
            'approx_count_distinct("' + name.replace('"', '""') + '")' for name in text_columns
        )
        counts = engine.con.execute(f"SELECT {expressions} FROM data").fetchone()
        distinct = {name: count for name, count in zip(text_columns, counts) if 1 < count <= 30}
        categories = list(distinct)

    second_measure = next(
        (name for hint in MEASURE_HINTS for name in numeric if hint in name.lower() and name != measure),
        None,
    )
    has_dates = any(type_name.upper().startswith(("DATE", "TIMESTAMP")) for type_name in types.values())

    # One example per kind of analysis: overview, breakdown, ranking, trend, comparison, data quality.
    examples = ["Give me an overview of this dataset"]
    if measure and categories:
        examples.append(f"Show total {measure} by {categories[0]} in a bar chart")
        # A "top 5" only makes sense for a column with more than 5 different values.
        ranked = max(categories, key=lambda name: distinct[name])
        if distinct[ranked] > 5:
            examples.append(f"What are the top 5 {ranked} by total {measure}?")
        else:
            examples.append(f"Which {ranked} has the highest total {measure}?")
    elif measure:
        examples.append(f"What are the average, minimum and maximum {measure}?")
    if measure and has_dates:
        examples.append(f"How did total {measure} change month by month?")
    if measure and second_measure and categories:
        examples.append(f"Compare total {measure} and {second_measure} by {categories[-1]}")
    examples.append("Which columns have missing values?")
    return examples


def _small_talk_reply(question, engine):
    """Answer greetings, thanks and "what can you do" without calling the model."""
    result = _empty_result(question, "chat")
    text = question.lower()
    if re.search(r"\b(thanks|thank|thx|ty)\b", text):
        result["answer"] = (
            "You're welcome! Happy to dig into the data whenever you're ready. "
            "Here are a few more things you could explore:"
        )
    elif re.search(r"\b(bye|goodbye|see you)\b", text):
        result["answer"] = (
            "Goodbye! Your chat history is saved, so you can come back and pick up right where you left off."
        )
    elif re.search(r"\b(ok|okay|cool|nice|great)\b", text):
        result["answer"] = "Great! What would you like to look at next? A few ideas:"
    else:
        result["answer"] = (
            "Hi! I'm your data analysis assistant for this CSV.\n\n"
            "Ask me anything about the data in plain English: totals, averages, top performers, "
            "trends over time, or comparisons between groups. I turn your question into a SQL query, "
            "run it on the full dataset, and double-check every number before I answer.\n\n"
            'Want a picture? Add "chart" or "plot" to your question and I\'ll draw a bar chart.\n\n'
            "Here are some questions to get you started:"
        )
    result["suggestions"] = _example_questions(engine)
    return result


def _clarify_reply(question, engine):
    """Reply when the model could not turn the message into a query about the data."""
    result = _empty_result(question, "clarify")
    result["answer"] = (
        "I'm not sure how to answer that from this dataset. I work best with questions about the data "
        "itself, such as a total, an average, a ranking, a trend over time, or a comparison between groups. "
        "You could try one of these:"
    )
    result["suggestions"] = _example_questions(engine)
    return result


def analyze(question, llm, engine, profile_text, missing, duplicates, chart_dir=None, on_status=None):
    """Answer one question about the loaded CSV and return a JSON-friendly result dict.

    on_status, if given, is called with short progress messages (e.g. SQL repair attempts).
    """
    start = time.time()

    def finish(result):
        result["elapsed_s"] = round(time.time() - start, 1)
        result["llm_report"] = llm.speed_report() if llm.stats["calls"] else None
        return result

    if SMALL_TALK.match(question):
        return finish(_small_talk_reply(question, engine))

    if is_dataset_description_request(question):
        result = _empty_result(question, "profile")
        result["profile"] = profile_text
        result["answer"] = "Here is an overview of the dataset, computed directly from the CSV."
        result["notes"] = [
            f"Exact duplicate rows: {duplicates}",
            f"Missing cells across all columns: {sum(missing.values())}",
            "This profile is computed directly from the CSV; no SQL or model interpretation is needed.",
        ]
        return finish(result)

    if CHURN_WHY_REQUEST.search(question) and CHURN_COLUMNS <= set(engine.column_types()):
        return finish(_churn_report(question, engine))

    plan = question_to_sql(llm, profile_text, question)
    if plan.get("answerable") is False:
        result = _empty_result(question, "unanswerable")
        result["answer"] = plan.get("reason") or "This dataset does not contain the needed data."
        return finish(result)

    repairs = []
    sql, frame, truncated = plan["sql"], None, False
    error = plan.get("error")
    if not sql and not error:
        # Valid reply but no query: the message is not a question about the data
        # (e.g. "ok then"). Retrying as a "repair" only wastes model calls.
        return finish(_clarify_reply(question, engine))
    for attempt in range(MAX_REPAIRS + 1):
        if sql:
            try:
                check_sql(sql)
                frame, truncated = engine.run(sql)
                break
            except Exception as exc:
                error = str(exc).split("\n")[0]
        else:
            error = error or NO_QUERY

        if attempt < MAX_REPAIRS:
            repairs.append(error[:300])
            if on_status:
                on_status(f"query failed: {error[:120]} - asking the model to fix it")
            plan = question_to_sql(llm, profile_text, question, failed_sql=sql or "(none)", error=error)
            if plan.get("answerable") is False:
                result = _empty_result(question, "unanswerable")
                result["answer"] = plan.get("reason") or "This dataset does not contain the needed data."
                result["repairs"] = repairs
                return finish(result)
            sql, error = plan["sql"], plan.get("error")

    if frame is None and error == NO_QUERY:
        result = _clarify_reply(question, engine)
        result["repairs"] = repairs
        return finish(result)
    if frame is None:
        result = _empty_result(question, "failed")
        result["answer"] = f"Could not compute a reliable answer (last error: {error}). No numbers are reported."
        result["sql"] = sql or None
        result["repairs"] = repairs
        return finish(result)

    share_context = add_share_context(question, sql, frame, engine)

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

    notes = []
    if share_context:
        notes.append(
            "The share percentage was independently recalculated using the full dataset total; "
            "the SQL shown is the model-generated query."
        )
        notes.append(
            f'Verified: {share_context["category_value"]} {share_context["measure_column"]} = '
            f'{share_context["category_amount"]:,.2f}; overall {share_context["measure_column"]} = '
            f'{share_context["overall_amount"]:,.2f}; share = {share_context["share_pct"]:.2f}%.'
        )
        if share_context["unassigned_rows"]:
            warnings.append(
                f'{share_context["unassigned_rows"]} rows have no "{share_context["category_column"]}" '
                f'value; their {share_context["unassigned_amount"]:,.2f} in '
                f'"{share_context["measure_column"]}" is included in the overall total but cannot be '
                "attributed to a category."
            )

    answer, grounded = write_answer(llm, question, frame, assumptions)

    chart_path = chart_error = None
    if wants_bar_chart(question):
        try:
            chart_path, chart_error = save_bar_chart(frame, question, output_dir=chart_dir)
        except Exception as exc:
            chart_error = f"Could not create the chart: {exc}"

    result = _empty_result(question, "answer")
    result.update(
        answer=answer,
        notes=notes,
        sql=sql,
        table=frame_to_table(frame),
        row_count=len(frame),
        truncated=truncated,
        assumptions=assumptions,
        warnings=warnings,
        repairs=repairs,
        grounded=grounded,
        chart_path=str(chart_path) if chart_path else None,
        chart_error=chart_error,
    )
    return finish(result)
