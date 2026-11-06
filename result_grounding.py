"""Check generated numeric claims against query results and write summaries."""

import re
import pandas as pd
TABLE_ROWS_FOR_ANSWER = 40

NUMBER = re.compile(
    r"(?<![\w.\-])(-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(%|million\b|thousand\b|k\b|m\b|bn\b|billion\b)?",
    re.I,
)
DATE = re.compile(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b")
SCALE = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "bn": 1e9,
    "billion": 1e9,
}
ANSWER_SYSTEM = (
    "You explain query results to a non-technical user. You only state facts visible in the result "
    "table. You never use outside knowledge."
)


def numbers_in(text):
    """Find numbers in text with a tolerance based on how they are rounded."""
    text = str(text)
    found = []
    for raw, suffix in NUMBER.findall(DATE.sub(" ", text.replace("\u2212", "-"))):
        digits = raw.replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        multiplier = SCALE.get(suffix.lower(), 1)
        if "." in digits:
            tolerance = 0.5 * 10 ** -len(digits.split(".")[1])
        else:
            body = digits.lstrip("-")
            trailing_zeros = len(body) - len(body.rstrip("0")) if body.strip("0") else 0
            tolerance = 0.5 * 10 ** trailing_zeros
        found.append((raw + suffix, value * multiplier, tolerance * multiplier + 1e-9))
    return found


def ungrounded(answer, df, question):
    """Return numeric claims in an answer that are not supported by the result or question."""
    allowed = [
        float(value)
        for value in df.to_numpy().ravel()
        if isinstance(value, (int, float)) and not pd.isna(value)
    ]
    for value in df.astype(str).to_numpy().ravel():
        allowed.extend(number for _, number, _ in numbers_in(value))
    allowed.extend(number for _, number, _ in numbers_in(question))
    allowed.append(len(df))

    unsupported = []
    for raw, value, tolerance in numbers_in(answer):
        if abs(value).is_integer() and abs(value) <= 10:
            continue
        if not any(
            abs(abs(value) - abs(allowed_value) * scale) <= tolerance
            for allowed_value in allowed
            for scale in (1, 100, 0.01)
        ):
            unsupported.append(raw)
    return unsupported


def write_answer(llm, question, df, assumptions):
    """Ask the language model for a concise answer and verify its numeric claims."""
    table = df.head(TABLE_ROWS_FOR_ANSWER).to_csv(index=False) if len(df) else "(no rows)"
    share_guidance = ""
    if {"category_amount", "overall_amount", "share_pct"}.issubset(df.columns):
        share_guidance = (
            " For this share-of-total result, state the category's amount, the overall total amount, "
            "and the percentage. If unassigned_category_rows is present and nonzero, also explain that "
            "unassigned_category_amount is included in the denominator but cannot be attributed to a category."
        )
    prompt = (
        f"Question: {question}\nAssumptions: {'; '.join(assumptions) or 'none'}\n\n"
        f"Result table ({len(df)} rows):\n{table}\n"
        "Answer in 2-5 plain sentences. Lead with the direct answer. Every number you write must appear in "
        f"the table (rounding is fine); do not calculate new numbers. If the table is empty, say no records match.{share_guidance}"
    )
    for _ in range(2):
        answer = llm.chat(ANSWER_SYSTEM, prompt, max_tokens=300)
        unsupported = ungrounded(answer, df, question)
        if not unsupported:
            return answer, True
        prompt += (
            f"\n\nYour previous answer used numbers not in the table: {', '.join(unsupported)}. "
            "Rewrite it."
        )
    return (
        "I computed the result table below but could not write a fully verified summary; "
        "read the figures directly from the table.",
        False,
    )
