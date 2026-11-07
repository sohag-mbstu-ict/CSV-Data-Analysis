"""Ollama client and natural-language-to-SQL prompt handling."""

import json
import re

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:2b"
CONTEXT_LENGTH = 8192

SQL_SYSTEM = (
    "You translate questions about a dataset into one DuckDB SQL query. You never compute answers "
    "yourself; the query result is shown to the user as evidence. Python handles chart rendering "
    "after the query runs, so do not refuse a question just because it requests a visualization."
)

SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "reason": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "sql": {"type": "string"},
    },
    "required": ["answerable", "reason", "assumptions", "sql"],
}

SQL_RULES = """Rules:
- If the question needs a column, value or time period that is not in the dataset, set answerable to false, explain why in reason, and leave sql empty. Never substitute a different column.
- Requests to show, display, visualize, chart, graph, plot, or otherwise present data visually are answerable when the requested data is available. Return the SELECT query for the underlying data; Python will render a visualization. Never mark a question unanswerable only because SQL returns tables rather than images.
- For a request for a single total or metric with no grouping (for example, total revenue in a bar chart), return one row with the requested aggregate so it can be plotted as a single bar.
- One read-only SELECT (WITH/CTEs allowed) on table data (or data_raw only for duplicate questions).
- Use only the listed columns; wrap column names in double quotes. String/category values are string literals in single quotes (for example, `"region" = 'Region A'`); never use a category value as an unquoted or double-quoted column name.
- Exclude NULLs from grouping keys and measured columns.
- Give output columns clear snake_case names. Percentages on a 0-100 scale with a _pct suffix. Round decimals to 2 places.
- For a share of a total, calculate numerator and denominator as separate aggregates (use scalar subqueries or CTEs); do not nest aggregate functions or use a window over a filtered result for the overall denominator. Unless the user says otherwise, the denominator includes all rows, including rows with a missing category. State that missing-category rows are included in the total but cannot be attributed to a category.
- Example: `WITH totals AS (SELECT SUM("revenue") AS all_revenue FROM data), selected AS (SELECT SUM("revenue") AS region_revenue FROM data WHERE "region" = 'Region A') SELECT ROUND(100.0 * selected.region_revenue / totals.all_revenue, 2) AS region_a_revenue_share_pct FROM selected CROSS JOIN totals`.
- Attributes repeated on every row of an entity (e.g. a customer's churn status) must be counted per unique id, not per row.
- For a customer churn percentage, first make one row per distinct customer_id, assign each customer a churned flag (1 if their churned value is 'Yes', otherwise 0), then calculate 100 * SUM(customer_churned_flag) / COUNT(*). Never sum churned order rows and divide by distinct customers; the result must be between 0 and 100.
- Order results meaningfully and return at most 50 rows unless asked for more.
- Do not use nested aggregates such as `SUM(SUM(...))`; compute aggregate totals in a CTE or subquery first.
- Date columns are DATE type: use EXTRACT(year FROM col) or strftime(col, '%Y-%m').
- List every choice the user did not specify in assumptions."""


class LLMError(RuntimeError):
    """The language model server could not be reached or returned an error."""


class LLM:
    """Small Ollama chat client with token and throughput statistics."""

    def __init__(self, model=DEFAULT_MODEL, show_prompts=False):
        self.model = model
        self.show_prompts = show_prompts
        self.stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "prompt_ns": 0,
            "output_tokens": 0,
            "output_ns": 0,
        }

    def chat(self, system, user, json_schema=None, max_tokens=600):
        if self.show_prompts:
            print(f"\n----- PROMPT -----\n{user}\n------------------")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": CONTEXT_LENGTH,
                "num_predict": max_tokens,
            },
        }
        if json_schema:
            payload["format"] = json_schema
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        except requests.ConnectionError as exc:
            raise LLMError("Cannot reach Ollama. Start it with: ollama serve") from exc
        except requests.Timeout as exc:
            raise LLMError("Ollama did not respond within 300 seconds.") from exc
        if response.status_code != 200:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")

        data = response.json()
        stats = self.stats
        stats["calls"] += 1
        stats["prompt_tokens"] += data.get("prompt_eval_count", 0)
        stats["prompt_ns"] += data.get("prompt_eval_duration", 0)
        stats["output_tokens"] += data.get("eval_count", 0)
        stats["output_ns"] += data.get("eval_duration", 0)
        text = data["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def speed_report(self):
        stats = self.stats
        prompt_rate = (
            stats["prompt_tokens"] / (stats["prompt_ns"] / 1e9)
            if stats["prompt_ns"] else 0
        )
        output_rate = (
            stats["output_tokens"] / (stats["output_ns"] / 1e9)
            if stats["output_ns"] else 0
        )
        return (
            f"{stats['calls']} LLM calls | prompt: {stats['prompt_tokens']} tokens at "
            f"{prompt_rate:,.0f} tok/s | output: {stats['output_tokens']} tokens at "
            f"{output_rate:,.0f} tok/s"
        )


def question_to_sql(llm, profile_text, question, failed_sql=None, error=None):
    """Ask the model for a safe SQL query and stated assumptions."""
    prompt = f"Dataset:\n{profile_text}\n\n{SQL_RULES}\n\nQuestion: {question}"
    if failed_sql:
        prompt += (
            f"\n\nYour previous query failed.\nQuery: {failed_sql}\n"
            f"Error: {error}\nReturn a corrected query."
        )
    reply = llm.chat(SQL_SYSTEM, prompt, json_schema=SQL_SCHEMA)
    try:
        result = json.loads(reply)
    except json.JSONDecodeError:
        return {
            "answerable": True,
            "reason": "",
            "assumptions": [],
            "sql": "",
            "error": "invalid JSON from model",
        }
    result["sql"] = (result.get("sql") or "").strip().rstrip(";")
    return result
