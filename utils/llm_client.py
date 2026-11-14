"""Language-model clients (Ollama or vLLM) and natural-language-to-SQL prompt handling."""

import json
import re
import time

import requests

BACKENDS = ("ollama", "vllm")
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"  # Ollama model name
# `vllm serve` exposes an OpenAI-compatible API. Its default port (8000) is the one Django's
# runserver uses, so the vLLM server is expected on 8001.
VLLM_URL = "http://localhost:8001/v1"
VLLM_SERVE_HINT = "./serve_vllm.sh (runs vllm serve with the settings this app expects)"
REQUEST_TIMEOUT_S = 300
CONTEXT_LENGTH = 8192
DEFAULT_TEMPERATURE = 0.2
# A repair at the same temperature tends to repeat the same failed query, so repairs
# sample with a little more randomness.
REPAIR_TEMPERATURE = 0.4

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
- One read-only query on table data (or data_raw only for duplicate questions). Prefer a plain SELECT ... GROUP BY. If you use WITH, the query must end with a final SELECT that reads from the CTE: `WITH t AS (...) SELECT ... FROM t`. A WITH block on its own is not a complete query.
- Use only the listed columns; wrap column names in double quotes. String/category values are string literals in single quotes (for example, `"region" = 'Region A'`); never use a category value as an unquoted or double-quoted column name.
- Exclude NULLs from grouping keys and measured columns.
- Give output columns clear snake_case names. Percentages on a 0-100 scale with a _pct suffix. Round decimals to 2 places.
- For a share of a total, calculate numerator and denominator as separate aggregates (use scalar subqueries or CTEs); do not nest aggregate functions or use a window over a filtered result for the overall denominator. Unless the user says otherwise, the denominator includes all rows, including rows with a missing category. State that missing-category rows are included in the total but cannot be attributed to a category.
- Example: `WITH totals AS (SELECT SUM("revenue") AS all_revenue FROM data), selected AS (SELECT SUM("revenue") AS region_revenue FROM data WHERE "region" = 'Region A') SELECT ROUND(100.0 * selected.region_revenue / totals.all_revenue, 2) AS region_a_revenue_share_pct FROM selected CROSS JOIN totals`.
- Attributes repeated on every row of an entity (e.g. a customer's churn status) must be counted per unique id, not per row.
- For a customer churn percentage, first make one row per distinct customer_id, assign each customer a churned flag using the actual `churned` column type and values shown in the dataset profile (use TRUE for a boolean column, or the exact positive string for a text column), then calculate 100 * SUM(customer_churned_flag) / COUNT(*). Never sum churned order rows and divide by distinct customers; the result must be between 0 and 100.
- Order results meaningfully and return at most 50 rows unless asked for more.
- Do not use nested aggregates such as `SUM(SUM(...))`; compute aggregate totals in a CTE or subquery first.
- Date columns are DATE type: use EXTRACT(year FROM col) or strftime(col, '%Y-%m').
- Example of a trend over time (adapt the column names to the dataset): `SELECT strftime("sale_date", '%Y-%m') AS month, ROUND(SUM("amount"), 2) AS total_amount FROM data WHERE "sale_date" IS NOT NULL AND "amount" IS NOT NULL GROUP BY month ORDER BY month`.
- List every choice the user did not specify in assumptions."""


class LLMError(RuntimeError):
    """The language model server could not be reached or returned an error."""


class BaseLLM:
    """Shared chat interface: prompt printing, <think> stripping and token statistics."""

    backend = ""

    def __init__(self, model, show_prompts=False):
        self.model = model
        self.show_prompts = show_prompts
        self.stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "prompt_ns": 0,     # time spent reading the prompt (0 when the server does not report it)
            "output_tokens": 0,
            "output_ns": 0,     # generation time, or whole-request time when not reported separately
        }

    def chat(self, system, user, json_schema=None, max_tokens=600, temperature=DEFAULT_TEMPERATURE):
        if self.show_prompts:
            print(f"\n----- PROMPT -----\n{user}\n------------------")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self._complete(messages, json_schema, max_tokens, temperature)
        self.stats["calls"] += 1
        return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()

    def _complete(self, messages, json_schema, max_tokens, temperature):
        """Send one chat request, update token stats and return the reply text."""
        raise NotImplementedError

    def _post(self, url, payload, server_name):
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
        except requests.ConnectionError as exc:
            raise LLMError(f"Cannot reach {server_name} at {url}. {self.start_hint}") from exc
        except requests.Timeout as exc:
            raise LLMError(f"{server_name} did not respond within {REQUEST_TIMEOUT_S} seconds.") from exc
        if response.status_code != 200:
            raise LLMError(f"{server_name} error {response.status_code}: {response.text[:300]}")
        return response.json()

    def speed_report(self):
        stats = self.stats
        prompt_rate = (
            f" at {stats['prompt_tokens'] / (stats['prompt_ns'] / 1e9):,.0f} tok/s"
            if stats["prompt_ns"] else ""
        )
        output_rate = (
            stats["output_tokens"] / (stats["output_ns"] / 1e9)
            if stats["output_ns"] else 0
        )
        return (
            f"{stats['calls']} LLM calls via {self.backend} | model: {self.model} | "
            f"prompt: {stats['prompt_tokens']} tokens{prompt_rate} | "
            f"output: {stats['output_tokens']} tokens at {output_rate:,.0f} tok/s"
        )


class OllamaLLM(BaseLLM):
    """Model served by Ollama (for example qwen3.5:4b)."""

    backend = "ollama"
    start_hint = "Start it with: ollama serve"

    def __init__(self, model=DEFAULT_MODEL, show_prompts=False, url=OLLAMA_URL):
        super().__init__(model, show_prompts)
        self.url = url

    def _complete(self, messages, json_schema, max_tokens, temperature):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_ctx": CONTEXT_LENGTH,
                "num_predict": max_tokens,
            },
        }
        if json_schema:
            payload["format"] = json_schema
        data = self._post(self.url, payload, "Ollama")
        stats = self.stats
        stats["prompt_tokens"] += data.get("prompt_eval_count", 0)
        stats["prompt_ns"] += data.get("prompt_eval_duration", 0)
        stats["output_tokens"] += data.get("eval_count", 0)
        stats["output_ns"] += data.get("eval_duration", 0)
        return data["message"]["content"]


# Kept so existing imports of `LLM` keep working; it is the Ollama client.
LLM = OllamaLLM


class VLLMClient(BaseLLM):
    """Local model weights served by `vllm serve` through its OpenAI-compatible API.

    With no model name, the client asks the server which model it serves, so it works
    whether or not vLLM was started with --served-model-name.
    """

    backend = "vllm"
    start_hint = f"Start it with: {VLLM_SERVE_HINT}"

    def __init__(self, model="", show_prompts=False, url=VLLM_URL):
        super().__init__(model, show_prompts)
        self.url = url.rstrip("/")

    def _served_model(self):
        if not self.model:
            try:
                response = requests.get(f"{self.url}/models", timeout=10)
                response.raise_for_status()
                models = response.json().get("data") or []
            except requests.RequestException as exc:
                raise LLMError(f"Cannot reach vLLM at {self.url}. {self.start_hint}") from exc
            if not models:
                raise LLMError(f"The vLLM server at {self.url} is not serving any model.")
            self.model = models[0]["id"]
        return self.model

    def _complete(self, messages, json_schema, max_tokens, temperature):
        payload = {
            "model": self._served_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Qwen3 models think before answering by default; this pipeline wants the answer only,
            # like Ollama's "think": False.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        started = time.perf_counter_ns()
        data = self._post(f"{self.url}/chat/completions", payload, "vLLM")
        usage = data.get("usage") or {}
        stats = self.stats
        stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
        stats["output_tokens"] += usage.get("completion_tokens", 0)
        # vLLM does not split prompt and generation time, so the rate covers the whole request.
        stats["output_ns"] += time.perf_counter_ns() - started
        return data["choices"][0]["message"].get("content")


def make_llm(backend="ollama", model=None, url=None, show_prompts=False):
    """Create the client for a backend: "ollama" (default) or "vllm"."""
    backend = (backend or "ollama").strip().lower()
    if backend == "ollama":
        return OllamaLLM(model or DEFAULT_MODEL, show_prompts, url or OLLAMA_URL)
    if backend == "vllm":
        return VLLMClient(model or "", show_prompts, url or VLLM_URL)
    raise ValueError(f"Unknown LLM backend {backend!r}; choose one of: {', '.join(BACKENDS)}")


def question_to_sql(llm, profile_text, question, failed_sql=None, error=None):
    """Ask the model for a safe SQL query and stated assumptions."""
    prompt = f"Dataset:\n{profile_text}\n\n{SQL_RULES}\n\nQuestion: {question}"
    temperature = DEFAULT_TEMPERATURE
    if failed_sql:
        prompt += (
            f"\n\nYour previous query failed.\nQuery: {failed_sql}\n"
            f"Error: {error}\nReturn a corrected query that fixes this error. "
            "Do not return the same query again. If the query starts with WITH, "
            "make sure a final SELECT reads from the CTE after its closing parenthesis."
        )
        temperature = REPAIR_TEMPERATURE
    reply = llm.chat(SQL_SYSTEM, prompt, json_schema=SQL_SCHEMA, temperature=temperature)
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
