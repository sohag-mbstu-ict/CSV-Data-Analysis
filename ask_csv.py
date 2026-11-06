import argparse
import json
import re
import duckdb
import sys
import threading
import time

import pandas as pd
import requests

# ----------------------------------------------------------------------------- settings
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
CSV_PATH = "data.csv"         # Change this to the path of your CSV file.
CONTEXT_LENGTH = 8192        # tokens of prompt + answer the model can hold
MAX_REPAIRS = 2              # times the model may fix a failing query
MAX_ROWS = 200               # rows kept from a query result
TABLE_ROWS_FOR_ANSWER = 40   # rows shown to the model when it writes the answer
QUERY_TIMEOUT_S = 20


# ----------------------------------------------------------------------------- 1. profile the CSV
def build_profile(csv_path):
    """Describe the dataset (columns, types, missing values, ranges) without sending raw rows."""
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        sys.exit("The CSV file has no data rows.")
    dups = int(df.duplicated().sum())
    # data	4	Duplicate rows removed
    # data_raw	5	Original data
    lines = [
        f"Table `data`: {len(df) - dups} rows (exact duplicate rows removed). Use this table by default.",
        f"Table `data_raw`: {len(df)} rows exactly as uploaded ({dups} duplicate rows). "
        f"Use it only for questions about duplicates.",
        "Columns:",
    ]
    missing = {}
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        missing[col] = n_missing
        info = f'- "{col}": missing {n_missing}, {s.nunique()} distinct'
        if pd.api.types.is_numeric_dtype(s):
            info = f'- "{col}" (number): missing {n_missing}, range {s.min()} to {s.max()}'
        else:
            text = s.dropna().astype(str) # Convert non-missing values into strings
            looks_like_date = len(text) and text.head(200).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9
            if looks_like_date:
                info = f'- "{col}" (date): missing {n_missing}, from {text.min()} to {text.max()}'
            elif s.nunique() <= 30:
                info += f", values: {sorted(text.unique().tolist())}"
            else:
                info += f", examples: {text.unique()[:3].tolist()}"
        lines.append(info)
    return "\n".join(lines), missing, dups


# ----------------------------------------------------------------------------- 2. sandboxed DuckDB
# In your application, DuckDB will perform these calculations instead of asking the LLM to calculate them.
class Engine:
    """Loads the CSV into in-memory DuckDB, then locks it down (no file or network access)."""

    def __init__(self, csv_path):
        # This means the database exists in RAM while your Python application is running.
        # It does not create a persistent DuckDB database file.
        # Create an in-memory database connection
        self.con = duckdb.connect(":memory:") # ":memory:" tells DuckDB to create a temporary database in RAM.
        # A connection is like a communication channel between Python and DuckDB
        # Through this connection, Python can send SQL commands to DuckDB.
        # In SQL, two consecutive single quotes represent one literal single quote inside a string.
        # This helps prevent the path from accidentally breaking the SQL string.
        path = str(csv_path).replace("'", "''")
        # DuckDB provides a function called read_csv_auto().
        # It reads a CSV file and automatically detects:
        # - Column names.
        # - Data types.
        # - Numeric columns.
        # - Text columns.
        # - Date-like columns.
        # sample_size=-1  tells DuckDB to use the entire CSV for type detection.
        self.con.execute(f"CREATE TABLE data_raw AS SELECT * FROM read_csv_auto('{path}', sample_size=-1)")
        # After execution Your CSV data is now available inside DuckDB as a table named data_raw.
        # data_raw: Original records, useful for duplicate analysis.
        # data: Cleaned records, used for normal analytics.
        self.con.execute("CREATE TABLE data AS SELECT DISTINCT * FROM data_raw")
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    def run(self, sql):
        # A cursor is an object used to execute SQL statements and retrieve their results.
        # A tool used to execute and fetch results through that connection.
        cur = self.con.cursor()
        timer = threading.Timer(QUERY_TIMEOUT_S, cur.interrupt)
        timer.start()
        try:
            cur.execute(sql)
            # Go through each column's metadata and take its first item, which is the column name.
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
        finally:
            timer.cancel()
            cur.close()
        return pd.DataFrame(rows[:MAX_ROWS], columns=columns), len(rows) > MAX_ROWS


FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|copy|export|install|load|pragma|set|call)\b"
                       r"|\b(read_\w+|glob|parquet_\w+|csv_scan)\s*\(|\b(from|join)\s+'", re.I)


def check_sql(sql):
    """Allow only one read-only SELECT. (DuckDB's own sandbox is the second safety layer.)"""
    code = re.sub(r"'(?:[^']|'')*'", "''", sql)  # ignore text inside quotes
    # Checking whether the query starts with SELECT or WITH
    if not re.match(r"^\s*(select|with)\b", code, re.I):
        raise ValueError("Only SELECT queries are allowed")
    if ";" in code.strip().rstrip(";"):
        raise ValueError("Only one statement is allowed")
    bad = FORBIDDEN.search(code)
    if bad:
        raise ValueError(f"Forbidden SQL: {bad.group(0)}")


# ----------------------------------------------------------------------------- 3. talk to Ollama
class LLM:
    def __init__(self, model, show_prompts=False):
        self.model, self.show_prompts = model, show_prompts
        # calls : Total number of LLM requests
        # prompt_tokens	: Total tokens processed from input prompts
        # prompt_ns	: Total nanoseconds spent on prompt evaluation
        # output_tokens	: Total tokens processed from output
        # output_ns	: Total nanoseconds spent on output evaluation
        self.stats = {"calls": 0, "prompt_tokens": 0, "prompt_ns": 0, "output_tokens": 0, "output_ns": 0}

    def chat(self, system, user, json_schema=None, max_tokens=600):
        if self.show_prompts:
            print(f"\n----- PROMPT -----\n{user}\n------------------")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "think": False,                      # no hidden reasoning tokens: much faster
            "keep_alive": "30m",                 # keep the model in GPU memory between questions
            "options": {"temperature": 0, "num_ctx": CONTEXT_LENGTH, "num_predict": max_tokens},
        }
        if json_schema:
            payload["format"] = json_schema      # Ollama forces the reply to match this JSON schema
        try:
            # Automatically converts your Python dictionary into JSON and sends it in the request body.
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        except requests.ConnectionError:
            sys.exit("Cannot reach Ollama. Start it with: ollama serve")
        if resp.status_code != 200:
            sys.exit(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        data = resp.json() # Convert JSON response into a Python dictionary
        s = self.stats # Accessing the statistics dictionary
        s["calls"] += 1
        # Why use .get() instead of direct indexing?
        # Because it prevents a KeyError if a particular statistics field is missing.
        s["prompt_tokens"] += data.get("prompt_eval_count", 0)
        s["prompt_ns"] += data.get("prompt_eval_duration", 0)
        s["output_tokens"] += data.get("eval_count", 0)
        s["output_ns"] += data.get("eval_duration", 0)
        text = data["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def speed_report(self):
        s = self.stats
        prompt_rate = s["prompt_tokens"] / (s["prompt_ns"] / 1e9) if s["prompt_ns"] else 0
        output_rate = s["output_tokens"] / (s["output_ns"] / 1e9) if s["output_ns"] else 0
        return (f"{s['calls']} LLM calls | prompt: {s['prompt_tokens']} tokens at {prompt_rate:,.0f} tok/s | "
                f"output: {s['output_tokens']} tokens at {output_rate:,.0f} tok/s")


# ----------------------------------------------------------------------------- 4. question -> SQL
SQL_SYSTEM = ("You translate questions about a dataset into one DuckDB SQL query. You never compute answers "
              "yourself; the query result is shown to the user as evidence.")

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
- One read-only SELECT (WITH/CTEs allowed) on table data (or data_raw only for duplicate questions).
- Use only the listed columns; wrap column names in double quotes.
- Exclude NULLs from grouping keys and measured columns.
- Give output columns clear snake_case names. Percentages on a 0-100 scale with a _pct suffix. Round decimals to 2 places.
- Attributes repeated on every row of an entity (e.g. a customer's churn status) must be counted per unique id, not per row.
- Order results meaningfully and return at most 50 rows unless asked for more.
- Date columns are DATE type: use EXTRACT(year FROM col) or strftime(col, '%Y-%m').
- List every choice the user did not specify in assumptions."""


def question_to_sql(llm, profile_text, question, failed_sql=None, error=None):
    prompt = f"Dataset:\n{profile_text}\n\n{SQL_RULES}\n\nQuestion: {question}"
    if failed_sql:
        prompt += f"\n\nYour previous query failed.\nQuery: {failed_sql}\nError: {error}\nReturn a corrected query."
    reply = llm.chat(SQL_SYSTEM, prompt, json_schema=SQL_SCHEMA)
    try:
        result = json.loads(reply)
    except json.JSONDecodeError:
        return {"answerable": True, "reason": "", "assumptions": [], "sql": "", "error": "invalid JSON from model"}
    result["sql"] = (result.get("sql") or "").strip().rstrip(";")
    return result


# ----------------------------------------------------------------------------- 5. grounding check
NUMBER = re.compile(r"(?<![\w.\-])(-?\d[\d,]*(?:\.\d+)?)\s*(%|million\b|thousand\b|k\b|m\b|bn\b|billion\b)?", re.I)
DATE = re.compile(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b")
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


def numbers_in(text):
    """Find numbers in text with a tolerance based on how they are rounded."""
    out = []
    for raw, suffix in NUMBER.findall(DATE.sub(" ", text.replace("\u2212", "-"))):
        digits = raw.replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        mult = SCALE.get(suffix.lower(), 1)
        if "." in digits:
            tol = 0.5 * 10 ** -len(digits.split(".")[1])
        else:
            body = digits.lstrip("-")
            # Your goal is to count how many zeros exist at the end of an integer.
            zeros = len(body) - len(body.rstrip("0")) if body.strip("0") else 0
            tol = 0.5 * 10 ** zeros
        out.append((raw + suffix, value * mult, tol * mult + 1e-9))
    return out

# Checks whether numerical values mentioned in an AI-generated answer can be supported by the actual data.
def ungrounded(answer, df, question):
    allowed = [float(v) for v in df.to_numpy().ravel() if isinstance(v, (int, float)) and not pd.isna(v)]
    for v in df.astype(str).to_numpy().ravel():
        allowed += [n for _, n, _ in numbers_in(v)]
        # The numbers_in() function extracts numerical values from these strings.
    allowed += [n for _, n, _ in numbers_in(question)] + [len(df)]
    bad = []
    for raw, value, tol in numbers_in(answer):
        if abs(value).is_integer() and abs(value) <= 10:   # "top 5", "3 regions"
            continue
        if not any(abs(abs(value) - abs(a) * k) <= tol for a in allowed for k in (1, 100, 0.01)):
            bad.append(raw)
    return bad


# ----------------------------------------------------------------------------- 6. result -> answer
ANSWER_SYSTEM = ("You explain query results to a non-technical user. You only state facts visible in the result "
                 "table. You never use outside knowledge.")


def write_answer(llm, question, df, assumptions):
    table = df.head(TABLE_ROWS_FOR_ANSWER).to_csv(index=False) if len(df) else "(no rows)"
    prompt = (f"Question: {question}\nAssumptions: {'; '.join(assumptions) or 'none'}\n\n"
              f"Result table ({len(df)} rows):\n{table}\n"
              "Answer in 2-5 plain sentences. Lead with the direct answer. Every number you write must appear in "
              "the table (rounding is fine); do not calculate new numbers. If the table is empty, say no records "
              "match.")
    for _ in range(2):
        answer = llm.chat(ANSWER_SYSTEM, prompt, max_tokens=300)
        bad = ungrounded(answer, df, question)
        if not bad:
            return answer, True
        prompt += f"\n\nYour previous answer used numbers not in the table: {', '.join(bad)}. Rewrite it."
    return "I computed the result table below but could not write a fully verified summary; read the figures directly from the table.", False


# ----------------------------------------------------------------------------- 7. the full pipeline
def ask(question, llm, engine, profile_text, missing, dups):
    start = time.time()
    plan = question_to_sql(llm, profile_text, question)

    if plan.get("answerable") is False:
        print(f"\nCANNOT ANSWER: {plan.get('reason') or 'this dataset does not contain the needed data.'}")
        return

    sql, df, truncated = plan["sql"], None, False
    error = plan.get("error")
    for attempt in range(MAX_REPAIRS + 1):
        if sql:
            try:
                check_sql(sql)
                df, truncated = engine.run(sql)
                break
            except Exception as exc:  # unsafe SQL or DuckDB error -> let the model repair it
                error = str(exc).split("\n")[0]
        else:
            error = error or "no query was produced"
        if attempt < MAX_REPAIRS:
            print(f"  (query failed: {error[:120]} - asking the model to fix it)")
            plan = question_to_sql(llm, profile_text, question, failed_sql=sql or "(none)", error=error)
            if plan.get("answerable") is False:
                print(f"\nCANNOT ANSWER: {plan.get('reason')}")
                return
            sql, error = plan["sql"], plan.get("error")

    if df is None:
        print(f"\nFAILED: could not compute a reliable answer (last error: {error}). No numbers are reported.")
        return

    assumptions = list(plan.get("assumptions") or [])
    if dups and not re.search(r"\bdata_raw\b", sql, re.I):
        assumptions.append(f"{dups} exact duplicate rows were removed before analysis.")
    warnings = [f'Column "{c}" has {n} missing values (excluded where used).'
                for c, n in missing.items() if n and re.search(rf'"{re.escape(c)}"|\b{re.escape(c)}\b', sql)]
    if truncated:
        warnings.append(f"Result truncated to {MAX_ROWS} rows.")

    answer, grounded = write_answer(llm, question, df, assumptions)

    print(f"\nANSWER:\n{answer}")
    print(f"\nSQL:\n{sql}")
    print(f"\nRESULT ({len(df)} rows):")
    print(df.head(20).to_string(index=False) if len(df) else "(no rows)")
    for title, items in (("ASSUMPTIONS", assumptions), ("WARNINGS", warnings)):
        if items:
            print(f"\n{title}:\n- " + "\n- ".join(items))
    print(f"\nVERIFIED: {'all numbers match the result table' if grounded else 'NO - check the table'}")
    print(f"TIME: {time.time() - start:.1f}s | {llm.speed_report()}")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a CSV with a local LLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-prompts", action="store_true", help="print the prompts sent to the model")
    args = parser.parse_args()

    profile_text, missing, dups = build_profile(CSV_PATH)
    engine = Engine(CSV_PATH)
    llm = LLM(args.model, args.show_prompts)
    print(f"Loaded {CSV_PATH}. Model: {args.model}")
    print("Enter a question, or type 'exit' to quit.")
    while True:
        try:
            question = input("\nEnter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() == "exit":
            break
        if question:
            ask(question, llm, engine, profile_text, missing, dups)


if __name__ == "__main__":
    main()


    