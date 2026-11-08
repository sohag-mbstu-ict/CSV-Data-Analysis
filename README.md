# CSV Data Analysis

Ask questions about a CSV file in plain English and get answers you can check.

A local language model (through [Ollama](https://ollama.com)) turns the question into SQL, [DuckDB](https://duckdb.org) runs that SQL on the full CSV, and the model explains the result. Every number in the explanation is checked against the query result before it is shown. Each answer comes with its evidence: the SQL, the result table, assumptions, data-quality warnings and an optional bar chart.

The prototype is available three ways, all running the same pipeline:

| Interface | Use it for |
|---|---|
| **Web application** (Django) | Upload a CSV and chat with it in the browser |
| **REST API** (Django REST Framework) | Call the same features from scripts or other apps |
| **CLI tool** | Ask questions in a terminal |

The model runs locally, so no data leaves your machine.

---

## The complete flow

| Step | What happens | Where |
|---|---|---|
| **1. Load a CSV** | The CSV is loaded into an in-memory DuckDB database as `data_raw` (exactly as uploaded) and `data` (exact duplicate rows removed). A profile is built for each column: type, range, missing values, distinct count and sample values. The model sees only this profile, never the raw rows. | `utils/query_engine.py`, `utils/csv_profile.py` |
| **2. Accept a natural-language query** | The question comes from the chat box, the API or the CLI. Greetings and "what can you do?" get a direct reply with example questions. Requests to describe the dataset are answered from the profile without the model. | `utils/analysis.py` |
| **3. Generate and run the analysis** | The model returns a DuckDB `SELECT` as structured JSON, following rules for NULLs, shares of a total, per-customer counts and so on. The SQL is checked to be a single read-only `SELECT` and then run in a sandbox (no file or network access, 20 s timeout, 200-row limit). If it fails, the error goes back to the model for up to 2 fixes. Questions the data can't answer are marked unanswerable instead of guessed. | `utils/llm_client.py`, `utils/query_engine.py` |
| **4. Return a correct answer with evidence** | The model writes a 2–5 sentence answer using only the result table. Every number in it is checked against the table, allowing for rounding, K/M/B and percentages. An unsupported number triggers a rewrite; if the rewrite still fails, the answer says to read the table directly. Shares of a total are recalculated independently. The answer is returned with the SQL, result rows, assumptions, warnings and a chart when one is requested. | `utils/result_grounding.py`, `utils/visualization.py` |

```
 CSV ──► DuckDB (data / data_raw) ──► column profile ─┐
                                                      ▼
 question ──► small talk / profile? ──no──► LLM: question → SQL (JSON)
                                                      │
                          safety check ◄──────────────┘
                               │            ▲
                               ▼            │ error → repair (max 2)
                         DuckDB executes ───┘
                               │
                               ▼
                 LLM: explain the result  ──►  number check (rewrite if needed)
                               │
                               ▼
   answer + SQL + result table + assumptions + warnings + chart + "verified" flag
```

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com/download) running locally, with a model pulled

```bash
ollama pull qwen3.5:2b      # default model; a larger one such as qwen3.5:4b writes better SQL
ollama serve                # skip this if the Ollama app is already running
```

## Setup

```bash
git clone https://github.com/sohag-mbstu-ict/CSV-Data-Analysis.git
cd CSV-Data-Analysis
uv sync                     # creates .venv and installs dependencies from uv.lock
uv run python manage.py migrate
```

Without uv:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python manage.py migrate
```

`requirements.txt` is generated from `uv.lock`, so both install the same pinned versions. After changing dependencies, regenerate it with:

```bash
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o requirements.txt
```

---

## Run it

### Web application

```bash
uv run python manage.py runserver
```

Open <http://127.0.0.1:8000/>, then:

1. Drop a CSV on **Upload a CSV**. The header shows the row and column counts, removed duplicates and a column list.
2. Click a suggested question or type your own. Press Enter to send and Shift+Enter for a new line.
3. Each answer shows the explanation, any warnings, a chart when asked, and a **"Numbers verified against result"** badge. Open **SQL and result** to see the exact query, the result table and the assumptions.

Chats are saved per dataset in SQLite and stay after a reload. The server terminal also prints a full text report for every question.

### CLI tool

```bash
uv run python utils/ask_csv.py                       # uses ./data.csv
uv run python utils/ask_csv.py --csv path/to/file.csv
uv run python utils/ask_csv.py --model qwen3.5:4b --show-prompts
```

Abridged example session with the included `data.csv`:

```
Enter query: Show total revenue by sales_channel in a bar chart

ANSWER:
Online generated 3,363,572.63 in revenue, In-Store 2,170,032.61 and Mobile App 1,361,353.76.

BAR CHART: saved to charts/show_total_revenue_by_sales_channel_..._a1b2c3.png

SQL:
SELECT "sales_channel", ROUND(SUM("revenue"), 2) AS total_revenue
FROM data WHERE "sales_channel" IS NOT NULL
GROUP BY "sales_channel" ORDER BY total_revenue DESC

RESULT (3 rows):
sales_channel  total_revenue
       Online     3363572.63
     In-Store     2170032.61
   Mobile App     1361353.76

ASSUMPTIONS:
- 12 exact duplicate rows were removed before analysis.

VERIFIED: YES - answer numbers match the result table
TIME: 15.7s | 2 LLM calls | ...
```

### REST API

The base URL is `http://127.0.0.1:8000/api/`. In debug mode, opening these URLs in a browser shows DRF's browsable API.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/datasets/` | List uploaded datasets |
| `POST` | `/api/datasets/` | Upload a CSV (multipart field `file`, `.csv`, max 50 MB) |
| `GET` | `/api/datasets/{id}/` | Dataset details: columns, row count, suggested questions |
| `DELETE` | `/api/datasets/{id}/` | Delete a dataset with its file, chat and charts |
| `GET` | `/api/datasets/{id}/messages/` | Chat history |
| `POST` | `/api/datasets/{id}/ask/` | Ask a question: `{"question": "..."}` |

```bash
curl -F file=@data.csv http://127.0.0.1:8000/api/datasets/
curl -H "Content-Type: application/json" \
     -d '{"question": "Which region has the highest total profit?"}' \
     http://127.0.0.1:8000/api/datasets/1/ask/
```

`ask` returns the saved user and assistant messages. The assistant message's `payload` holds the evidence:

```jsonc
{
  "kind": "answer",            // answer | profile | churn | chat | clarify | unanswerable | failed | error
  "answer": "<2-5 sentence explanation>",
  "sql": "SELECT ...",
  "table": {"columns": ["region", "total_profit"], "rows": [["Region A", <number>], ...]},
  "row_count": 4,
  "truncated": false,
  "assumptions": ["12 exact duplicate rows were removed before analysis."],
  "warnings": ["Column \"region\" has 77 missing values (excluded where used)."],
  "notes": [],
  "grounded": true,            // every number in "answer" was found in "table"
  "chart_url": null,
  "repairs": [],               // SQL errors the model was asked to fix
  "suggestions": [],
  "elapsed_s": 12.4
}
```

If Ollama is unreachable, `ask` returns HTTP 503 with an explanation in the payload.

---

## How correctness is protected

- **The model never does the maths.** DuckDB computes every figure, and the model only writes SQL and explains the result.
- **Every number is checked.** Numbers in the explanation must appear in the result table (rounding and K/M/B/% allowed), or the answer is rewritten or replaced with "read the table".
- **Shares of a total are recalculated.** The percentage is computed again with a denominator that includes rows with a missing category, and those rows are reported.
- **Unanswerable questions are refused, not guessed.** If a needed column or period isn't in the data, the reply says why.
- **Duplicates and missing values are reported.** Exact duplicate rows are removed and noted as an assumption, and any column the query uses with missing values gets a warning.
- **Entity-level counts are handled.** Values that repeat on every order row, such as a customer's churn status, are counted per unique customer.
- **Churn reasons are associations, not causes.** "Why do customers churn?" returns a churned-versus-active comparison and a churn rate per segment, labelled as associations because the data can't prove causes.
- **Generated SQL can't modify anything.** Only one read-only `SELECT` is allowed, DuckDB's external file and network access is disabled, and queries have a timeout and a row limit.

---

## Project structure

```
├── utils/                    # Core analysis pipeline (shared by CLI and web)
│   ├── analysis.py           # analyze(): routing, SQL repair loop, evidence assembly
│   ├── csv_profile.py        # Column profile sent to the model; description-request detection
│   ├── llm_client.py         # Ollama client, SQL rules and JSON schema
│   ├── query_engine.py       # DuckDB loading, SQL safety check, bounded execution
│   ├── result_grounding.py   # Answer writing and numeric grounding check
│   ├── visualization.py      # Bar charts (matplotlib)
│   ├── report.py             # Plain-text report for the CLI and server log
│   └── ask_csv.py            # CLI entry point
├── data_analysis_app/        # REST API: models, serializers, views, engine cache, tests
├── data_analysis/            # Web frontend: template, app.js, app.css
├── config/                   # Django settings and URLs
├── data.csv                  # Sample e-commerce dataset (9,725 orders, 22 columns)
├── manage.py
└── pyproject.toml / uv.lock
```

## Configuration

Settings are read from environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.5:2b` | Model used by the web app and API (the CLI uses `--model`) |
| `CSV_MAX_UPLOAD_MB` | `50` | Largest CSV accepted for upload |
| `DJANGO_DEBUG` | `1` | `0` turns off debug mode |
| `DJANGO_SECRET_KEY` | dev placeholder | Set a private value outside local development |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated host names |
| `ANALYSIS_LOG_LEVEL` | `INFO` | `WARNING` hides the per-question report in the server terminal |

## Tests

```bash
uv run python manage.py test data_analysis_app
```

The tests cover CSV upload and validation, the duplicate-free row count, profile answers, small talk, the no-SQL clarification, the engine cache, DECIMAL results and safe chart deletion. None of them need Ollama.

## Limitations

- **Answer quality depends on the model.** A 2B model can write SQL that runs but is wrong. The number check confirms the answer matches the result table, not that the SQL is right, so the SQL is always shown for review. Larger models do noticeably better.
- **Each question stands alone.** Follow-ups like "and by month?" don't carry context from the previous question.
- **Only bar charts are drawn.** Other chart types are not supported.
- **There is no login.** It's a local, single-user tool, so add authentication before running it on a shared network.
- **Answers take time.** A local model often needs 10–40 seconds per answer, depending on hardware.
