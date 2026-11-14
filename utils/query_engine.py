"""Read-only DuckDB execution for CSV analysis."""

import re
import threading
from decimal import Decimal

import duckdb
import pandas as pd

MAX_ROWS = 200
QUERY_TIMEOUT_S = 20

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|copy|export|install|load|pragma|set|call)\b"
    r"|\b(read_\w+|glob|parquet_\w+|csv_scan)\s*\(|\b(from|join)\s+'",
    re.I,
)


QUOTED = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
# A CTE definition: `WITH name AS (` or `, name AS (`, optionally with a column list.
CTE_DEFINITION = re.compile(
    r"(?:\bwith\s+(?:recursive\s+)?|,\s*)"
    r"(\"_*\"|[A-Za-z_]\w*)\s*(?:\([^()]*\)\s*)?as\s*(?:(?:not\s+)?materialized\s*)?\(",
    re.I,
)
INCOMPLETE_CTE = (
    "Incomplete query: it defines a WITH block but has no final SELECT after it. "
    "Add a SELECT that reads from the CTE after its closing parenthesis, "
    "or write a plain SELECT without WITH."
)


def _mask_quotes(sql):
    """Blank out quoted strings and identifiers, keeping every character position."""
    return QUOTED.sub(lambda m: m.group(0)[0] + "_" * (len(m.group(0)) - 2) + m.group(0)[-1], sql)


def _strip_end(sql):
    return sql.rstrip().rstrip(";").rstrip()


def _top_level_text(masked):
    """Return only the characters outside any parentheses."""
    depth, kept = 0, []
    for char in masked:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0:
            kept.append(char)
    return "".join(kept)


def _has_final_select(masked):
    """A WITH query must have a SELECT outside the CTE bodies."""
    if not re.match(r"\s*with\b", masked, re.I):
        return True
    return bool(re.search(r"\bselect\b", _top_level_text(masked), re.I))


def dangling_cte_name(sql):
    """Return the last CTE's name if the query ends right after its WITH block, else None."""
    sql = _strip_end(sql)
    masked = _mask_quotes(sql)
    if _has_final_select(masked):
        return None
    last = None
    for match in CTE_DEFINITION.finditer(masked):
        before = masked[:match.start()]
        if before.count("(") == before.count(")"):
            last = match
    if last is None:
        return None
    # The last CTE body must run to the very end of the query.
    depth = 0
    for index in range(last.end() - 1, len(masked)):
        if masked[index] == "(":
            depth += 1
        elif masked[index] == ")":
            depth -= 1
            if depth == 0:
                return sql[last.start(1):last.end(1)] if index == len(masked) - 1 else None
    return None


def complete_dangling_cte(sql):
    """Add `SELECT * FROM <last cte>` to a query that stops after its WITH block.

    Small models sometimes write a correct CTE and then stop. Returns (sql, cte_name);
    cte_name is None when the query was left unchanged.
    """
    name = dangling_cte_name(sql)
    if name is None:
        return sql, None
    return f"{_strip_end(sql)}\nSELECT * FROM {name}", name


def check_sql(sql):
    """Allow one read-only SELECT; DuckDB's sandbox is a second safety layer."""
    code = re.sub(r"'(?:[^']|'')*'", "''", sql)
    if not re.match(r"^\s*(select|with)\b", code, re.I):
        raise ValueError("Only SELECT queries are allowed")
    if ";" in code.strip().rstrip(";"):
        raise ValueError("Only one statement is allowed")
    bad = FORBIDDEN.search(code)
    if bad:
        raise ValueError(f"Forbidden SQL: {bad.group(0)}")
    if not _has_final_select(_mask_quotes(_strip_end(sql))):
        raise ValueError(INCOMPLETE_CTE)


class Engine:
    """Load a CSV into in-memory DuckDB and execute bounded queries."""

    def __init__(self, csv_path):
        self.con = duckdb.connect(":memory:")
        path = str(csv_path).replace("'", "''")
        self.con.execute(
            f"CREATE TABLE data_raw AS SELECT * FROM read_csv_auto('{path}', sample_size=-1)"
        )
        self.con.execute("CREATE TABLE data AS SELECT DISTINCT * FROM data_raw")
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    def run(self, sql):
        cursor = self.con.cursor()
        timer = threading.Timer(QUERY_TIMEOUT_S, cursor.interrupt)
        timer.start()
        try:
            cursor.execute(sql)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
        finally:
            timer.cancel()
            cursor.close()
        frame = pd.DataFrame(rows[:MAX_ROWS], columns=columns)
        # DuckDB DECIMAL values arrive as Python Decimal objects, which pandas stores as
        # non-numeric "object" columns; charts and number checks need real floats.
        for column in frame.columns:
            values = frame[column].dropna()
            if len(values) and frame[column].dtype == object and all(isinstance(v, Decimal) for v in values):
                frame[column] = frame[column].astype(float)
        return frame, len(rows) > MAX_ROWS

    def column_types(self):
        """Return the loaded table's column names and DuckDB type names."""
        return {
            row[0]: row[1]
            for row in self.con.execute("DESCRIBE data").fetchall()
        }

    def row_count(self, table="data"):
        """Return the row count for a known in-memory table."""
        if table not in {"data", "data_raw"}:
            raise ValueError("Only the loaded data and data_raw tables are available")
        return self.con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    def share_summary(self, category_column, category_value, measure_column):
        """Compute category and full-dataset totals for a share, including missing categories."""
        column_types = self.column_types()
        if category_column not in column_types or measure_column not in column_types:
            raise ValueError("Share summary refers to a column that is not in the dataset")

        category_identifier = '"' + category_column.replace('"', '""') + '"'
        measure_identifier = '"' + measure_column.replace('"', '""') + '"'
        query = f"""
            SELECT
                SUM({measure_identifier}) FILTER (WHERE {category_identifier} = ?) AS category_amount,
                SUM({measure_identifier}) AS overall_amount,
                SUM({measure_identifier}) FILTER (WHERE {category_identifier} IS NULL) AS unassigned_amount,
                COUNT(*) FILTER (WHERE {category_identifier} IS NULL) AS unassigned_rows
            FROM data
        """
        row = self.con.execute(query, [category_value]).fetchone()
        category_amount, overall_amount, unassigned_amount, unassigned_rows = row
        share_pct = (
            float(category_amount) / float(overall_amount) * 100
            if category_amount is not None and overall_amount
            else None
        )
        return {
            "category_amount": category_amount,
            "overall_amount": overall_amount,
            "unassigned_amount": unassigned_amount,
            "unassigned_rows": unassigned_rows,
            "share_pct": share_pct,
        }

    def churn_associations(self):
        """Summarize customer-level differences associated with churn, not causal factors."""
        customer_cte = """
            WITH customer_level AS (
                SELECT
                    customer_id,
                    MAX(CASE WHEN churned THEN 1 ELSE 0 END) AS is_churned,
                    ANY_VALUE(customer_segment) AS customer_segment,
                    AVG(customer_age) AS avg_age,
                    AVG(satisfaction_score) AS avg_satisfaction,
                    AVG(support_tickets) AS avg_tickets_per_order,
                    COUNT(*) AS order_count
                FROM data
                WHERE customer_id IS NOT NULL
                GROUP BY customer_id
            )
        """
        cohorts = self.con.execute(
            customer_cte + """
                SELECT
                    is_churned,
                    COUNT(*) AS customers,
                    ROUND(AVG(avg_age), 2) AS avg_customer_age,
                    ROUND(AVG(avg_satisfaction), 2) AS avg_satisfaction_per_customer,
                    ROUND(AVG(avg_tickets_per_order), 2) AS avg_tickets_per_order_per_customer,
                    ROUND(AVG(order_count), 2) AS avg_orders_per_customer
                FROM customer_level
                GROUP BY is_churned
                ORDER BY is_churned
            """
        ).fetchdf()
        segments = self.con.execute(
            customer_cte + """
                SELECT
                    customer_segment,
                    COUNT(*) AS customers,
                    SUM(is_churned) AS churned_customers,
                    ROUND(100.0 * SUM(is_churned) / COUNT(*), 2) AS churn_rate_pct
                FROM customer_level
                WHERE customer_segment IS NOT NULL
                GROUP BY customer_segment
                ORDER BY churn_rate_pct DESC
            """
        ).fetchdf()
        return cohorts, segments

    def close(self):
        """Close the in-memory DuckDB connection."""
        self.con.close()
