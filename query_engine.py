"""Read-only DuckDB execution for CSV analysis."""

import re
import threading

import duckdb
import pandas as pd

MAX_ROWS = 200
QUERY_TIMEOUT_S = 20

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|copy|export|install|load|pragma|set|call)\b"
    r"|\b(read_\w+|glob|parquet_\w+|csv_scan)\s*\(|\b(from|join)\s+'",
    re.I,
)


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
        return pd.DataFrame(rows[:MAX_ROWS], columns=columns), len(rows) > MAX_ROWS

    def column_types(self):
        """Return the loaded table's column names and DuckDB type names."""
        return {
            row[0]: row[1]
            for row in self.con.execute("DESCRIBE data").fetchall()
        }

    def row_count(self, table="data"):
        """Return the number of rows in `data` (deduplicated) or `data_raw`."""
        if table not in ("data", "data_raw"):
            raise ValueError("Unknown table")
        return self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

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
