"""Part A6: execute and validate the seven SQL queries in sql/queries.sql."""

from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
SQL_FILE = ROOT / "sql" / "queries.sql"


def load_queries():
    """Read the seven semicolon-terminated queries, ignoring SQL comments."""
    source = SQL_FILE.read_text(encoding="utf-8")
    statements = []
    for statement in source.split(";"):
        lines = [
            line for line in statement.splitlines()
            if line.strip() and not line.lstrip().startswith("--")
        ]
        query = "\n".join(lines).strip()
        if query:
            statements.append(query)

    if len(statements) != 7:
        raise AssertionError(
            f"Expected 7 SQL queries in {SQL_FILE}, found {len(statements)}."
        )
    return statements


def run_sql_analysis():
    """Execute every rubric SQL query with pandas and check key outputs."""
    if not DB.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB}. Run python -m src.data_pipeline first."
        )

    results = []
    with sqlite3.connect(DB) as connection:
        for index, query in enumerate(load_queries(), start=1):
            # The assignment requires each SQL statement to be executed through
            # pandas.read_sql_query rather than only being stored in a .sql file.
            result = pd.read_sql_query(query, connection)
            results.append(result)
            print(f"\nQ{index}: {len(result):,} rows")
            print(result.head(10).to_string(index=False))

    q1, q2, q3, q4, q5, q6, q7 = results

    # Result-shape and data-integrity checks for the supplied synthetic dataset.
    assert len(q1) == 30, f"Q1 expected 30 rows; got {len(q1)}"
    assert len(q2) == 30, f"Q2 expected 30 rows; got {len(q2)}"
    assert round(float(q2["revenue"].sum()), 2) == 2_998_668.80
    assert len(q3) == 11, f"Q3 expected 11 rows; got {len(q3)}"
    assert len(q4) == 25, f"Q4 expected 25 rows; got {len(q4)}"
    assert len(q5) == 550, f"Q5 expected 550 rows; got {len(q5)}"
    assert round(float(q5["total_spend"].mean()), 2) == 4_516.11
    assert len(q7) == 183, f"Q7 expected 183 rows; got {len(q7)}"
    assert round(float(q7.iloc[-1]["cumulative_revenue"]), 2) == 2_998_668.80

    # RANK and DENSE_RANK can tie; ROW_NUMBER must still be unique per city.
    assert q6["row_num"].notna().all()
    assert q6.groupby("city")["row_num"].nunique().eq(
        q6.groupby("city").size()
    ).all()

    print("\nA6 SQL CHECKS PASSED: all 7 queries executed with pd.read_sql_query().")
    return results


if __name__ == "__main__":
    run_sql_analysis()
