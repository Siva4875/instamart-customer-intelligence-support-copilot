"""Part A: audit, clean, validate, and load data into SQLite."""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"


# ---------------------------------------------------------------------------
# Raw-data audit
# ---------------------------------------------------------------------------

def audit():
    """Print dataset shapes, dtypes, missing values, and key quality checks."""
    for name in ["customers", "products", "orders"]:
        df = pd.read_csv(RAW / f"{name}.csv", keep_default_na=False)

        print(f"\n{name}: shape={df.shape}")
        print(df.dtypes)

        # Count both pandas missing values and blank/whitespace-only strings.
        missing = df.isna() | df.astype("string").apply(
            lambda column: column.str.strip().eq("")
        )
        missing_summary = pd.DataFrame(
            {
                "missing_count": missing.sum(),
                "missing_pct": (missing.mean() * 100).round(2),
            }
        )
        print(missing_summary)

    orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)
    customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)

    print(
        "\nOrders audit:",
        {
            "rows": len(orders),
            "exact_duplicates": int(orders.duplicated().sum()),
            "pending": int(orders.status.eq("pending").sum()),
            "distinct_order_id": int(orders.order_id.nunique()),
            "bad_amount": int(
                pd.to_numeric(orders.order_amount, errors="coerce").isna().sum()
            ),
            "negative_delivery": int(
                pd.to_numeric(orders.delivery_minutes, errors="coerce")
                .lt(0)
                .sum()
            ),
            "blank_delivery": int(
                orders.delivery_minutes.astype(str).str.strip().eq("").sum()
            ),
        },
    )

    print(
        "Customer audit:",
        {
            "blank_age": int(
                customers.age.astype(str).str.strip().eq("").sum()
            ),
            "age_outside_10_100": int(
                pd.to_numeric(customers.age, errors="coerce")
                .pipe(lambda values: ((values < 10) | (values > 100)).sum())
            ),
            "raw_city_spellings": int(customers.city.nunique()),
            "raw_gender_spellings": int(
                customers.loc[
                    customers.gender.astype(str).str.strip().ne(""),
                    "gender",
                ].nunique()
            ),
            "blank_gender": int(
                customers.gender.astype(str).str.strip().eq("").sum()
            ),
        },
    )


# ---------------------------------------------------------------------------
# Cleaning and persistence
# ---------------------------------------------------------------------------

def clean_data():
    """Clean the customer and order tables, then save CSV and SQLite outputs."""
    CLEAN.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

    # Standardize customer text fields and normalize unknown gender values.
    customers["city"] = (
        customers.city.astype(str).str.strip().str.title()
    )
    customers["gender"] = (
        customers.gender.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "male": "male",
                "m": "male",
                "female": "female",
                "f": "female",
            }
        )
        .fillna("unknown")
    )

    # Treat invalid ages as missing, then impute from the customer's city median.
    customers["age"] = pd.to_numeric(customers.age, errors="coerce")
    customers.loc[~customers.age.between(10, 100), "age"] = np.nan
    customers["age"] = (
        customers.age
        .fillna(customers.groupby("city").age.transform("median"))
        .round()
        .astype(int)
    )
    customers["signup_date"] = (
        pd.to_datetime(customers.signup_date).dt.strftime("%Y-%m-%d")
    )

    # Parse timestamps before resolving duplicate order records.
    orders["order_ts"] = pd.to_datetime(orders.order_ts)
    orders = orders.drop_duplicates()

    # Latest timestamp wins when the same order appears as pending and final.
    orders = (
        orders.sort_values(["order_id", "order_ts"])
        .drop_duplicates("order_id", keep="last")
        .copy()
    )

    # Convert numeric columns and use zero for missing discount percentages.
    orders["quantity"] = pd.to_numeric(orders.quantity, errors="coerce")
    orders["discount_pct"] = (
        pd.to_numeric(orders.discount_pct, errors="coerce").fillna(0)
    )
    orders["order_amount"] = pd.to_numeric(
        orders.order_amount, errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders.delivery_minutes, errors="coerce"
    )

    # Negative delivery durations are invalid and are treated as missing.
    orders.loc[orders.delivery_minutes < 0, "delivery_minutes"] = np.nan

    # Recalculate missing order amounts from catalogue price, quantity, and discount.
    price = products.set_index("product_id").price_inr
    missing = orders.order_amount.isna()
    orders.loc[missing, "order_amount"] = (
        orders.loc[missing, "product_id"].map(price)
        * orders.loc[missing, "quantity"]
        * (1 - orders.loc[missing, "discount_pct"] / 100)
    ).round(2)

    # Impute missing delivery time only for completed orders, by product category.
    orders["category"] = orders.product_id.map(
        products.set_index("product_id").category
    )
    completed = orders.status.eq("completed")

    # Impute missing delivery times only for completed orders. The category
    # median is calculated from completed orders and broadcast back to each
    # completed row with groupby().transform(), preserving row alignment.
    completed_delivery = orders.loc[completed, "delivery_minutes"]
    category_medians = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )
    orders.loc[completed, "delivery_minutes"] = (
        completed_delivery.fillna(category_medians)
    )

    # Cancelled orders remain missing: no delivery took place.

    orders["delivery_minutes"] = (
        orders.delivery_minutes.round().astype("Int64")
    )
    orders = orders.drop(columns=["category"])
    orders["order_ts"] = orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    # Persist the three cleaned tables as CSV files.
    for name, dataframe in [
        ("customers", customers),
        ("products", products),
        ("orders", orders),
    ]:
        dataframe.to_csv(CLEAN / f"{name}.csv", index=False)

    # Persist the same cleaned tables in SQLite.
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as connection:
        for name, dataframe in [
            ("customers", customers),
            ("products", products),
            ("orders", orders),
        ]:
            dataframe.to_sql(name, connection, if_exists="replace", index=False)

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
    }


# ---------------------------------------------------------------------------
# Data and database validation
# ---------------------------------------------------------------------------


def validate():
    """Assert the Part A reference checkpoints and verify SQLite tables."""
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv", keep_default_na=False)

    # ---------------------------------------------------------------
    # Customer reference checkpoints
    # ---------------------------------------------------------------

    expected_city_counts = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }

    expected_gender_counts = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }

    assert len(customers) == 1200
    assert customers.city.value_counts().to_dict() == expected_city_counts
    assert customers.gender.value_counts().to_dict() == expected_gender_counts
    assert customers.age.notna().all()
    assert customers.age.between(18, 60).all()

    # ---------------------------------------------------------------
    # Order reference checkpoints
    # ---------------------------------------------------------------

    assert len(orders) == 12309
    assert orders.order_id.is_unique

    expected_status_counts = {
        "completed": 11110,
        "cancelled": 1199,
    }

    assert orders.status.value_counts().to_dict() == expected_status_counts
    assert not orders.status.eq("pending").any()

    order_amount = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    assert order_amount.notna().all()

    completed = orders.status.eq("completed")
    cancelled = orders.status.eq("cancelled")

    completed_delivery = pd.to_numeric(
        orders.loc[completed, "delivery_minutes"],
        errors="coerce",
    )
    cancelled_delivery = pd.to_numeric(
        orders.loc[cancelled, "delivery_minutes"],
        errors="coerce",
    )

    assert completed_delivery.notna().all()
    assert completed_delivery.isna().sum() == 0
    assert cancelled_delivery.isna().all()
    assert cancelled_delivery.isna().sum() == 1199

    completed_revenue = order_amount.loc[completed].sum()
    assert np.isclose(
        completed_revenue,
        2998668.80,
        atol=0.01,
    )

    # ---------------------------------------------------------------
    # SQLite table and row-count checkpoints
    # ---------------------------------------------------------------

    expected_table_counts = {
        "customers": 1200,
        "products": 40,
        "orders": 12309,
    }

    with sqlite3.connect(DB) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        assert set(expected_table_counts) <= table_names

        for table_name, expected_count in expected_table_counts.items():
            actual_count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            assert actual_count == expected_count, (
                f"{table_name}: expected {expected_count} rows, "
                f"found {actual_count}"
            )

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    audit()
    clean_data()
    validate()
