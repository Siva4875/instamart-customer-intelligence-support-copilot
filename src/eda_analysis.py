"""Part A7: reproducible pandas analysis of completed Instamart orders."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"


def run_eda_analysis():
    """Run pivot, named aggregation, descriptive, IQR, and correlation analyses."""
    orders = pd.read_csv(CLEAN / "orders.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    customers = pd.read_csv(CLEAN / "customers.csv")

    completed = orders.loc[orders["status"].eq("completed")].copy()
    completed = completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    ).merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    if completed[["category", "city"]].isna().any().any():
        raise AssertionError("A completed order could not be mapped to its product or customer.")

    # A7.1: Pivot table of completed revenue by city and product category.
    pivot = pd.pivot_table(
        completed,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="Grand Total",
    ).round(2)

    # A7.2: Named aggregation for city-level business metrics.
    city_summary = (
        completed.groupby("city")
        .agg(
            revenue_inr=("order_amount", "sum"),
            average_order_amount_inr=("order_amount", "mean"),
            completed_orders=("order_id", "count"),
        )
        .round({"revenue_inr": 2, "average_order_amount_inr": 2})
        .sort_index()
    )

    # A7.3: Descriptive statistics and IQR outlier analysis.
    amount = completed["order_amount"]
    descriptive = amount.describe().to_frame(name="order_amount_inr")
    descriptive.loc["skewness"] = amount.skew()

    q1 = float(amount.quantile(0.25))
    q3 = float(amount.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    below_fence = int(amount.lt(lower_fence).sum())
    above_fence = int(amount.gt(upper_fence).sum())

    # A7.4: Correlation among the four numeric order variables.
    numeric_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]
    correlation = completed[numeric_columns].corr().round(3)

    # Guard against accidental changes to source data or transformations.
    assert len(completed) == 11_110, f"Expected 11,110 completed orders; got {len(completed)}"
    assert round(float(pivot.loc["Grand Total", "Grand Total"]), 2) == 2_998_668.80
    assert round(float(city_summary.loc["Hyderabad", "revenue_inr"]), 2) == 528_589.90
    assert round(float(amount.mean()), 2) == 269.91
    assert round(float(amount.median()), 2) == 135.00
    assert round(q1, 2) == 64.80 and round(q3, 2) == 297.00
    assert round(iqr, 2) == 232.20
    assert round(upper_fence, 2) == 645.30
    assert below_fence == 0 and above_fence == 897
    assert round(float(correlation.loc["order_amount", "quantity"]), 3) == 0.087

    print("\n=== PART A7.1: REVENUE PIVOT TABLE ===")
    print(pivot.to_string())

    print("\n=== PART A7.2: CITY-LEVEL NAMED AGGREGATION ===")
    print(city_summary.to_string())

    print("\n=== PART A7.3: ORDER-AMOUNT DESCRIPTIVE STATISTICS ===")
    print(descriptive.round(2).to_string())
    print(
        "\nIQR analysis: "
        f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
        f"lower_fence={lower_fence:.2f}, upper_fence={upper_fence:.2f}, "
        f"below={below_fence}, above={above_fence}"
    )

    print("\n=== PART A7.4: NUMERIC CORRELATION MATRIX ===")
    print(correlation.to_string())

    interpretation = [
        "The dataset contains synthetic Instamart-style customer, product, and order records; this analysis covers 11,110 completed orders.",
        "Order amounts are strongly right-skewed: the mean is INR 269.91, the median is INR 135.00, and skewness is 4.61. A relatively small number of high-value orders raise the mean.",
        "The IQR fences are INR -283.50 and INR 645.30. There are 897 observations above the upper fence and none below the lower fence. Investigate these high-value orders rather than removing them automatically.",
        "Order amount has weak negative correlation with discount percentage (-0.095) and weak positive correlation with quantity (0.087). These associations are small and do not establish causation.",
    ]
    word_count = sum(len(sentence.split()) for sentence in interpretation)
    assert word_count <= 200, f"EDA interpretation exceeds 200 words ({word_count})."

    print(f"\n=== PART A7.5: INTERPRETATION ({word_count} words) ===")
    for index, sentence in enumerate(interpretation, start=1):
        print(f"{index}. {sentence}")

    print("\nA7 PANDAS ANALYSIS CHECKS PASSED.")
    return {
        "pivot": pivot,
        "city_summary": city_summary,
        "descriptive": descriptive,
        "iqr": {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "below_fence": below_fence,
            "above_fence": above_fence,
        },
        "correlation": correlation,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    run_eda_analysis()
