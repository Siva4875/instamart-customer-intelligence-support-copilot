"""
Part B: Customer Segmentation and Lapse Prediction.

This is a readability refactor of the existing implementation. It preserves
the existing public function names, fixed dates, random seeds, model grids,
feature definitions, and model-selection behaviour.

Run the functions from the repository root:
    from src.customer_models import rfm_and_segments, build_lapse_model
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# 1. Paths, dates, and feature configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
LAPSE_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LAPSE_WINDOW_END = pd.Timestamp("2025-10-01 00:00:00")

RFM_COLUMNS = ["recency", "frequency", "monetary"]

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "favourite_category",
    "city",
    "gender",
]

LAPSE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================================
# 2. Data loading
# ============================================================================

def load():
    """Load the clean customer and order tables."""
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    return customers, orders


def load_product_categories():
    """Load product IDs and category names for favourite-category features."""
    products = pd.read_csv(CLEAN / "products.csv")
    return products[["product_id", "category"]]


def ensure_output_directory():
    """Create outputs/ when it does not already exist."""
    OUT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 3. RFM table and K-means segmentation
# ============================================================================

def build_rfm_table(orders):
    """Calculate recency, frequency, and monetary value from completed orders."""
    completed = orders.loc[orders["status"].eq("completed")].copy()

    rfm = (
        completed
        .groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days

    return rfm.drop(columns=["last_order"])


def evaluate_kmeans(rfm):
    """Fit StandardScaler and calculate inertia/silhouette for K=2,...,8."""
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[RFM_COLUMNS])

    diagnostics = []

    for number_of_clusters in range(2, 9):
        model = KMeans(
            n_clusters=number_of_clusters,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(scaled_rfm)

        diagnostics.append(
            {
                "k": number_of_clusters,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled_rfm, labels),
            }
        )

    return scaled_rfm, pd.DataFrame(diagnostics)


def build_segment_profile(rfm):
    """Calculate the size and mean RFM values for each cluster."""
    return (
        rfm
        .groupby("segment_id")
        .agg(
            size=("customer_id", "size"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .reset_index()
    )



def assign_segment_names(rfm, profile):
    """
    Assign four distinct business names to the customer segments.

    Premium: highest average monetary value.
    Dormant: highest average recency among remaining segments.
    Frequent: highest average frequency among remaining segments.
    Occasional: remaining segment.

    Cluster IDs are arbitrary labels and do not imply ranking.
    """
    remaining_ids = set(profile["segment_id"].astype(int))

    # Identify the highest-spending segment.
    premium_segment_id = int(
        profile.loc[profile["monetary"].idxmax(), "segment_id"]
    )
    remaining_ids.remove(premium_segment_id)

    # Identify the least-recently-active segment.
    remaining_profile = profile[
        profile["segment_id"].isin(remaining_ids)
    ]
    dormant_segment_id = int(
        remaining_profile.loc[
            remaining_profile["recency"].idxmax(),
            "segment_id",
        ]
    )
    remaining_ids.remove(dormant_segment_id)

    # Identify the most frequent remaining segment.
    remaining_profile = profile[
        profile["segment_id"].isin(remaining_ids)
    ]
    frequent_segment_id = int(
        remaining_profile.loc[
            remaining_profile["frequency"].idxmax(),
            "segment_id",
        ]
    )
    remaining_ids.remove(frequent_segment_id)

    # Assign the remaining segment as Occasional.
    occasional_segment_id = int(next(iter(remaining_ids)))

    name_map = {
        premium_segment_id: "Premium",
        dormant_segment_id: "Dormant",
        frequent_segment_id: "Frequent",
        occasional_segment_id: "Occasional",
    }

    rfm["segment_name"] = rfm["segment_id"].map(name_map)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_segment_id)

    return rfm


def save_customer_segments(rfm):
    """Write the required customer_segments.csv file."""
    ensure_output_directory()

    columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]

    output_path = OUT / "customer_segments.csv"
    rfm[columns].to_csv(output_path, index=False)

    return output_path


def rfm_and_segments(k=4):
    """
    Run RFM segmentation and PCA.

    Returns:
        rfm: Customer-level RFM values and segment assignments.
        diagnostics: K-means inertia and silhouette table for K=2,...,8.
        profile: Segment sizes and mean RFM values.
        explained_variance_ratio: PCA explained-variance ratios.
    """
    _, orders = load()

    rfm = build_rfm_table(orders)
    scaled_rfm, diagnostics = evaluate_kmeans(rfm)

    final_model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )
    rfm["segment_id"] = final_model.fit_predict(scaled_rfm)

    profile = build_segment_profile(rfm)
    rfm = assign_segment_names(rfm, profile)

    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)

    save_customer_segments(rfm)

    return (
        rfm,
        diagnostics,
        profile,
        pca.explained_variance_ratio_,
    )


# ============================================================================
# 4. Lapse cohort, historical features, and target label
# ============================================================================

def build_lapse_cohort(customers, orders):
    """
    Build the lapse modelling cohort.

    All predictor features use orders strictly before 2025-09-01.
    The lapsed label uses completed orders from 2025-09-01 through
    2025-09-30. Segment labels are not included as model features.
    """
    historical_orders = orders.loc[
        orders["order_ts"] < LAPSE_CUTOFF
    ].copy()

    completed_history = historical_orders.loc[
        historical_orders["status"].eq("completed")
    ].copy()

    # Customers enter the cohort only if they have a completed pre-cutoff order.
    cohort = (
        completed_history
        .groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount=("discount_pct", "mean"),
            avg_delivery=("delivery_minutes", "mean"),
        )
        .reset_index()
    )

    cohort["recency"] = (LAPSE_CUTOFF - cohort["last_order"]).dt.days
    cohort = cohort.drop(columns=["last_order"])

    # Cancellation rate: cancelled orders / all completed or cancelled
    # orders before the cutoff.
    final_status_orders = historical_orders.loc[
        historical_orders["status"].isin(["completed", "cancelled"])
    ]

    final_status_counts = (
        final_status_orders
        .groupby("customer_id")["status"]
        .agg(
            total="count",
            cancelled=lambda values: values.eq("cancelled").sum(),
        )
        .reset_index()
    )

    cohort = cohort.merge(
        final_status_counts,
        on="customer_id",
        how="left",
    )

    cohort["cancellation_rate"] = (
        cohort["cancelled"] / cohort["total"]
    )

    # Most frequently ordered product category; alphabetical order breaks ties.
    product_categories = load_product_categories()

    category_counts = (
        completed_history
        .merge(product_categories, on="product_id", how="left")
        .groupby(["customer_id", "category"])
        .size()
        .rename("order_count")
        .reset_index()
    )

    favourite_category = (
        category_counts
        .sort_values(
            ["customer_id", "order_count", "category"],
            ascending=[True, False, True],
        )
        .drop_duplicates("customer_id")
        .rename(columns={"category": "favourite_category"})
        [["customer_id", "favourite_category"]]
    )

    cohort = cohort.merge(
        favourite_category,
        on="customer_id",
        how="left",
    )
    cohort = cohort.merge(
        customers,
        on="customer_id",
        how="left",
    )

    cohort["signup_date"] = pd.to_datetime(cohort["signup_date"])
    cohort["tenure_days"] = (
        LAPSE_CUTOFF - cohort["signup_date"]
    ).dt.days

    # September activity is used only to construct the target label.
    september_completed = orders.loc[
        orders["order_ts"].ge(LAPSE_CUTOFF)
        & orders["order_ts"].lt(LAPSE_WINDOW_END)
        & orders["status"].eq("completed")
    ]

    september_active_ids = set(september_completed["customer_id"])

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(september_active_ids)
    ).astype(int)

    # Remove columns that are not model predictors or the target.
    cohort = cohort.drop(
        columns=["signup_date", "total", "cancelled"]
    )

    return cohort


# ============================================================================
# 5. Train/test split and preprocessing pipeline
# ============================================================================

def split_lapse_data(cohort):
    """Create the required stratified 80/20 split."""
    X = cohort[LAPSE_FEATURES].copy()
    y = cohort["lapsed"].copy()

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )


def build_preprocessor():
    """
    Build preprocessing steps to be fitted on training rows only.

    Numeric columns are median-imputed and standardised. Categorical columns
    are imputed with 'unknown' and one-hot encoded with drop='first'.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


# ============================================================================
# 6. Model tuning and test-set evaluation
# ============================================================================

def get_model_configurations():
    """Return the two required estimators and their parameter grids."""
    return [
        (
            "logistic",
            LogisticRegression(max_iter=1000),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
            },
        ),
        (
            "forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [5, 10, 20],
            },
        ),
    ]


def evaluate_model(search, X_test, y_test):
    """Calculate positive-class metrics and confusion matrix on the test set."""
    predictions = search.predict(X_test)

    return {
        "best_params": search.best_params_,
        "cv_f1": search.best_score_,
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
        "search": search,
    }


def tune_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Run both five-fold GridSearchCV experiments."""
    preprocessor = build_preprocessor()
    results = {}

    for model_name, estimator, parameter_grid in get_model_configurations():
        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        # Grid search sees training rows only.
        search.fit(X_train, y_train)

        # Test rows are used only after tuning has completed.
        results[model_name] = evaluate_model(
            search,
            X_test,
            y_test,
        )

    return results


# ============================================================================
# 7. Model persistence and Premium retention list
# ============================================================================

def run_logistic_oversampling(
    X_train,
    X_test,
    y_train,
    y_test,
    best_logistic_search,
):
    """
    Oversample only the training data and evaluate Logistic Regression
    against the original, unchanged test set.
    """
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    # Identify the majority and minority classes in training data.
    class_counts_before = y_train.value_counts().sort_index()
    majority_count = int(class_counts_before.max())
    minority_class = int(class_counts_before.idxmin())
    minority_count = int(class_counts_before.min())

    # Select only minority-class training rows.
    minority_indices = y_train[
        y_train == minority_class
    ].index

    minority_features = X_train.loc[minority_indices]

    # Duplicate minority-class rows with replacement.
    additional_rows = minority_features.sample(
        n=majority_count - minority_count,
        replace=True,
        random_state=42,
    )

    # Combine original training rows and sampled rows.
    X_balanced = pd.concat(
        [
            X_train.reset_index(drop=True),
            additional_rows.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    y_balanced = pd.concat(
        [
            y_train.reset_index(drop=True),
            pd.Series(
                [minority_class] * len(additional_rows),
                name=y_train.name,
            ),
        ],
        ignore_index=True,
    )

    # Confirm that both training classes are now balanced.
    class_counts_after = y_balanced.value_counts().sort_index()

    assert class_counts_after.nunique() == 1, (
        "Oversampling did not produce balanced classes."
    )

    # Clone the best tuned Logistic Regression pipeline.
    # Its preprocessing is fitted again using balanced training data only.
    balanced_model = clone(
        best_logistic_search.best_estimator_
    )

    balanced_model.fit(X_balanced, y_balanced)

    # Evaluate on the original, unchanged test set.
    predictions = balanced_model.predict(X_test)

    return {
        "class_counts_before": class_counts_before.to_dict(),
        "class_counts_after": class_counts_after.to_dict(),
        "training_rows_after_oversampling": len(X_balanced),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }


def extract_feature_insights(model_results):
    """Extract Random Forest importance and Logistic Regression coefficients."""

    # Random Forest feature importance
    forest_pipeline = model_results["forest"]["search"].best_estimator_

    forest_preprocessor = forest_pipeline.named_steps["prep"]
    forest_model = forest_pipeline.named_steps["model"]

    forest_feature_names = forest_preprocessor.get_feature_names_out()
    forest_importances = forest_model.feature_importances_

    top_indices = np.argsort(forest_importances)[::-1][:5]

    top_forest_features = [
        {
            "feature": str(forest_feature_names[index]),
            "importance": float(forest_importances[index]),
        }
        for index in top_indices
    ]

    # Logistic Regression coefficient interpretation
    logistic_pipeline = model_results["logistic"]["search"].best_estimator_

    logistic_preprocessor = logistic_pipeline.named_steps["prep"]
    logistic_model = logistic_pipeline.named_steps["model"]

    logistic_feature_names = logistic_preprocessor.get_feature_names_out()
    logistic_coefficients = logistic_model.coef_[0]

    top_coefficient_indices = np.argsort(
        np.abs(logistic_coefficients)
    )[::-1][:3]

    top_logistic_coefficients = [
        {
            "feature": str(logistic_feature_names[index]),
            "coefficient": float(logistic_coefficients[index]),
            "direction": (
                "Positive" if logistic_coefficients[index] > 0
                else "Negative"
            ),
        }
        for index in top_coefficient_indices
    ]

    return {
        "random_forest_top_5": top_forest_features,
        "logistic_regression_top_3_coefficients": (
            top_logistic_coefficients
        ),
    }


def save_and_verify_model(selected_model, X_test):
    """Persist the selected model and verify its reloaded predictions."""
    ensure_output_directory()

    model_path = OUT / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)

    reloaded_model = joblib.load(model_path)

    original_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)

    assert np.array_equal(
        original_predictions,
        reloaded_predictions,
    ), "Reloaded model predictions differ from original predictions."

    return model_path


def create_retention_list(
    selected_model,
    X_test,
    y_test,
    test_customer_ids,
    premium_customer_ids,
):
    """
    Save up to ten Premium test-set customers with the highest lapse score.

    The Premium flag is a post-hoc whole-window label. The resulting list is
    illustrative and is not a leakage-free evaluation.
    """
    probabilities = selected_model.predict_proba(X_test)
    positive_class_index = list(selected_model.classes_).index(1)

    lapse_probabilities = probabilities[:, positive_class_index]

    retention_list = pd.DataFrame(
        {
            "customer_id": test_customer_ids,
            "lapse_probability": lapse_probabilities,
            "actually_lapsed": y_test.to_numpy(),
        }
    )

    retention_list = (
        retention_list.loc[
            retention_list["customer_id"].isin(premium_customer_ids)
        ]
        .sort_values(
            "lapse_probability",
            ascending=False,
        )
        .head(10)
        .reset_index(drop=True)
    )

    ensure_output_directory()
    retention_list.to_csv(
        OUT / "retention_list.csv",
        index=False,
    )

    return retention_list


# ============================================================================
# 8. Main lapse-model workflow (existing public function)
# ============================================================================

def build_lapse_model():
    """
    Build, tune, evaluate, and persist the lapse model.

    The return dictionary retains the same top-level keys as the original
    implementation for compatibility with existing project code.
    """
    customers, orders = load()

    cohort = build_lapse_cohort(customers, orders)

    X_train, X_test, y_train, y_test = split_lapse_data(cohort)

    model_results = tune_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    oversampling_results = run_logistic_oversampling(
        X_train,
        X_test,
        y_train,
        y_test,
        model_results["logistic"]["search"],
    )

    feature_insights = extract_feature_insights(model_results)

    # Preserve the original implementation's selected model: random forest.
    selected_model = model_results["forest"]["search"]

    model_path = save_and_verify_model(
        selected_model,
        X_test,
    )

    # Map test rows back to their customer IDs using the original cohort index.
    test_customer_ids = cohort.loc[
        X_test.index,
        "customer_id",
    ].to_numpy()

    # Premium is derived from whole-window RFM and is post-hoc.
    segments, _, _, _ = rfm_and_segments()
    premium_customer_ids = set(
        segments.loc[segments["is_premium"], "customer_id"]
    )

    retention_list = create_retention_list(
        selected_model=selected_model,
        X_test=X_test,
        y_test=y_test,
        test_customer_ids=test_customer_ids,
        premium_customer_ids=premium_customer_ids,
    )

    # Keep the original result structure, excluding fitted search objects.
    reportable_models = {
        model_name: {
            key: value
            for key, value in result.items()
            if key != "search"
        }
        for model_name, result in model_results.items()
    }

    baseline_predictions = np.zeros(
        len(y_test),
        dtype=int,
    )

    return {
        "cohort_rows": len(cohort),
        "lapsed": int(cohort["lapsed"].sum()),
        "baseline_accuracy": accuracy_score(
            y_test,
            baseline_predictions,
        ),
        "models": reportable_models,
        "oversampling": oversampling_results,
        "feature_insights": feature_insights,
        "retention_list": retention_list,
        "model_path": model_path,
    }


# Note:
# The original implementation does not run automatically on import.
# Call rfm_and_segments() and build_lapse_model() from your project runner.
