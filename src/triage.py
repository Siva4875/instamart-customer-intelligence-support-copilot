"""Part C: privacy-first support-ticket triage pipeline.

Run from the project root:
    python -m src.triage
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

import mock_llm


# 1. Paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TICKETS_FILE = RAW_DATA_DIR / "tickets.csv"
CUSTOMER_SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
TOKEN_BUDGET = 100_000
BUDGET_DEMO_TOKEN_BUDGET = 6_000

BASE_DELAY = 0.05
MAX_RETRIES = 3

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

OUTPUT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]


# 2. System prompts
ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket.
Allowed categories: billing, delivery, technical, fraud.
Return a JSON-only reply in exactly this form:
{"category": "...", "reason": "..."}
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket.
Allowed categories: billing, delivery, technical, fraud.
Return a JSON-only reply in exactly this form:
{"category": "...", "reason": "..."}

EXAMPLES:
Ticket: I noticed an unauthorized transaction on my account.
Reply: {"category": "fraud", "reason": "unauthorized transaction"}

Ticket: My order arrived later than the promised delivery time.
Reply: {"category": "delivery", "reason": "late delivery"}

Ticket: The application crashes whenever I try to log in.
Reply: {"category": "technical", "reason": "application crash"}
""".strip()


# 3. Privacy: redact and verify personal information
REDACTION_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact_personal_information(text: str) -> tuple[str, dict[str, int]]:
    """Redact email, phone, and order ID patterns in the required order."""
    redacted_text = text
    substitution_counts: dict[str, int] = {}

    for name, pattern, placeholder in REDACTION_PATTERNS:
        redacted_text, count = pattern.subn(placeholder, redacted_text)
        substitution_counts[name] = count

    return redacted_text, substitution_counts


def verify_redaction(redacted_text: str) -> bool:
    """Return True only if none of the PII patterns remain."""
    return all(
        pattern.search(redacted_text) is None
        for _, pattern, _ in REDACTION_PATTERNS
    )


# 4. Input guardrails
def is_empty_ticket(text: str) -> bool:
    """Check for empty or whitespace-only ticket text."""
    return not text.strip()


def is_long_ticket(text: str) -> bool:
    """Check whether the estimated input exceeds the 400-token limit."""
    return mock_llm.est_tokens(text) > MAX_INPUT_TOKENS


# 5. Resilient model calls
def call_with_retry(messages: list[dict[str, str]], ticket_id: str) -> Any:
    """Retry transient model failures up to three times after the first call."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )
            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={retry_number} "
                f"wait_seconds={wait_seconds:.4f}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Retry loop ended unexpectedly for {ticket_id}")


# 6. Validate every model response
def validate_model_response(response_text: str) -> tuple[str, str]:
    """Return (category, status), using keywords only for non-JSON replies."""
    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError:
        words = re.findall(r"[a-z]+", response_text.lower())
        found_categories = [
            word for word in words if word in ALLOWED_CATEGORIES
        ]

        if len(found_categories) == 1:
            return found_categories[0], "classified_keyword"

        return "", "flagged_invalid_output"

    # Valid JSON with an unsupported/missing category is flagged, not guessed.
    if not isinstance(parsed_response, dict):
        return "", "flagged_invalid_output"

    category = parsed_response.get("category")
    if isinstance(category, str) and category in ALLOWED_CATEGORIES:
        return category, "classified_json"

    return "", "flagged_invalid_output"


# 7. Load customer segments and apply routing
def load_customer_segments() -> dict[str, dict[str, Any]]:
    """Load Part B segments keyed by customer_id."""
    if not CUSTOMER_SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Part B segment file not found: {CUSTOMER_SEGMENTS_FILE}"
        )

    segments = pd.read_csv(
        CUSTOMER_SEGMENTS_FILE,
        dtype={"customer_id": str},
    )

    required_columns = {"customer_id", "segment_name", "is_premium"}
    missing_columns = required_columns.difference(segments.columns)
    if missing_columns:
        raise ValueError(
            f"customer_segments.csv missing columns: {sorted(missing_columns)}"
        )

    return segments.set_index("customer_id").to_dict(orient="index")


def customer_is_premium(segment: dict[str, Any]) -> bool:
    """Interpret the premium flag without treating the string 'False' as true."""
    value = segment.get("is_premium", False)

    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes"}


def determine_ticket_route(
    status: str,
    category: str,
    segment: dict[str, Any],
) -> tuple[str, str]:
    """Apply the assignment's routing rules in the required order."""
    human_review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    if status in human_review_statuses:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if customer_is_premium(segment):
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


# 8. Calculate accuracy, token metrics, and confusion table
def calculate_run_metrics(
    internal_results: pd.DataFrame,
    total_tokens: int,
    redaction_totals: dict[str, int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    classified_mask = internal_results["status"].isin(
        {"classified_json", "classified_keyword"}
    )
    classified = internal_results.loc[classified_mask]

    classified_count = len(classified)
    if classified_count:
        correct_count = int(
            (classified["category"] == classified["true_category"]).sum()
        )
        accuracy = correct_count / classified_count
        tokens_per_ticket = total_tokens / classified_count
    else:
        correct_count = 0
        accuracy = 0.0
        tokens_per_ticket = 0.0

    confusion_table = pd.crosstab(
        classified["true_category"],
        classified["category"],
    ).reindex(
        index=sorted(ALLOWED_CATEGORIES),
        columns=sorted(ALLOWED_CATEGORIES),
        fill_value=0,
    )

    metrics = {
        "total_tickets": len(internal_results),
        "classified_tickets": classified_count,
        "correct_classifications": correct_count,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": tokens_per_ticket,
        "budget_exhausted": int(
            (internal_results["status"] == "budget_exhausted").sum()
        ),
        "status_counts": internal_results["status"].value_counts().to_dict(),
        "redaction_counts": redaction_totals,
        "redaction_total": sum(redaction_totals.values()),
    }
    return metrics, confusion_table


# 9. Main ticket-processing pipeline
def run_pipeline(
    prompt: str,
    token_budget: int | None = None,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Process tickets in source-file order and optionally save the CSV."""
    # Reset immediately before every complete run.
    mock_llm.reset()
    random.seed(42)

    if not TICKETS_FILE.exists():
        raise FileNotFoundError(f"Ticket file not found: {TICKETS_FILE}")

    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False,
        dtype={
            "ticket_id": str,
            "customer_id": str,
            "order_id": str,
            "text": str,
            "true_category": str,
        },
    )

    required_columns = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }
    missing_columns = required_columns.difference(tickets.columns)
    if missing_columns:
        raise ValueError(
            f"tickets.csv missing columns: {sorted(missing_columns)}"
        )

    segment_lookup = load_customer_segments()

    total_tokens = 0
    redaction_totals = {name: 0 for name, _, _ in REDACTION_PATTERNS}
    result_rows: list[dict[str, Any]] = []

    for ticket in tickets.itertuples(index=False):
        ticket_id = str(ticket.ticket_id)
        customer_id = str(ticket.customer_id)
        original_text = str(ticket.text)

        category = ""
        tokens_used = 0

        # Count substitutions across all 164 tickets, including empty/long.
        safe_text, substitution_counts = redact_personal_information(
            original_text
        )
        for name, count in substitution_counts.items():
            redaction_totals[name] += count

        # Reapplying all patterns must find no remaining PII.
        assert verify_redaction(safe_text), (
            f"PII remains after redaction for ticket {ticket_id}"
        )

        # Input guardrails run before any model call.
        if is_empty_ticket(original_text):
            status = "skipped_empty"

        elif is_long_ticket(original_text):
            status = "flagged_long"

        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"

        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": safe_text},
            ]

            # The user message is exactly the redacted ticket text.
            assert messages[1]["content"] == safe_text
            assert verify_redaction(messages[1]["content"])

            response = call_with_retry(messages, ticket_id)
            tokens_used = int(response.usage.total_tokens)
            total_tokens += tokens_used

            response_text = response.choices[0].message.content or ""
            category, status = validate_model_response(response_text)

        segment = segment_lookup.get(customer_id, {})
        segment_name = segment.get("segment_name", "")
        if pd.isna(segment_name):
            segment_name = ""

        route, priority = determine_ticket_route(
            status=status,
            category=category,
            segment=segment,
        )

        result_rows.append(
            {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": status,
                "category": category,
                "segment_name": segment_name,
                "priority": priority,
                "route": route,
                "tokens_used": tokens_used,
                # Used for evaluation metrics, excluded from exported CSV.
                "true_category": str(ticket.true_category),
            }
        )

    internal_results = pd.DataFrame(result_rows)
    metrics, confusion_table = calculate_run_metrics(
        internal_results,
        total_tokens,
        redaction_totals,
    )

    exported_results = internal_results[OUTPUT_COLUMNS].copy()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported_results.to_csv(output_path, index=False)

    return exported_results, metrics, confusion_table


# 10. Print reports and execute all required Part C runs
def print_run_report(
    run_name: str,
    metrics: dict[str, Any],
    confusion_table: pd.DataFrame | None = None,
) -> None:
    print(f"\n{'=' * 68}")
    print(run_name)
    print(f"{'=' * 68}")
    print(f"Tickets processed: {metrics['total_tickets']}")
    print(f"Tickets classified: {metrics['classified_tickets']}")
    print(f"Correct classifications: {metrics['correct_classifications']}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.1f}%")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(
        "Tokens per classified ticket: "
        f"{metrics['tokens_per_classified_ticket']:.2f}"
    )
    print(f"Budget-exhausted tickets: {metrics['budget_exhausted']}")
    print(f"Status counts: {metrics['status_counts']}")
    print(f"Redaction counts: {metrics['redaction_counts']}")
    print(f"Total redactions: {metrics['redaction_total']}")

    if confusion_table is not None:
        print("\nConfusion table (rows=true, columns=predicted):")
        print(confusion_table.to_string())


def main() -> None:
    """Run comparison, budget demonstration, and final few-shot triage."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # A. Zero-shot comparison run, without a token budget.
    _, zero_metrics, zero_confusion = run_pipeline(
        prompt=ZERO_SHOT_PROMPT,
        token_budget=None,
    )
    print_run_report(
        "ZERO-SHOT COMPARISON (NO TOKEN BUDGET)",
        zero_metrics,
        zero_confusion,
    )

    # B. Few-shot comparison run, without a token budget.
    _, few_metrics, _ = run_pipeline(
        prompt=FEW_SHOT_PROMPT,
        token_budget=None,
    )
    print_run_report(
        "FEW-SHOT COMPARISON (NO TOKEN BUDGET)",
        few_metrics,
    )

    # C. Zero-shot budget demo. Retry with 5,000 only if 6,000 exhausts none.
    budget_used = BUDGET_DEMO_TOKEN_BUDGET
    budget_results, budget_metrics, _ = run_pipeline(
        prompt=ZERO_SHOT_PROMPT,
        token_budget=budget_used,
        output_path=OUTPUT_DIR / "triage_budget_demo.csv",
    )

    if budget_metrics["budget_exhausted"] == 0:
        budget_used = 5_000
        budget_results, budget_metrics, _ = run_pipeline(
            prompt=ZERO_SHOT_PROMPT,
            token_budget=budget_used,
            output_path=OUTPUT_DIR / "triage_budget_demo.csv",
        )

    print_run_report(
        f"ZERO-SHOT BUDGET DEMO (TOKEN_BUDGET={budget_used})",
        budget_metrics,
    )

    if budget_metrics["budget_exhausted"] == 0:
        raise RuntimeError(
            "No ticket was budget-exhausted at either 6,000 or 5,000 tokens."
        )

    # D. Final few-shot results with the required 100,000-token budget.
    final_results, final_metrics, _ = run_pipeline(
        prompt=FEW_SHOT_PROMPT,
        token_budget=TOKEN_BUDGET,
        output_path=OUTPUT_DIR / "triage_results.csv",
    )
    print_run_report(
        "FINAL FEW-SHOT TRIAGE (TOKEN_BUDGET=100000)",
        final_metrics,
    )

    # Confirm both exported CSVs have the required shape and columns.
    assert len(budget_results) == 164
    assert len(final_results) == 164
    assert list(budget_results.columns) == OUTPUT_COLUMNS
    assert list(final_results.columns) == OUTPUT_COLUMNS

    print("\nPart C output files saved:")
    print(f"  {OUTPUT_DIR / 'triage_budget_demo.csv'}")
    print(f"  {OUTPUT_DIR / 'triage_results.csv'}")
    print(f"Budget demo limit used: {budget_used}")
    print(f"Budget-exhausted tickets: {budget_metrics['budget_exhausted']}")


if __name__ == "__main__":
    main()
