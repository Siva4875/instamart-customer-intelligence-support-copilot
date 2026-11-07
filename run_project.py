"""Run the Instamart Customer Intelligence & Support Copilot project.

Run from the repository root:
    python run_project.py

The project uses synthetic data and the local mock_llm.py implementation.
No external API key or network call is required.
"""

from pathlib import Path
import json
import math
import uuid

import pandas as pd
import mock_llm

from src.data_pipeline import audit, clean_data, validate
from src.sql_analysis import run_sql_analysis
from src.eda_analysis import run_eda_analysis
from src.customer_models import rfm_and_segments, build_lapse_model
from src.triage import main as run_triage_experiments
from src.policy_rag import PolicyRAG, evaluate
from src.refund_agent import run_agent, resume_agent, TOOL_MAP, TOOLS_SPEC


ROOT = Path(__file__).resolve().parent
SCENARIO_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"


def run_part_a():
    """Audit, clean, validate, and persist the source data."""
    print("\n=== PART A: DATA AUDIT, CLEANING, AND VALIDATION ===")
    audit()
    clean_data()
    validate()
    print("\n=== PART A6: SQL QUERY EXECUTION ===")
    run_sql_analysis()
    print("\n=== PART A7: PANDAS EXPLORATORY ANALYSIS ===")
    run_eda_analysis()


def run_part_b():
    """Build customer segments and the lapse prediction model."""
    print("\n=== PART B: CUSTOMER SEGMENTS AND LAPSE MODEL ===")
    rfm, diagnostics, profile, pca_variance = rfm_and_segments(k=4)

    print("K-means diagnostics:")
    print(diagnostics.to_string(index=False))

    print("\nCustomer segment profile:")
    print(profile.to_string(index=False))

    print("\nPCA explained variance ratio:", pca_variance)
    print("Lapse model summary:", build_lapse_model())


def run_part_c():
    """Run all required Part C comparisons, budget demo, and final triage."""
    print("\n=== PART C: SUPPORT TICKET TRIAGE ===")
    run_triage_experiments()


def run_part_d_policy_rag():
    """Evaluate policy retrieval with the required overlap setting."""
    print("\n=== PART D: POLICY RAG ===")
    rag = PolicyRAG(chunk_size=40, overlap=10)
    evaluation = evaluate(rag)

    print("Policy chunk count:", len(rag.chunks))
    print(evaluation.to_string(index=False))
    print("Expected-document hits:", int(evaluation.retrieved_doc.sum()))
    print("Expected-fact hits:", int(evaluation.fact_found.sum()))
    print("Fallbacks:", int(evaluation.fallback.sum()))

    # Required comparison: repeat the same evaluation without overlap.
    no_overlap_rag = PolicyRAG(chunk_size=40, overlap=0)
    no_overlap_evaluation = evaluate(no_overlap_rag)
    print("\nPolicy RAG comparison (overlap=0):")
    print(no_overlap_evaluation.to_string(index=False))
    print(
        "Answerable questions failing expected-fact check:",
        int((
            (no_overlap_evaluation["fact_found"] == False)
            & (no_overlap_evaluation["fallback"] == False)
        ).sum())
    )

    no_overlap_fact_failures = int((
        (no_overlap_evaluation["fact_found"] == False)
        & (no_overlap_evaluation["fallback"] == False)
    ).sum())
    assert len(rag.chunks) == 15, f"Expected 15 policy chunks; got {len(rag.chunks)}."
    assert int(evaluation.retrieved_doc.sum()) == 12
    assert int(evaluation.fact_found.sum()) == 12
    assert int(evaluation.fallback.sum()) == 2
    assert no_overlap_fact_failures == 2
    print("Policy RAG acceptance checks passed.")


def _run_one_scenario(row):
    """Run one scenario; simulate and recover the S1 checkpoint crash."""
    scenario_id = str(row["scenario_id"])
    message = str(row["message"])

    # Keep each scenario's deterministic mock-model counter independent.
    mock_llm.reset()

    if scenario_id == "S1":
        run_id = "S1-checkpoint-resume-" + uuid.uuid4().hex[:8]
        try:
            run_agent(
                message,
                scenario_id=scenario_id,
                run_id=run_id,
                max_iterations=6,
                crash_after=2
            )
        except RuntimeError as error:
            if "Simulated crash after checkpoint" not in str(error):
                raise
            print("S1: simulated crash after tool 2; resuming checkpoint.")

        # Three further model turns are sufficient after the first two tools:
        # calculate_refund, issue_refund, and the final assistant response.
        return resume_agent(
            run_id,
            max_iterations=4,
            token_limit=6000,
            time_limit_seconds=10
        )

    return run_agent(
        message,
        scenario_id=scenario_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10
    )


def run_part_d_agent():
    """Execute S1-S7 and the three required safety demonstrations."""
    print("\n=== PART D: REFUND AGENT SCENARIOS ===")

    registered_tools = set(TOOL_MAP)
    specified_tools = {tool["name"] for tool in TOOLS_SPEC}
    if len(registered_tools) != 5 or registered_tools != specified_tools:
        raise AssertionError(
            "The five tool implementations and tool specifications must match."
        )

    if not SCENARIO_FILE.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIO_FILE}. "
            "Run python generate_data.py first."
        )

    scenarios = pd.read_csv(SCENARIO_FILE, keep_default_na=False)
    required_columns = {
        "scenario_id",
        "customer_id",
        "message",
        "expected_outcome",
        "expected_refund_inr"
    }
    missing = required_columns.difference(scenarios.columns)
    if missing:
        raise ValueError(
            "agent_scenarios.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    if scenarios["scenario_id"].tolist() != [f"S{i}" for i in range(1, 8)]:
        raise ValueError("Expected exactly the ordered scenarios S1 through S7.")

    run_results = []
    for _, row in scenarios.iterrows():
        result = _run_one_scenario(row)
        expected_status = str(row["expected_outcome"])

        print(
            f"{row['scenario_id']}: status={result['status']}, "
            f"model_calls={result['llm_calls']}, "
            f"tokens={result['tokens_used']}"
        )

        actual_status = result["status"]

        # Normalize detailed fallback statuses for rubric comparison.
        actual_outcome = (
            "FALLBACK"
            if actual_status.startswith("FALLBACK")
            else actual_status
        )

        run_results.append({
            "scenario_id": row["scenario_id"],
            "customer_id": row["customer_id"],
            "expected_outcome": expected_status,
            "actual_outcome": actual_outcome,
            "actual_status": actual_status,
            "outcome_match": actual_outcome == expected_status,
            "expected_refund_inr": float(row["expected_refund_inr"]),
            "actual_refund_inr": _refund_amount(result),
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "tools_executed": result.get("tools_executed", [])
        })

    for item in run_results:
        if not item["outcome_match"]:
            raise AssertionError(
                f"{item['scenario_id']} expected {item['expected_outcome']} "
                f"but received {item['actual_outcome']}."
            )
        if not math.isclose(
            item["actual_refund_inr"],
            item["expected_refund_inr"],
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            raise AssertionError(
                f"{item['scenario_id']} refund mismatch: expected "
                f"{item['expected_refund_inr']:.2f}, got "
                f"{item['actual_refund_inr']:.2f}."
            )

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "agent_scenario_summary.json"
    summary_path.write_text(
        json.dumps(run_results, indent=2),
        encoding="utf-8"
    )
    print("\nScenario summary:")
    print(pd.DataFrame(run_results).to_string(index=False))

    # Token-limit and timeout fallbacks are isolated demonstrations.
    first_message = str(scenarios.iloc[0]["message"])

    mock_llm.reset()
    token_demo = run_agent(
        first_message,
        scenario_id="token-limit-demo",
        token_limit=200
    )
    print("\nToken-limit demonstration:", token_demo["status"])
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"

    mock_llm.reset()
    timeout_demo = run_agent(
        first_message,
        scenario_id="timeout-demo",
        time_limit_seconds=0
    )
    print("Timeout demonstration:", timeout_demo["status"])
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"
    print("Refund-agent acceptance checks passed for 7 scenarios and 3 safety limits.")


def _refund_amount(result):
    """Extract the amount recorded by issue_refund or escalation tools."""
    for step in result.get("steps", []):
        observation = step.get("observation", {})
        if step.get("tool") in {"issue_refund", "escalate_to_human"}:
            return float(observation.get("refund_inr", 0.0))
    return 0.0


def main():
    run_part_a()
    run_part_b()
    run_part_c()
    run_part_d_policy_rag()
    run_part_d_agent()
    print("\nProject pipeline finished. Review data/, outputs/, and state/.")


if __name__ == "__main__":
    main()
