"""Part D: database-backed refund tools and bounded ReAct loop."""

from pathlib import Path
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

import mock_llm
from src.policy_rag import PolicyRAG


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"

rag = PolicyRAG()


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            "select order_id,status,order_amount,delivery_minutes "
            "from orders where order_id=?",
            (order_id,)
        ).fetchone()

    if not row:
        return {"error": "order not found"}

    return dict(zip(
        ["order_id", "status", "order_amount", "delivery_minutes"],
        row
    ))


def search_policy(query):
    return {"results": rag.retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    amount = (
        float(order_amount)
        if delivery_minutes is not None and float(delivery_minutes) > 45
        else 0.0
    )
    return {"refund_inr": round(amount, 2)}



def _ledger(tool, order_id, refund_inr):
    OUT.mkdir(exist_ok=True)
    path = OUT / "refund_ledger.jsonl"

    # Prevent duplicate transactions for the same tool and order.
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    existing = json.loads(line)

                    if (
                        existing.get("tool") == tool
                        and existing.get("order_id") == order_id
                    ):
                        return {
                            "refund_inr": existing["refund_inr"],
                            "status": "ALREADY_PROCESSED"
                        }

                except json.JSONDecodeError:
                    continue

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "refund_inr": refund_inr,
        "status": "PROCESSED"
    }


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve an order using its order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search refund policies for applicable rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund based on order amount and delivery duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {
                    "type": ["number", "null"]
                }
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a high-value refund to a human support lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]



def _save_checkpoint(run_id, scenario_id, messages, tokens,
                     steps, executed, llm_calls, step_number,
                     completed=False, result=None):
    STATE.mkdir(exist_ok=True)

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens,
        "step_number": step_number,
        "tools_executed": executed,
        "steps": steps,
        "llm_calls": llm_calls,
        "completed": completed,
        "result": result
    }

    path = STATE / f"{run_id}.json"
    path.write_text(
        json.dumps(checkpoint, default=str, indent=2),
        encoding="utf-8"
    )


def _finish(run_id, scenario_id, status, final_answer,
            steps, executed, llm_calls, tokens, messages):
    result = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "tools_executed": executed,
        "llm_calls": llm_calls,
        "tokens_used": tokens
    }

    OUT.mkdir(exist_ok=True)
    log_path = OUT / "agent_runs.jsonl"

    # Avoid writing the same completed run more than once.
    already_logged = False

    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("run_id") == run_id:
                        already_logged = True
                        break
                except json.JSONDecodeError:
                    continue

    if not already_logged:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result
            }) + "\n")

    _save_checkpoint(
        run_id, scenario_id, messages, tokens, steps, executed,
        llm_calls, len(executed), completed=True, result=result
    )

    return result


def _run_loop(run_id, scenario_id, messages, tokens,
              steps, executed, llm_calls, max_iterations,
              token_limit, time_limit_seconds, crash_after=None):

    start = time.monotonic()

    for _ in range(max_iterations):

        if time.monotonic() - start >= time_limit_seconds:
            return _finish(
                run_id, scenario_id,
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: hand off to a human agent.",
                steps, executed, llm_calls, tokens , messages
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens += response.usage.total_tokens

        if tokens > token_limit:
            return _finish(
                run_id, scenario_id,
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: hand off to a human agent.",
                steps, executed, llm_calls, tokens, messages
            )

        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            status = content.split(":", 1)[0]

            return _finish(
                run_id, scenario_id, status, content,
                steps, executed, llm_calls, tokens, messages
            )

        tc = msg.tool_calls[0]
        args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "name": tc.name,
                "arguments": args
            }]
        })

        # Safely handle unknown tool names.
        if tc.name not in TOOL_MAP:
            observation = {
                "error": f"Unknown tool: {tc.name}"     
            }

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(observation)
            })

            steps.append({
                "tool": tc.name,
                "arguments": args,
                "observation": observation
            })

            return _finish(
                run_id,
                scenario_id,
                "FALLBACK_INVALID_TOOL",
                "FALLBACK_INVALID_TOOL: unknown tool requested. "
                "Hand off to a human agent.",
                steps,
                executed,
                llm_calls,
                tokens,
                messages
            )

        # Execute the validated tool name.
        observation = TOOL_MAP[tc.name](**args)
        executed.append(tc.name)

        # Record the tool result in the conversation.
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(observation)
        })

        steps.append({
            "tool": tc.name,
            "arguments": args,
            "observation": observation
        })

        # Stop if the refund was already processed.
        if observation.get("status") == "ALREADY_PROCESSED":
            return _finish(
                run_id,
                scenario_id,
                "ALREADY_PROCESSED",
                "ALREADY_PROCESSED: refund was already processed.",
                steps,
                executed,
                llm_calls,
                tokens,
                messages
            )

    # Save only after the tool result has been added to the conversation.
    _save_checkpoint(
        run_id, scenario_id, messages, tokens,
        steps, executed, llm_calls, len(executed)
    )

    if crash_after is not None and len(executed) == crash_after:
        raise RuntimeError("Simulated crash after checkpoint")

    return _finish(
        run_id, scenario_id,
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: hand off to a human agent.",
        steps, executed, llm_calls, tokens, messages
    )


def run_agent(user_message, scenario_id="demo", max_iterations=6,
              token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after=None):

    run_id = run_id or str(uuid.uuid4())
    STATE.mkdir(exist_ok=True)

    messages = [
        {
            "role": "system",
            "content": "AGENT: Use tools to verify orders and follow refund policy."
        },
        {"role": "user", "content": user_message}
    ]

    return _run_loop(
        run_id, scenario_id, messages,
        tokens=0,
        steps=[],
        executed=[],
        llm_calls=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after=crash_after
    )


def resume_agent(run_id, max_iterations=6,
                 token_limit=6000, time_limit_seconds=10):

    checkpoint_path = STATE / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run_id: {run_id}"
        )

    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )

    if checkpoint.get("completed"):
        raise ValueError(
            f"Run {run_id} is already completed and cannot be resumed."
        )

    return _run_loop(
        run_id=run_id,
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        tokens=checkpoint["tokens_used"],
        steps=checkpoint["steps"],
        executed=checkpoint["tools_executed"],
        llm_calls=checkpoint["llm_calls"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )