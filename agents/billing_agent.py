"""Billing Agent — a graph node reached once routing assigns
`assigned_team = "payments_team"` (see pipeline/graph.py).

No LLM call here on purpose: by this point `compacted_info` already has
everything needed, and the "agent" is really just deterministic tool
orchestration + a templated reply. The diagram in the spec only marks the
Info-Gathering and Compaction nodes as LLM calls — specialist agents just
act on the structured facts they're handed.
"""
from concurrent.futures import ThreadPoolExecutor

from pipeline.state import SessionState, now_iso
from tools.mock_tools import check_payment_status, lookup_invoice


def run(state: SessionState) -> SessionState:
    info = state["compacted_info"]
    order_id = info.order_id or "UNKNOWN"
    print("[BILLING_AGENT] compacted_info:", info)
    print("[BILLING_AGENT] order_id:", order_id)

    print("[BILLING_AGENT] starting invoice and payment lookups in parallel")
    with ThreadPoolExecutor(max_workers=2) as executor:
        invoice_future = executor.submit(lookup_invoice, order_id)
        payment_future = executor.submit(check_payment_status, order_id)
        invoice = invoice_future.result()
        payment = payment_future.result()
    print("[BILLING_AGENT] invoice lookup result:", invoice)
    print("[BILLING_AGENT] payment status result:", payment)

    state["tool_calls"].append({"tool": "lookup_invoice", "args": {"order_id": order_id}, "result": invoice})
    state["tool_calls"].append({"tool": "check_payment_status", "args": {"order_id": order_id}, "result": payment})
    state["tool_result"] = {"invoice": invoice, "payment_status": payment}

    if not invoice["found"]:
        reply = f"I couldn't find any order matching {order_id} in our records — could you double-check the order ID?"
    else:
        reply = (
            f"I looked into order {order_id}. {invoice['description']} "
            f"Payment status: {payment['status']}. Let me know if you'd like more detail."
        )
    print("[BILLING_AGENT] final reply:", reply)
    state["conversation_history"].append({"role": "assistant", "content": reply, "timestamp": now_iso()})
    return state
