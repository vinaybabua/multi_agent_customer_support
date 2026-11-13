import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import billing_agent


class BillingAgentTests(unittest.TestCase):
    def test_independent_lookups_run_concurrently_and_results_are_recorded(self):
        both_lookups_started = threading.Barrier(2)

        def lookup_invoice(order_id):
            both_lookups_started.wait(timeout=2)
            return {"order_id": order_id, "found": True, "description": "Invoice found"}

        def check_payment_status(order_id):
            both_lookups_started.wait(timeout=2)
            return {"order_id": order_id, "found": True, "status": "paid"}

        state = {
            "compacted_info": SimpleNamespace(order_id="ORD-1003"),
            "tool_calls": [],
            "tool_result": None,
            "conversation_history": [],
        }

        with (
            patch.object(billing_agent, "lookup_invoice", side_effect=lookup_invoice),
            patch.object(billing_agent, "check_payment_status", side_effect=check_payment_status),
        ):
            result = billing_agent.run(state)

        self.assertEqual(result["tool_result"]["invoice"]["description"], "Invoice found")
        self.assertEqual(result["tool_result"]["payment_status"]["status"], "paid")
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertIn("Payment status: paid", result["conversation_history"][-1]["content"])


if __name__ == "__main__":
    unittest.main()