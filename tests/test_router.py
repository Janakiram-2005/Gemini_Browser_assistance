import os
import sys
import unittest
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    app,
    evaluate_tier1_dom,
    route_action,
    TIER1_TIMEOUT,
    TIER2_TIMEOUT,
    TIER3_TIMEOUT,
    CONFIDENCE_THRESHOLD,
    MAX_ESCALATION_ATTEMPTS,
    safe_float
)

class TestTieredActionRouter(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_timeout_configurations(self):
        """Verify dynamic 10s and 30s tier timeouts replace static 180s limits."""
        self.assertEqual(TIER1_TIMEOUT, 10)
        self.assertEqual(TIER2_TIMEOUT, 10)
        self.assertEqual(TIER3_TIMEOUT, 30)
        self.assertEqual(CONFIDENCE_THRESHOLD, 0.60)
        self.assertEqual(MAX_ESCALATION_ATTEMPTS, 2)

    def test_tier1_routine_navigation_latency_and_confidence(self):
        """Verify standard navigation steps resolve in Tier 1 in under 600ms (<20ms) with confidence >= 0.60."""
        start = time.time()
        res = evaluate_tier1_dom(
            command="open youtube.com",
            context="<p>Welcome</p>",
            page_url="",
            page_links=[],
            last_action="",
            action_history=[],
            subtasks=[]
        )
        elapsed_ms = (time.time() - start) * 1000

        self.assertLess(elapsed_ms, 600)  # Sub-600ms criterion
        self.assertLess(elapsed_ms, 20)   # 20ms router overhead limit
        self.assertEqual(res["action"], "open_new_tab")
        self.assertEqual(res["value"], "https://www.youtube.com")
        self.assertGreaterEqual(res["confidence"], 0.60)

    def test_tier1_routine_click_step(self):
        """Verify routine button click with exact DOM link match resolves in Tier 1 with high confidence."""
        page_links = [
            {"text": "Search", "sel": "button#search-icon-legacy"},
            {"text": "Home", "sel": "a#home-link"}
        ]
        start = time.time()
        res = evaluate_tier1_dom(
            command="click Search button",
            context="",
            page_url="https://www.youtube.com",
            page_links=page_links,
            last_action="type",
            action_history=[],
            subtasks=[]
        )
        elapsed_ms = (time.time() - start) * 1000

        self.assertLess(elapsed_ms, 20)
        self.assertEqual(res["action"], "click")
        self.assertEqual(res["selector"], "button#search-icon-legacy")
        self.assertGreaterEqual(res["confidence"], 0.60)

    def test_tier1_routine_type_step(self):
        """Verify routine text input typing resolves in Tier 1 with high confidence."""
        page_links = [
            {"text": "Search input", "sel": "input#search"}
        ]
        res = evaluate_tier1_dom(
            command="type Believer into search",
            context="",
            page_url="https://www.youtube.com",
            page_links=page_links,
            last_action="open_new_tab",
            action_history=[],
            subtasks=[]
        )

        self.assertEqual(res["action"], "type")
        self.assertEqual(res["value"], "Believer")
        self.assertGreaterEqual(res["confidence"], 0.60)

    def test_escalation_for_low_confidence_visual_edge_cases(self):
        """Verify visual edge cases (unlabelled icon buttons/canvas) return confidence < 0.60 and trigger escalation."""
        page_links = [
            {"text": "", "sel": "button.unlabelled-camera-icon"}
        ]
        res = evaluate_tier1_dom(
            command="click unlabelled camera icon button",
            context="",
            page_url="https://example.com",
            page_links=page_links,
            last_action="",
            action_history=[],
            subtasks=[]
        )

        self.assertLess(res["confidence"], 0.60)  # Must trigger escalation

    def test_context_and_subtasks_preserved_across_escalations(self):
        """Verify action history and subtasks roadmap state remain intact across tier routing."""
        history = [
            {"step": 1, "action": "open_new_tab", "value": "https://www.youtube.com", "caption": "Opened YouTube"}
        ]
        subtasks = [
            {"title": "Open YouTube", "status": "completed"},
            {"title": "Search for song", "status": "active"}
        ]
        payload = {
            "command": "click unlabelled camera icon button",
            "context": "Visual page canvas",
            "page_url": "https://www.youtube.com",
            "page_links": [],
            "last_action": "open_new_tab",
            "step_number": 2,
            "action_history": history,
            "subtasks": subtasks,
            "model": "gemma2"
        }

        routed = route_action(payload)
        self.assertIn("tier", routed)
        self.assertIn("router_latency_ms", routed)
        # Verify subtasks structure is preserved
        self.assertEqual(len(routed.get("subtasks", [])), 2)
        self.assertEqual(routed["subtasks"][0]["title"], "Open YouTube")

    def test_agent_endpoint_returns_tier_metadata(self):
        """Verify /agent API endpoint returns step metadata with tier, escalated flag, and latency."""
        response = self.app.post("/agent", json={
            "command": "open google.com",
            "context": "",
            "page_url": "",
            "page_links": [],
            "step_number": 1,
            "model": "gemma2"
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("tier", data)
        self.assertEqual(data["tier"], "tier1")
        self.assertEqual(data["escalated"], False)
        self.assertIn("router_latency_ms", data)
        self.assertLess(data["router_latency_ms"], 600)


if __name__ == "__main__":
    unittest.main()
