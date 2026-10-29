"""
Tests de l'API web (faux providers, aucun réseau). Nécessite les extras de développement : pip install -e ".[dev]".
Lancer :  python -m unittest discover -s tests -v
"""
import unittest

try:
    from fastapi.testclient import TestClient
except Exception:           # httpx absent : ces tests sont ignorés
    TestClient = None

import openai  # noqa: E402
from test_router import RouterTestCase, err_429, err_status, fake, reply  # noqa: E402

if TestClient is not None:
    from ricochet_llm.server import create_app


@unittest.skipIf(TestClient is None, "httpx requis pour fastapi.testclient")
class ServerTests(RouterTestCase):
    def client(self, *providers, disabled=()):
        self.install(*providers, disabled=disabled)
        return TestClient(create_app())

    def test_index_page_and_static_assets_are_served(self):
        c = self.client(fake("a", [reply()]))
        page = c.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Ricochet", page.text)
        self.assertEqual(page.headers["cache-control"], "no-store")
        for asset in ("/static/app.css", "/static/app.js"):
            self.assertEqual(c.get(asset).status_code, 200, asset)

    def test_status_exposes_providers_next_events_and_order(self):
        c = self.client(fake("a", [reply()]), fake("b", [reply()]),
                        disabled=[{"name": "z", "label": "Z", "env_key": "Z_KEY", "supports_tools": True}])
        body = c.get("/api/status").json()
        self.assertEqual(body["order"], ["a", "b", "z"])
        self.assertEqual(body["providers"]["z"]["status"], "disabled")
        self.assertEqual(body["next"]["provider"], "a")
        self.assertEqual(body["events"], [])

    def test_chat_returns_answer_and_routing_meta(self):
        c = self.client(fake("a", [err_429()]), fake("b", [reply("bonjour")]))
        res = c.post("/api/chat", json={"session_id": "s1", "message": "salut"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual((body["status"], body["response"]), ("ok", "bonjour"))
        self.assertEqual(body["meta"]["provider"], "b")
        self.assertEqual(body["meta"]["trail"][0]["provider"], "a")

    def test_chat_keeps_history_per_session_and_reset_clears_it(self):
        b = fake("b", [reply("r1"), reply("r2"), reply("r3")])
        c = self.client(b)
        c.post("/api/chat", json={"session_id": "s1", "message": "un"})
        c.post("/api/chat", json={"session_id": "s1", "message": "deux"})
        self.assertEqual(sum(m["role"] == "user" for m in b["calls"][1]["messages"]), 2)
        c.post("/api/reset", json={"session_id": "s1"})
        c.post("/api/chat", json={"session_id": "s1", "message": "trois"})
        self.assertEqual(sum(m["role"] == "user" for m in b["calls"][2]["messages"]), 1)

    def test_chat_returns_503_with_reason_when_every_provider_is_down(self):
        c = self.client(fake("a", [err_429()]))
        res = c.post("/api/chat", json={"session_id": "s1", "message": "salut"})
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["status"], "error")
        self.assertIn("Aucun provider disponible", res.json()["response"])

    def test_chat_validates_input(self):
        c = self.client(fake("a", [reply()]))
        self.assertEqual(c.post("/api/chat", json={"session_id": "s1", "message": ""}).status_code, 422)
        self.assertEqual(c.post("/api/chat", json={"session_id": "", "message": "x"}).status_code, 422)
        self.assertEqual(c.post("/api/chat", json={"session_id": "s1", "message": "x" * 9000}).status_code, 422)

    def test_ping_endpoint(self):
        c = self.client(fake("a", [reply()]), fake("b", [err_status(openai.AuthenticationError, 401)]))
        self.assertTrue(c.post("/api/providers/a/ping").json()["ok"])
        bad = c.post("/api/providers/b/ping").json()
        self.assertEqual((bad["ok"], bad["kind"]), (False, "config"))

    def test_unknown_provider_returns_404(self):
        c = self.client(fake("a", [reply()]))
        self.assertEqual(c.post("/api/providers/nope/ping").status_code, 404)
        self.assertEqual(c.post("/api/providers/nope/reset").status_code, 404)

    def test_provider_reset_endpoint_lifts_cooldown(self):
        c = self.client(fake("a", [err_429()]), fake("b", [reply()]))
        c.post("/api/chat", json={"session_id": "s1", "message": "x"})
        self.assertEqual(c.get("/api/status").json()["providers"]["a"]["status"], "cooldown")
        self.assertEqual(c.post("/api/providers/a/reset").json()["status"], "ok")
        self.assertEqual(c.get("/api/status").json()["providers"]["a"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
