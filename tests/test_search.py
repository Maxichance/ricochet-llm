"""
Recherche Web par API gratuite. Aucun appel réseau : les trois API sont simulées avec les schémas de leur documentation
officielle. Réponses SYNTHÉTIQUES.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet import search as sr  # noqa: E402
from ricochet.tools import registry  # noqa: E402

EXA = {"requestId": "r", "results": [
    {"title": "Guide des routeurs", "url": "https://exemple.org/guide",
     "highlights": ["Le routeur bascule seul.", "Le quota est suivi."], "publishedDate": "2026-10-01"},
    {"title": "Doublon", "url": "https://exemple.org/guide", "highlights": []},
    {"title": "Sans adresse valide", "url": "javascript:alert(1)"}], "costDollars": {"total": 0.007}}
TAVILY = {"query": "q", "results": [
    {"title": "Guide", "url": "https://exemple.net/g", "content": "Extrait Tavily", "score": 0.9,
     "raw_content": "Texte integral " * 5},
    {"title": "Autre", "url": "https://exemple.net/a", "content": "Autre extrait", "score": 0.5}],
    "usage": {"credits": 1}, "request_id": "x"}
LANGSEARCH = {"code": 200, "log_id": "l", "msg": None, "data": {"_type": "SearchResponse", "webPages": {"value": [
    {"id": "1", "name": "Page LS", "url": "https://exemple.com/p", "displayUrl": "x", "snippet": "Resume LS",
     "datePublished": None, "text": "Texte LS"}]}}, "usage": {"input_tokens": 12, "output_tokens": 30}}

HOSTS = {"api.exa.ai": "exa", "api.tavily.com": "tavily", "api.langsearch.com": "langsearch"}
ENV_VARS = ("EXA_API_KEY", "TAVILY_API_KEY", "LANGSEARCH_API_KEY", sr.ORDER_ENV_VAR, sr.STATE_ENV_VAR,
            "RICOCHET_EXA_TYPE", "RICOCHET_SEARCH_COUNTRY")


class SearchTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = {k: v for k, v in os.environ.items() if k not in ENV_VARS}
        env[sr.STATE_ENV_VAR] = str(Path(tmp.name) / "usage.json")
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        sr.reset()
        self.addCleanup(sr.reset)

    def key(self, *names):
        for name in names:
            os.environ[sr.PROVIDERS[name]["env_key"]] = f"cle-{name}"

    def run_search(self, routes, calls=None, **kw):
        """`routes` : nom du provider -> (statut, corps[, en-têtes])."""
        def handler(req):
            name = HOSTS[req.url.host]
            if calls is not None:
                calls.append((name, dict(req.headers), json.loads(req.content)))
            status, body, *headers = routes[name]
            return httpx.Response(status, json=body, headers=headers[0] if headers else {})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return sr.search("routeur gratuit", client=client, **kw)


class AdapterTests(SearchTestCase):
    def test_exa_request_and_reading(self):
        self.key("exa")
        calls = []
        r = self.run_search({"exa": (200, EXA)}, calls, n=5, content=True)
        _, headers, body = calls[0]
        self.assertEqual(headers["x-api-key"], "cle-exa")
        self.assertEqual(body, {"query": "routeur gratuit", "numResults": 5, "type": "auto",
                                "contents": {"highlights": True}})
        self.assertEqual((r.status, r.engine, r.cost), (sr.SUCCESS, "exa", 0.007))
        self.assertEqual([p["url"] for p in r.results], ["https://exemple.org/guide"])     # doublon et adresse invalide écartés
        self.assertEqual(r.results[0]["content"], "Le routeur bascule seul. ... Le quota est suivi.")

    def test_tavily_request_and_reading(self):
        self.key("tavily")
        calls = []
        r = self.run_search({"tavily": (200, TAVILY)}, calls, content=True)
        _, headers, body = calls[0]
        self.assertEqual(headers["authorization"], "Bearer cle-tavily")
        self.assertEqual((body["search_depth"], body["include_raw_content"], body["include_usage"]), ("basic", "text", True))
        self.assertNotIn("country", body)                       # le pays n'est envoyé que s'il est demandé
        self.assertEqual((r.engine, r.cost, r.results[0]["snippet"]), ("tavily", 1.0, "Extrait Tavily"))
        self.assertTrue(r.results[0]["content"].startswith("Texte integral"))
        self.assertNotIn("content", r.results[1])               # Tavily ne rend le texte que pour une partie des résultats

    def test_tavily_country_is_configurable(self):
        self.key("tavily")
        os.environ["RICOCHET_SEARCH_COUNTRY"] = "France"
        calls = []
        self.run_search({"tavily": (200, TAVILY)}, calls)
        self.assertEqual(calls[0][2]["country"], "france")

    def test_langsearch_request_and_reading(self):
        self.key("langsearch")
        calls = []
        r = self.run_search({"langsearch": (200, LANGSEARCH)}, calls, content=True)
        _, headers, body = calls[0]
        self.assertEqual(headers["authorization"], "Bearer cle-langsearch")
        self.assertEqual((body["contents"], body["count"]), ({"text": {"maxCharacters": sr.CONTENT_MAX}}, 6))
        self.assertEqual((r.engine, r.cost, r.results[0]["title"], r.results[0]["content"]),
                         ("langsearch", 42.0, "Page LS", "Texte LS"))

    def test_langsearch_sends_no_summary_when_the_text_is_requested(self):
        """Avec `contents.text`, l'API n'envoie plus de `snippet` ; l'extrait vient du texte."""
        self.key("langsearch")
        real = {"code": 200, "log_id": "l", "msg": None, "data": {"webPages": {"value": [
            {"id": "1", "name": "Page", "url": "https://exemple.org/p", "displayUrl": "x", "datePublished": None,
             "text": "Texte integral de la page. " * 30}]}}, "usage": {"input_tokens": 9, "output_tokens": 961}}
        r = self.run_search({"langsearch": (200, real)}, content=True)
        self.assertTrue(r.results[0]["snippet"].startswith("Texte integral de la page."))
        self.assertEqual((len(r.results[0]["snippet"]), r.cost), (sr.SNIPPET_MAX, 970.0))

    def test_without_content_nothing_more_is_requested(self):
        self.key("tavily")
        calls = []
        self.run_search({"tavily": (200, TAVILY)}, calls)
        self.assertNotIn("include_raw_content", calls[0][2])

    def test_exa_search_type_is_configurable(self):
        self.key("exa")
        os.environ["RICOCHET_EXA_TYPE"] = "lite"
        calls = []
        self.run_search({"exa": (200, EXA)}, calls)
        self.assertEqual(calls[0][2]["type"], "lite")


class FailoverTests(SearchTestCase):
    def test_the_next_provider_answers_when_the_first_refuses_and_the_first_is_not_retried(self):
        self.key("exa", "tavily")
        r = self.run_search({"exa": (429, {}, {"retry-after": "120"}), "tavily": (200, TAVILY)})
        self.assertEqual(r.engine, "tavily")
        self.assertTrue(next(p for p in sr.status()["providers"] if p["name"] == "exa")["paused"])
        calls = []
        self.run_search({"exa": (200, EXA), "tavily": (200, TAVILY)}, calls)
        self.assertEqual([c[0] for c in calls], ["tavily"])

    def test_exhausted_quota_402_is_remembered_on_disk_until_renewal(self):
        self.key("exa", "tavily")
        self.run_search({"exa": (402, {}), "tavily": (200, TAVILY)})
        self.assertGreaterEqual(sr.used("exa"), sr.PROVIDERS["exa"]["quota"])              # poussé au plafond
        sr._cooldown.clear()                                                                # nouveau processus : plus de pause en mémoire
        calls = []
        self.run_search({"exa": (200, EXA), "tavily": (200, TAVILY)}, calls)
        self.assertEqual([c[0] for c in calls], ["tavily"])                                 # Exa reste écarté grâce au compteur

    def test_tavily_432_is_an_exhausted_quota(self):
        self.key("tavily")
        r = self.run_search({"tavily": (432, {"detail": "limit"})})
        self.assertEqual(r.status, sr.RATE_LIMITED)
        self.assertGreaterEqual(sr.used("tavily"), sr.PROVIDERS["tavily"]["quota"])

    def test_it_stops_before_the_providers_own_ceiling(self):
        self.key("tavily")
        for _ in range(3):
            sr._count("tavily", 300.0)                          # 900 crédits : 90 % du quota gratuit
        self.assertNotIn("tavily", sr.available())
        r = self.run_search({"tavily": (200, TAVILY)})
        self.assertEqual(r.status, sr.ERROR)
        self.assertIn("aucun moteur", r.error)

    def test_the_counter_adds_the_real_cost_and_forgets_past_periods(self):
        self.key("tavily")
        self.run_search({"tavily": (200, TAVILY)})
        self.run_search({"tavily": (200, TAVILY)})
        self.assertEqual(sr.used("tavily"), 2.0)
        state = json.loads(sr.state_file().read_text(encoding="utf-8"))
        self.assertEqual(list(state["tavily"]), [sr._period_key(sr.PROVIDERS["tavily"])])
        state["tavily"] = {"2020-01": 999.0}                    # un ancien mois n'est plus compté
        sr.state_file().write_text(json.dumps(state), encoding="utf-8")
        self.assertEqual(sr.used("tavily"), 0.0)

    def test_a_refused_key_pauses_the_provider_for_a_long_time(self):
        self.key("exa")
        r = self.run_search({"exa": (401, {})})
        self.assertEqual(r.status, sr.BLOCKED)
        self.assertGreater(sr._cooldown["exa"] - time.monotonic(), 600)

    def test_an_application_error_code_in_a_200_falls_over(self):
        self.key("langsearch", "tavily")
        os.environ[sr.ORDER_ENV_VAR] = "langsearch,tavily"
        r = self.run_search({"langsearch": (200, {"code": 500, "msg": "boom"}), "tavily": (200, TAVILY)})
        self.assertEqual(r.engine, "tavily")

    def test_order_is_configurable_and_unknown_names_are_ignored(self):
        os.environ[sr.ORDER_ENV_VAR] = "tavily, inconnu ,exa"
        self.assertEqual(sr._order(), ["tavily", "exa"])
        os.environ[sr.ORDER_ENV_VAR] = "n'importe quoi"
        self.assertEqual(sr._order(), list(sr.DEFAULT_ORDER))

    def test_fallback_is_called_when_no_api_is_usable(self):
        def ddg(query, n):
            return sr.SearchResult(sr.SUCCESS, query, [{"title": "t", "url": "https://x.fr", "snippet": ""}], engine="secours")
        self.assertEqual(self.run_search({}, fallback=ddg).engine, "secours")                 # aucune clé configurée
        self.key("exa")
        self.assertEqual(self.run_search({"exa": (429, {})}, fallback=ddg).engine, "secours")  # une API qui refuse

    def test_without_fallback_the_last_refusal_is_returned(self):
        self.key("exa")
        self.assertEqual(self.run_search({"exa": (429, {})}).status, sr.RATE_LIMITED)
        self.assertEqual(self.run_search({}).status, sr.ERROR)

    def test_an_empty_answer_and_an_unreadable_one(self):
        self.key("tavily")
        self.assertEqual(self.run_search({"tavily": (200, {"results": []})}).status, sr.EMPTY)
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text="pas du json"))) as client:
            self.assertEqual(sr.search("q", client=client).status, sr.ERROR)

    def test_a_network_error_is_a_refusal_not_an_exception(self):
        self.key("exa")

        def boom(req):
            raise httpx.ConnectError("injoignable")
        with httpx.Client(transport=httpx.MockTransport(boom)) as client:
            r = sr.search("q", client=client)
        self.assertEqual(r.status, sr.ERROR)
        self.assertNotIn("injoignable", json.dumps(sr.search_web("q")))

    def test_an_empty_query_calls_nothing(self):
        self.key("exa")
        self.assertEqual(sr.search("   ").status, sr.ERROR)


class StatusAndToolTests(SearchTestCase):
    def test_status_needs_no_network_and_never_shows_a_key(self):
        os.environ["EXA_API_KEY"] = "secret-a-ne-jamais-montrer"
        status = sr.status()
        self.assertNotIn("secret-a-ne-jamais-montrer", json.dumps(status))
        exa = next(p for p in status["providers"] if p["name"] == "exa")
        self.assertTrue(exa["configured"] and exa["available"])
        self.assertEqual(exa["quota"], 10.0)
        self.assertFalse(next(p for p in status["providers"] if p["name"] == "tavily")["configured"])

    def test_the_tool_hides_every_technical_detail_from_the_model(self):
        def blocked(query, n):
            return sr.SearchResult(sr.BLOCKED, query, error="défi anti-robot du moteur")
        with mock.patch.object(sr, "search", side_effect=lambda q, **kw: blocked(q, 1)):
            out = sr.search_web("routeur")
        self.assertEqual(out, {"error": "recherche indisponible", "status": sr.BLOCKED})

    def test_the_tool_returns_pages(self):
        pages = [{"title": "t", "url": "https://x.fr", "snippet": "s"}]
        with mock.patch.object(sr, "search", return_value=sr.SearchResult(sr.SUCCESS, "q", pages)):
            self.assertEqual(sr.search_web("q"), {"results": pages})
        with mock.patch.object(sr, "search", return_value=sr.SearchResult(sr.EMPTY, "q")):
            self.assertEqual(sr.search_web("q"), {"results": []})

    def test_the_tool_is_registered_with_a_schema_the_model_can_use(self):
        saved = dict(registry._tools)
        self.addCleanup(lambda: (registry._tools.clear(), registry._tools.update(saved)))
        registry.clear()
        sr.register_search_tool()
        sr.register_search_tool()                               # idempotent
        schema = registry.schemas()[0]["function"]
        self.assertEqual(schema["name"], "search_web")
        self.assertEqual(schema["parameters"]["required"], ["query"])
        self.assertIn("query", schema["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
