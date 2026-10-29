"""
Appel isolé `generate()` et état PAR MODÈLE : pourquoi un refus n'écarte qu'un modèle.

  * un 429 ou un 413 n'écarte qu'UN modèle ; le provider n'est en pause que si plus aucun de ses modèles ne répond
  * une réponse vide (raisonnement qui a mangé max_tokens) fait passer au modèle suivant
  * `model_options` : réglages propres à un modèle (couper le raisonnement des Nemotron)
  * `timeout` allonge le délai du provider pour un appel, ne le raccourcit jamais
Aucun appel réseau, aucun quota consommé.
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai  # noqa: E402
from test_router import RouterTestCase, _make_error, err_429, err_status, fake  # noqa: E402

from ricochet import router as r  # noqa: E402

THINKING_OFF = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def answer(content="ok", finish="stop", reasoning=None, tokens_in=11, tokens_out=7):
    """Réponse de l'API avec usage détaillé et, si demandé, un `reasoning_content` (modèles qui raisonnent)."""
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning_content=reasoning)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish)],
        usage=SimpleNamespace(total_tokens=tokens_in + tokens_out, prompt_tokens=tokens_in, completion_tokens=tokens_out))
    return SimpleNamespace(headers={}, parse=lambda: resp)


def multi(name, script, models=("big", "mid"), options=None, limits=None, timeout=20, **kw):
    """Provider à plusieurs modèles : `{name}-big` (lourd), puis les modèles de repli, `{name}-small` en LIGHT."""
    p = fake(name, script, **kw)
    names = [f"{name}-{m}" for m in models]
    p["model_heavy"], p["extra_models"] = names[0], names[1:]
    p["model_options"] = {f"{name}-{m}": o for m, o in (options or {}).items()}
    p["token_limits"] = {f"{name}-{m}": n for m, n in (limits or {}).items()}
    p["timeout"] = timeout
    p["timeouts_asked"] = []
    client = p["client"]
    client.with_options = lambda **o: (p["timeouts_asked"].append(o.get("timeout")), client)[1]
    return p


def by_model(table, default=None):
    """script : une fonction(kwargs) qui répond selon le modèle demandé."""
    return lambda kw: table.get(kw["model"], default if default is not None else answer())


class GenerateTests(RouterTestCase):
    def test_returns_text_provider_model_and_tokens(self):
        a = multi("a", [answer("bonjour", tokens_in=120, tokens_out=40)])
        self.install(a)
        res = r.generate("dis bonjour")
        self.assertEqual((res.text, res.provider, res.model, res.mode), ("bonjour", "a", "a-big", "HEAVY"))
        self.assertEqual((res.tokens_in, res.tokens_out, res.truncated, res.trail), (120, 40, False, []))

    def test_system_prompt_and_message_list_are_sent(self):
        a = multi("a", [answer()])
        self.install(a)
        r.generate([{"role": "user", "content": "x"}], system="Tu es concis.", max_tokens=123, temperature=0.5)
        sent = a["calls"][0]
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        self.assertEqual((sent["max_tokens"], sent["temperature"]), (123, 0.5))
        self.assertNotIn("tools", sent)                      # appel isolé : jamais d'outils

    # ---------- réglages propres à un modèle ----------

    def test_model_options_extra_body_is_sent_only_to_that_model(self):
        a = multi("a", [by_model({"a-big": err_429(5)})], options={"big": THINKING_OFF})
        self.install(a)
        res = r.generate("x")
        self.assertEqual(res.model, "a-mid")
        first, second = a["calls"]
        self.assertEqual(first["extra_body"], THINKING_OFF["extra_body"])    # le Nemotron qui raisonne : raisonnement coupé
        self.assertNotIn("extra_body", second)                               # le modèle sans réglage n'en reçoit aucun

    def test_chat_loop_and_ping_use_the_same_model_options(self):
        a = multi("a", [answer("réponse")], options={"big": THINKING_OFF, "small": THINKING_OFF})
        self.install(a)
        r.chat([], "bonjour")
        self.assertEqual(a["calls"][-1]["extra_body"], THINKING_OFF["extra_body"])
        r.ping_provider("a")
        self.assertEqual(a["calls"][-1]["model"], "a-small")
        self.assertEqual(a["calls"][-1]["extra_body"], THINKING_OFF["extra_body"])

    # ---------- un 429 écarte un MODÈLE, pas le provider ----------

    def test_429_on_one_model_falls_to_the_next_model_of_the_same_provider(self):
        a = multi("a", [by_model({"a-big": err_429(30)})])
        b = multi("b", [answer("depuis b")])
        self.install(a, b)
        res = r.generate("x")
        self.assertEqual((res.provider, res.model), ("a", "a-mid"))                # le même provider, le modèle suivant
        self.assertEqual(res.trail[0]["outcome"], "model_limited")
        self.assertEqual(r._provider_cooldown_until["a"], 0.0)                      # le provider N'EST PAS en pause
        self.assertGreater(r._model_wait(a, "a-big"), 0)                            # seul le modèle l'est

    def test_the_paused_model_is_skipped_until_its_pause_ends(self):
        a = multi("a", [by_model({"a-big": err_429(30)})])
        self.install(a)
        r.generate("x")
        a["calls"].clear()
        r.generate("x")                                       # tout de suite : le lourd est encore en pause
        self.assertEqual([c["model"] for c in a["calls"]], ["a-mid"])
        self.clock[0] += 31
        a["calls"].clear()
        r.generate("x")                                       # la pause est échue : le lourd redevient le premier choix
        self.assertEqual(a["calls"][0]["model"], "a-big")

    def test_429_on_every_model_pauses_the_provider_until_the_nearest_return(self):
        a = multi("a", [by_model({"a-big": err_429(40), "a-mid": err_429(10)})])
        b = multi("b", [answer("depuis b")])
        self.install(a, b)
        res = r.generate("x")
        self.assertEqual(res.provider, "b")
        remaining = r._provider_cooldown_until["a"] - self.clock[0]
        self.assertAlmostEqual(remaining, 10, delta=1.5)      # le provider revient avec son premier modèle disponible

    def test_single_model_provider_keeps_the_old_behaviour(self):
        a = fake("a", [err_429(30)])
        b = fake("b", [answer("depuis b")])
        self.install(a, b)
        res = r.generate("x")
        self.assertEqual(res.provider, "b")
        self.assertGreater(r._provider_cooldown_until["a"], self.clock[0])

    # ---------- 413 : limite apprise, un seul modèle écarté ----------

    def test_413_learns_the_limit_and_does_not_bench_the_provider(self):
        too_big = _make_error(openai.APIStatusError, "Error code: 413 - Request too large ... TPM: Limit 8000, Requested 11342", 413)
        a = multi("a", [by_model({"a-big": too_big})])
        self.install(a)
        res = r.generate("x")
        self.assertEqual(res.model, "a-mid")
        self.assertEqual(r._model_limit[("a", "a-big")], 8000)
        self.assertEqual(r._provider_cooldown_until["a"], 0.0)   # un seul 413 n'écarte pas tout le provider

    def test_known_token_limit_skips_a_model_that_would_refuse(self):
        a = multi("a", [answer("ok")], limits={"big": 50})
        self.install(a)
        res = r.generate("x" * 600)                              # ~200 jetons estimés > 50
        self.assertEqual(res.model, "a-mid")
        self.assertEqual([c["model"] for c in a["calls"]], ["a-mid"])    # a-big n'a jamais été appelé

    # ---------- modèle retiré du catalogue ----------

    def test_404_on_one_model_benches_that_model_only(self):
        a = multi("a", [by_model({"a-big": err_status(openai.NotFoundError, 404)})])
        self.install(a)
        res = r.generate("x")
        self.assertEqual(res.model, "a-mid")
        self.assertEqual(r._provider_cooldown_until["a"], 0.0)

    def test_404_on_a_single_model_provider_still_benches_the_provider(self):
        a = fake("a", [err_status(openai.NotFoundError, 404)])
        b = fake("b", [answer("depuis b")])
        self.install(a, b)
        self.assertEqual(r.generate("x").provider, "b")
        self.assertGreater(r._provider_cooldown_until["a"], self.clock[0])

    # ---------- réponse vide ----------

    def test_empty_answer_from_a_reasoning_model_moves_to_the_next_model(self):
        """Le modèle a dépensé tous ses jetons à raisonner : content vide, finish_reason=length."""
        a = multi("a", [by_model({"a-big": answer("", finish="length", reasoning="Here's a thinking process...")})])
        self.install(a)
        res = r.generate("x", max_tokens=8192)
        self.assertEqual(res.model, "a-mid")
        self.assertEqual(res.trail[0]["outcome"], "empty")
        self.assertIn("raisonnement", res.trail[0]["detail"])
        self.assertGreater(r._model_wait(a, "a-big"), 0)

    def test_empty_answer_is_never_returned(self):
        a = fake("a", [answer("")])
        self.install(a)
        with self.assertRaises(r.LLMRouterError):
            r.generate("x")

    def test_reasoning_flag_and_truncation_are_reported(self):
        a = multi("a", [answer('{"offres": [', finish="length", reasoning="...")])
        self.install(a)
        res = r.generate("x")
        self.assertTrue(res.reasoning_used and res.truncated)    # texte coupé : à l'appelant de ne pas le prendre pour un JSON

    # ---------- délai, restriction, préférence ----------

    def test_timeout_only_lengthens_the_provider_timeout(self):
        a = multi("a", [answer()], timeout=20)
        self.install(a)
        r.generate("x", timeout=60)
        r.generate("x", timeout=5)
        r.generate("x")
        self.assertEqual(a["timeouts_asked"], [60.0])            # 5 s ne raccourcit pas les 20 s ; None ne change rien

    def test_only_restricts_and_prefer_orders(self):
        a, b = multi("a", [answer("a")]), multi("b", [answer("b")])
        self.install(a, b)
        self.assertEqual(r.generate("x").provider, "a")
        self.assertEqual(r.generate("x", prefer="b").provider, "b")
        self.assertEqual(r.generate("x", only=["b"]).provider, "b")
        self.assertEqual(r.generate("x", prefer="inconnu").provider, "a")     # préférence impossible : ignorée
        with self.assertRaises(r.LLMRouterError):
            r.generate("x", only=["inconnu"])

    def test_prefer_falls_back_when_the_preferred_provider_fails(self):
        a, b = multi("a", [answer("a")]), multi("b", [by_model({"b-big": err_429(30), "b-mid": err_429(30)})])
        self.install(a, b)
        res = r.generate("x", prefer="b")
        self.assertEqual(res.provider, "a")
        self.assertEqual([t["provider"] for t in res.trail], ["b", "b"])      # les deux modèles de b ont d'abord été essayés

    def test_available_providers_excludes_paused_ones(self):
        a, b = multi("a", [answer()]), multi("b", [answer()])
        self.install(a, b)
        self.assertEqual(r.available_providers(), ["a", "b"])
        r._put_in_cooldown("a", 60)
        self.assertEqual(r.available_providers(), ["b"])
        for m in ("b-big", "b-mid"):
            r._bench_model(b, m, 60, "test")
        self.assertEqual(r.available_providers(), [])            # tous les modèles de b en pause : b ne peut pas répondre

    def test_quota_status_lists_models_and_their_pauses(self):
        a = multi("a", [by_model({"a-big": err_429(30)})])
        self.install(a)
        r.generate("x")
        st = r.quota_status()["a"]
        self.assertEqual(st["models"], ["a-big", "a-mid"])
        self.assertIn("a-big", st["model_pauses"])


class ProviderEnvironmentTests(unittest.TestCase):
    """Variables alternatives, variables exigées et adresse à gabarit (Cloudflare : l'adresse porte l'identifiant du compte)."""

    SPEC = {"name": "cloudflare", "env_key": "CLOUDFLARE_API_KEY", "env_required": ["CLOUDFLARE_ACCOUNT_ID"],
            "env_aliases": {"CLOUDFLARE_API_KEY": ["CLOUDFLARE_API_TOKEN"],
                            "CLOUDFLARE_ACCOUNT_ID": ["CLOUDFLAIRE_ACCOUNT_ID", "CF_ACCOUNT_ID"]},
            "base_url": "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"}

    def configured(self, env):
        with mock.patch.dict(os.environ, env, clear=True):
            return r._is_configured(self.SPEC), r._resolve_base_url(self.SPEC)

    def test_key_alone_is_not_enough_when_the_account_id_is_required(self):
        self.assertFalse(self.configured({"CLOUDFLARE_API_KEY": "k"})[0])
        self.assertFalse(self.configured({"CLOUDFLARE_ACCOUNT_ID": "acc"})[0])

    def test_both_variables_configure_the_provider_and_fill_the_url(self):
        ok, url = self.configured({"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "acc123"})
        self.assertTrue(ok)
        self.assertTrue(url.endswith("/accounts/acc123/ai/v1"))

    def test_aliases_are_accepted_including_the_typo_found_in_a_real_env(self):
        ok, url = self.configured({"CLOUDFLARE_API_TOKEN": "k", "CLOUDFLAIRE_ACCOUNT_ID": "acc9"})
        self.assertTrue(ok)
        self.assertIn("/accounts/acc9/", url)

    def test_blank_values_do_not_count(self):
        self.assertFalse(self.configured({"CLOUDFLARE_API_KEY": "  ", "CLOUDFLARE_ACCOUNT_ID": "acc"})[0])


class ProbeModelsTests(RouterTestCase):
    """`probe_models` : lesquels des modèles LISTÉS répondent vraiment ? (NVIDIA : 32 sur 47 en 404, d'autres expirent.)"""

    def provider(self, table):
        a = multi("a", [by_model(table)], models=("big",))
        a["client"].models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in table]))
        self.install(a)
        return a

    def test_classifies_each_listed_model(self):
        timeout = _make_error(openai.APITimeoutError, "Request timed out.")
        a = self.provider({
            "bon-modele": answer("ok"),
            "retire": err_status(openai.NotFoundError, 404),
            "lent": timeout,
            "quota": err_429(5),
            "raisonneur": answer("", finish="length", reasoning="Here's a thinking process..."),
            "vide": answer(""),
        })
        rows = {x["model"]: x["status"] for x in r.probe_models("a", workers=1)}
        self.assertEqual(rows, {"bon-modele": "ok", "retire": "404 non appelable", "lent": "délai dépassé",
                                "quota": "429 quota", "raisonneur": "raisonne", "vide": "réponse vide"})
        self.assertEqual(len(a["calls"]), 6)

    def test_non_text_models_are_skipped_unless_asked(self):
        self.provider({"texte": answer("ok"), "vendor/embed-8b": answer("ok"), "x-rerank-1": answer("ok")})
        self.assertEqual([x["model"] for x in r.probe_models("a")], ["texte"])
        self.assertEqual(len(r.probe_models("a", include_all=True)), 3)

    def test_answering_models_come_first_and_the_probe_has_no_side_effect(self):
        self.provider({"mort": err_status(openai.NotFoundError, 404), "vivant": answer("ok"), "quota": err_429(60)})
        rows = r.probe_models("a", workers=1)
        self.assertEqual(rows[0]["model"], "vivant")
        self.assertEqual(r._provider_cooldown_until["a"], 0.0)          # diagnostic seul : aucune pause déclenchée
        self.assertEqual(r._model_cooldown_until, {})
        self.assertEqual(r._provider_stats["a"]["errors"], 0)

    def test_model_options_are_applied_to_the_probe(self):
        a = self.provider({"nemotron": answer("ok")})
        a["model_options"] = {"nemotron": THINKING_OFF}
        r.probe_models("a")
        self.assertEqual(a["calls"][0]["extra_body"], THINKING_OFF["extra_body"])

    def test_falls_back_to_the_catalog_models_when_listing_fails(self):
        a = self.provider({"a-big": answer("ok")})
        def boom():
            raise RuntimeError("liste indisponible")
        a["client"].models = SimpleNamespace(list=boom)
        self.assertEqual([x["model"] for x in r.probe_models("a")], ["a-big"])

    def test_unknown_provider(self):
        self.provider({"m": answer("ok")})
        with self.assertRaises(KeyError):
            r.probe_models("inconnu")


if __name__ == "__main__":
    unittest.main()
