"""
Refusals that free providers return, and what the router infers from them:

  * 402: offer not activated or model on a paid plan, one-hour pause (not one minute)
  * per-minute limit of ZERO (Mistral without a free offer): one-hour pause
  * 429 on a DAILY cap (Gemini, Cloudflare, OpenRouter): the model is back in hours, not in 60 s
  * 429 without Retry-After but with `x-ratelimit-reset-tokens`: wait for the announced reset
  * token budget announced by the headers of a successful response: a model without enough room is not called
  * `<think>…</think>` left in `content`: never returned as the answer
No network call, no quota consumed.
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai  # noqa: E402
from test_generate import answer, by_model, multi  # noqa: E402
from test_router import RouterTestCase, _make_error, fake, reply  # noqa: E402

from ricochet_llm import cleaning  # noqa: E402
from ricochet_llm import router as r  # noqa: E402


def err_402():
    return _make_error(openai.APIStatusError, "Error code: 402 - payment required", 402)


def err_429_with(headers=None, body=None, message="Error code: 429 - rate limited"):
    e = _make_error(openai.RateLimitError, message, 429, headers or {})
    e.body = body
    return e


class PaymentTests(RouterTestCase):
    def test_402_pauses_the_provider_for_an_hour_and_switches(self):
        a, b = self.install(fake("a", [err_402()]), fake("b", [reply("from b")]))
        self.assertEqual(r.chat([], "q"), "from b")
        self.assertAlmostEqual(r.quota_status()["a"]["cooldown_s"], r.COOLDOWN_PAYMENT_S, delta=2)
        r.chat([], "q2")
        self.assertEqual(len(a["calls"]), 1)                 # a plan that does not exist is not retried on every question

    def test_402_is_classified_apart_from_other_rejections(self):
        self.assertEqual(r._classify_error(err_402()), "payment")
        self.assertEqual(r._classify_error(_make_error(openai.APIStatusError, "x", 400)), "rejected")

    def test_a_limit_of_zero_requests_per_minute_means_the_free_offer_is_not_active(self):
        zero = err_429_with({"x-ratelimit-limit-req-minute": "0"})
        a, b = self.install(fake("a", [zero]), fake("b", [reply("from b")]))
        r.chat([], "q")
        self.assertAlmostEqual(r.quota_status()["a"]["cooldown_s"], r.COOLDOWN_PAYMENT_S, delta=2)

    def test_a_nonzero_limit_is_an_ordinary_429(self):
        ok = err_429_with({"x-ratelimit-limit-req-minute": "60"})
        self.install(fake("a", [ok]), fake("b", [reply("from b")]))
        r.chat([], "q")
        self.assertAlmostEqual(r.quota_status()["a"]["cooldown_s"], r.COOLDOWN_RATE_LIMIT_S, delta=2)


class RateLimitDelayTests(RouterTestCase):
    def test_a_daily_cap_in_the_body_pauses_the_model_for_an_hour(self):
        daily = err_429_with(body={"error": {"details": [{"violations": [
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
        a = multi("a", [by_model({"a-big": daily})])
        self.install(a)
        res = r.generate("x")
        self.assertEqual(res.model, "a-mid")
        self.assertAlmostEqual(r._model_wait(a, "a-big"), r.COOLDOWN_PAYMENT_S, delta=2)

    def test_a_daily_cap_in_the_message_is_recognised(self):
        daily = err_429_with(message="Error code: 429 - You have exhausted your daily free allocation")
        self.assertEqual(r._rate_limit_delay(daily), r.COOLDOWN_PAYMENT_S)

    def test_a_per_minute_429_stays_short(self):
        self.assertEqual(r._rate_limit_delay(err_429_with(message="Error code: 429 - rate limited")),
                         r.COOLDOWN_RATE_LIMIT_S)

    def test_retry_after_wins_over_everything(self):
        e = err_429_with({"retry-after": "7", "x-ratelimit-reset-tokens": "30s"}, message="daily")
        self.assertEqual(r._rate_limit_delay(e), 7)

    def test_the_announced_token_reset_is_used_without_retry_after(self):
        self.assertAlmostEqual(r._rate_limit_delay(err_429_with({"x-ratelimit-reset-tokens": "1m30.5s"})), 90.5)

    def test_durations_are_read_in_every_form_providers_use(self):
        cases = {"46.5s": 46.5, "1m30.5s": 90.5, "250ms": 0.25, "2m0s": 120.0, "12": 12.0, "1h": 3600.0, " 3s ": 3.0}
        for text, expected in cases.items():
            self.assertAlmostEqual(r._duration_text(text), expected, msg=text)
        for text in ("", "soon", None):
            self.assertIsNone(r._duration_text(text), msg=repr(text))


class BudgetTests(RouterTestCase):
    def budget_headers(self, remaining, reset="30s", limit="8000"):
        return {"x-ratelimit-limit-tokens": limit, "x-ratelimit-remaining-tokens": str(remaining),
                "x-ratelimit-reset-tokens": reset}

    def with_headers(self, headers):
        base = answer("ok")
        return SimpleNamespace(headers=headers, parse=base.parse)

    def test_the_announced_budget_is_remembered_after_a_successful_call(self):
        a = multi("a", [self.with_headers(self.budget_headers(500))])
        self.install(a)
        r.generate("x")
        self.assertEqual(r._model_limit[("a", "a-big")], 8000)                  # limit learned without waiting for a 413
        self.assertEqual(r._model_budget[("a", "a-big")][0], 500)

    def test_a_model_without_the_tokens_for_the_request_is_not_called(self):
        a = multi("a", [self.with_headers(self.budget_headers(100))])
        self.install(a)
        r.generate("x")                                                         # the heavy model answers and announces 100 tokens left
        a["calls"].clear()
        r.generate("y" * 900)                                                   # ~300 estimated tokens > 100 left
        self.assertEqual([c["model"] for c in a["calls"]], ["a-mid"])           # skipped without a call, hence no avoidable 429

    def test_the_model_comes_back_when_its_window_resets(self):
        a = multi("a", [self.with_headers(self.budget_headers(100, reset="30s"))])
        self.install(a)
        r.generate("x")
        self.assertGreater(r._model_wait(a, "a-big", need_tokens=300), 0)
        self.clock[0] += 31
        self.assertEqual(r._model_wait(a, "a-big", need_tokens=300), 0)

    def test_a_small_request_still_passes_with_a_low_budget(self):
        a = multi("a", [self.with_headers(self.budget_headers(100))])
        self.install(a)
        r.generate("x")
        self.assertEqual(r._model_wait(a, "a-big", need_tokens=20), 0)

    def test_unreadable_headers_are_ignored(self):
        a = multi("a", [self.with_headers({"x-ratelimit-limit-tokens": "lots", "x-ratelimit-remaining-tokens": "?"})])
        self.install(a)
        self.assertEqual(r.generate("x").model, "a-big")
        self.assertNotIn(("a", "a-big"), r._model_budget)

    def test_a_provider_that_announces_nothing_is_unaffected(self):
        a = multi("a", [answer("ok")])
        self.install(a)
        r.generate("x" * 3000)
        self.assertEqual(r._model_wait(a, "a-big", need_tokens=1000), 0)


class ThinkBlockTests(RouterTestCase):
    def test_a_think_block_is_not_part_of_the_answer(self):
        self.assertEqual(cleaning.strip_think("<think>let me think\nfor a while</think>The answer."), "The answer.")

    def test_an_unclosed_think_block_is_kept_rather_than_guessed(self):
        self.assertTrue(cleaning.strip_think("<think>cut off midway").startswith("<think>"))

    def test_generate_never_returns_an_answer_made_only_of_thoughts(self):
        a = multi("a", [by_model({"a-big": answer("<think>nothing to say</think>")})])
        self.install(a)
        res = r.generate("x")
        self.assertEqual((res.model, res.text), ("a-mid", "ok"))
        self.assertEqual(res.trail[0]["outcome"], "empty")

    def test_chat_strips_the_think_block(self):
        self.install(fake("a", [reply("<think>hmm</think>Hello!")]))
        self.assertEqual(r.chat([], "hi"), "Hello!")


class OrderEnvTests(unittest.TestCase):
    def test_ricochet_order_moves_providers_first_and_ignores_unknown_names(self):
        from unittest import mock

        from ricochet_llm import catalog
        with mock.patch.dict(os.environ, {catalog.ORDER_ENV_VAR: "gemini, whatever ,groq"}):
            names = [s["name"] for s in catalog.load_specs()]
        self.assertEqual(names[:2], ["gemini", "groq"])
        self.assertEqual(len(names), len(set(names)))

    def test_default_order_puts_gonka_after_openrouter(self):
        from ricochet_llm import catalog
        names = [s["name"] for s in catalog.BUILTIN_SPECS]
        self.assertLess(names.index("openrouter"), names.index("gonka"))      # 6 requests/min and 1 M tokens PER MONTH
        self.assertEqual(names[:4], ["groq", "nvidia", "ollama", "cloudflare"])


if __name__ == "__main__":
    unittest.main()
