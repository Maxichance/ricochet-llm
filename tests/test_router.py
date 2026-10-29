"""
Router tests with fake providers: no network call, no quota consumed.
Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest import mock

import openai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet_llm import router as r  # noqa: E402
from ricochet_llm.tools import registry  # noqa: E402

# ---------- fake provider factory ----------

def _make_error(cls, message, status=None, headers=None):
    """Instantiate an SDK exception without going through its constructor (whose signature changes
    from one openai version to the next); only the attributes read by the router are set."""
    e = cls.__new__(cls)
    e.args = (message,)
    e.message = message
    e.body = None
    e.status_code = status
    e.response = SimpleNamespace(status_code=status, headers=headers or {})
    return e


def err_429(retry_after=None):
    h = {"retry-after": str(retry_after)} if retry_after is not None else None
    return _make_error(openai.RateLimitError, "Error code: 429 - rate limited", 429, h)


def err_status(cls, status):
    return _make_error(cls, f"Error code: {status} - boom", status)


def err_connection():
    return _make_error(openai.APIConnectionError, "Connection error.")


def reply(content="ok", finish="stop", tool_calls=None, headers=None, choices=True, tokens=0):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish)] if choices else [],
        usage=SimpleNamespace(total_tokens=tokens) if tokens else None,
    )
    return SimpleNamespace(headers=headers or {}, parse=lambda: resp)


def tool_call(name, args="{}", call_id="c1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


def tool_reply(name, args="{}", call_id="c1"):
    return reply(content=None, finish="tool_calls", tool_calls=[tool_call(name, args, call_id)])


def fake(name, script, supports_tools=True, quota_header="x-q", quota_min=100):
    """script: list of items consumed in order (the last one repeats) —
    an exception (raised), a response (returned) or a function(kwargs)."""
    script, calls = list(script), []

    def create(**kwargs):
        calls.append(kwargs)
        item = script.pop(0) if len(script) > 1 else script[0]
        if callable(item):
            item = item(kwargs)
        if isinstance(item, BaseException):
            raise item
        return item

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create))))
    return {"name": name, "client": client, "calls": calls, "model_heavy": f"{name}-big",
            "model_light": f"{name}-small", "quota_header": quota_header, "quota_min": quota_min,
            "supports_tools": supports_tools}


class RouterTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = [1000.0]
        self._saved = (r._PROVIDERS, r._DISABLED, r._now, r._sleep)
        r._now = lambda: self.clock[0]
        r._sleep = lambda s: None
        self._saved_tools = dict(registry._tools)
        registry.clear()

        @registry.register
        def find_customer(email: str = None) -> dict:
            """Look up a customer by email."""
            return {"email": email, "name": "John Smith"}

    def tearDown(self):
        r._PROVIDERS, r._DISABLED, r._now, r._sleep = self._saved
        r._reset_state()
        registry._tools.clear()
        registry._tools.update(self._saved_tools)

    def install(self, *providers, disabled=()):
        r._PROVIDERS = list(providers)
        r._DISABLED = list(disabled)
        r._reset_state()
        return providers


class FailoverTests(RouterTestCase):
    def test_nominal_uses_first_provider(self):
        a, b = self.install(fake("a", [reply("hi")]), fake("b", [reply("other")]))
        history = []
        self.assertEqual(r.chat(history, "hello"), "hi")
        self.assertEqual((len(a["calls"]), len(b["calls"])), (1, 0))
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])

    def test_429_switches_and_cools_down_using_retry_after(self):
        a, b = self.install(fake("a", [err_429(retry_after=120)]), fake("b", [reply("from b")]))
        self.assertEqual(r.chat([], "q1"), "from b")
        st = r.quota_status()["a"]
        self.assertEqual(st["status"], "cooldown")
        self.assertTrue(100 <= st["cooldown_s"] <= 120)

        r.chat([], "q2")                       # a is in cooldown: it is not called again
        self.assertEqual(len(a["calls"]), 1)

        self.clock[0] += 121                   # cooldown over: a is probed again
        self.assertEqual(r.quota_status()["a"]["status"], "unknown")
        r.chat([], "q3")
        self.assertEqual(len(a["calls"]), 2)

    def test_429_without_retry_after_uses_default_cooldown(self):
        self.install(fake("a", [err_429()]), fake("b", [reply()]))
        r.chat([], "q")
        self.assertAlmostEqual(r.quota_status()["a"]["cooldown_s"], r.COOLDOWN_RATE_LIMIT_S, delta=2)

    def test_all_providers_rate_limited_terminates_and_rolls_back_history(self):
        # Regression: the old version looped forever in this case.
        a, b = self.install(fake("a", [err_429()]), fake("b", [err_429()]))
        history = [{"role": "user", "content": "before"}, {"role": "assistant", "content": "answer"}]
        with self.assertRaises(r.LLMRouterError) as ctx:
            r.chat(history, "question")
        self.assertIn("Retry in", str(ctx.exception))
        self.assertEqual((len(a["calls"]), len(b["calls"])), (1, 1))
        self.assertEqual(len(history), 2)      # the user's message did not stay in the history

    def test_model_removed_404_switches_with_long_cooldown(self):
        a, _ = self.install(fake("a", [err_status(openai.NotFoundError, 404)]), fake("b", [reply("ok b")]))
        self.assertEqual(r.chat([], "q"), "ok b")
        self.assertAlmostEqual(r.quota_status()["a"]["cooldown_s"], r.COOLDOWN_CONFIG_S, delta=2)

    def test_invalid_key_401_switches(self):
        self.install(fake("a", [err_status(openai.AuthenticationError, 401)]), fake("b", [reply("ok b")]))
        self.assertEqual(r.chat([], "q"), "ok b")

    def test_request_rejected_400_switches_without_cooldown(self):
        self.install(fake("a", [err_status(openai.BadRequestError, 400)]), fake("b", [reply("ok b")]))
        self.assertEqual(r.chat([], "q"), "ok b")
        self.assertEqual(r.quota_status()["a"]["status"], "unknown")   # no cooldown: specific to the request

    def test_transient_error_is_retried_on_same_provider(self):
        a, b = self.install(fake("a", [err_connection(), reply("ok a")]), fake("b", [reply("ok b")]))
        self.assertEqual(r.chat([], "q"), "ok a")
        self.assertEqual((len(a["calls"]), len(b["calls"])), (2, 0))

    def test_persistent_transient_error_switches_after_retry(self):
        a, _ = self.install(fake("a", [err_status(openai.InternalServerError, 503)]), fake("b", [reply("ok b")]))
        self.assertEqual(r.chat([], "q"), "ok b")
        self.assertEqual(len(a["calls"]), 1 + r.TRANSIENT_RETRIES)
        self.assertEqual(r.quota_status()["a"]["status"], "cooldown")

    def test_programming_error_is_not_swallowed(self):
        a, b = self.install(fake("a", [TypeError("local bug")]), fake("b", [reply()]))
        with self.assertRaises(TypeError):
            r.chat([], "q")
        self.assertEqual(len(b["calls"]), 0)

    def test_response_without_choices_switches(self):
        self.install(fake("a", [reply(choices=False)]), fake("b", [reply("ok b")]))
        self.assertEqual(r.chat([], "q"), "ok b")
        self.assertEqual(r.quota_status()["a"]["status"], "cooldown")

    def test_empty_answer_switches_and_nudges_once(self):
        a, b = self.install(fake("a", [reply("")]), fake("b", [reply("ok b")]))
        self.assertEqual(r.chat([], "q"), "ok b")
        sent = b["calls"][0]["messages"]
        self.assertEqual(sum(m["content"] == "Answer my previous question." for m in sent), 1)

    def test_error_message_reports_causes_when_everyone_rejected_the_request(self):
        self.install(fake("a", [err_status(openai.BadRequestError, 400)]),
                     fake("b", [err_status(openai.BadRequestError, 400)]))
        with self.assertRaises(r.LLMRouterError) as ctx:
            r.chat([], "q")
        self.assertIn("a → BadRequestError", str(ctx.exception))
        self.assertIn("b → BadRequestError", str(ctx.exception))

    def test_no_provider_configured(self):
        self.install()
        with self.assertRaises(r.LLMRouterError):
            r.chat([], "q")


class QuotaTests(RouterTestCase):
    def test_low_quota_header_makes_provider_ineligible_for_heavy(self):
        a, b = self.install(fake("a", [reply(headers={"x-q": "50"})]), fake("b", [reply("ok b")]))
        r.chat([], "q1")
        self.assertEqual(r.quota_status()["a"]["status"], "limit")
        r.chat([], "q2")
        self.assertEqual(len(b["calls"]), 1)
        self.assertEqual(b["calls"][0]["model"], "b-big")

    def test_light_mode_when_everyone_is_low(self):
        a, b = self.install(
            fake("a", [reply(headers={"x-q": "10"})]),
            fake("b", [reply(headers={"x-q": "90"})]),
        )
        r.chat([], "q1")     # a answers (quota unknown → HEAVY) and drops to 10
        r.chat([], "q2")     # b takes over and drops to 90
        r.chat([], "q3")     # all below the threshold (100) → LIGHT on the one with the most quota: b
        self.assertEqual(b["calls"][-1]["model"], "b-small")

    def test_stale_quota_is_probed_again(self):
        self.install(fake("a", [reply(headers={"x-q": "10"})]), fake("b", [reply()]))
        r.chat([], "q")
        self.assertEqual(r.quota_status()["a"]["status"], "limit")
        self.clock[0] += r.QUOTA_STALE_S + 1
        self.assertEqual(r.quota_status()["a"]["status"], "unknown")


class ToolTests(RouterTestCase):
    def test_tool_call_is_executed_and_result_sent_back(self):
        (a,) = self.install(fake("a", [tool_reply("find_customer", '{"email": "x@y.fr"}'), reply("found")]))
        history = []
        self.assertEqual(r.chat(history, "look up x@y.fr"), "found")
        tool_msg = [m for m in a["calls"][1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("x@y.fr", tool_msg["content"])

    def test_unknown_tool_is_reported_to_model_not_raised(self):
        (a,) = self.install(fake("a", [tool_reply("nonexistent_tool"), reply("sorry")]))
        self.assertEqual(r.chat([], "q"), "sorry")
        tool_msg = [m for m in a["calls"][1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("error", tool_msg["content"])
        self.assertIn("nonexistent_tool", tool_msg["content"])

    def test_bad_arguments_are_reported_to_model_not_raised(self):
        (a,) = self.install(fake("a", [tool_reply("find_customer", '{"bogus": 1}'), reply("ok")]))
        self.assertEqual(r.chat([], "q"), "ok")
        tool_msg = [m for m in a["calls"][1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("TypeError", tool_msg["content"])

    def test_invalid_json_arguments_are_reported_to_model(self):
        (a,) = self.install(fake("a", [tool_reply("find_customer", "{not json"), reply("ok")]))
        self.assertEqual(r.chat([], "q"), "ok")
        tool_msg = [m for m in a["calls"][1]["messages"] if m["role"] == "tool"][0]
        self.assertIn("error", tool_msg["content"])

    def test_tool_loop_is_capped(self):
        (a,) = self.install(fake("a", [tool_reply("find_customer")]))
        history = []
        with self.assertRaises(r.LLMRouterError):
            r.chat(history, "q")
        self.assertEqual(len(a["calls"]), r.MAX_TOOL_ROUNDS + 1)
        self.assertEqual(history, [])

    def test_mid_tool_exchange_never_falls_back_to_provider_without_tools(self):
        a, b = self.install(
            fake("a", [tool_reply("find_customer"), err_429()]),
            fake("b", [reply("must not be called")], supports_tools=False),
        )
        with self.assertRaises(r.LLMRouterError):
            r.chat([], "q")
        self.assertEqual(len(b["calls"]), 0)

    def test_no_registered_tool_means_no_tools_param_even_for_tool_providers(self):
        registry.clear()
        (a,) = self.install(fake("a", [reply()]))
        r.chat([], "q")
        self.assertNotIn("tools", a["calls"][0])

    def test_without_tools_the_catalog_order_wins_over_tool_support(self):
        registry.clear()
        a, b = self.install(fake("a", [reply("from a")], supports_tools=False), fake("b", [reply("from b")]))
        self.assertEqual(r.chat([], "q"), "from a")

    def test_system_prompt_is_configurable(self):
        (a,) = self.install(fake("a", [reply()]))
        previous = r.get_system_prompt()
        try:
            r.set_system_prompt("Answer like a pirate")
            r.chat([], "q")
        finally:
            r.set_system_prompt(previous)
        self.assertEqual(a["calls"][0]["messages"][0], {"role": "system", "content": "Answer like a pirate"})

    def test_provider_without_tools_gets_no_tools_param(self):
        (a,) = self.install(fake("a", [reply()], supports_tools=False))
        r.chat([], "q")
        self.assertNotIn("tools", a["calls"][0])


class ObservabilityTests(RouterTestCase):
    def test_chat_detailed_reports_provider_model_and_failover_trail(self):
        self.install(fake("a", [err_429()]), fake("b", [reply("from b")]))
        result = r.chat_detailed([], "q")
        self.assertEqual((result.answer, result.provider, result.model, result.mode), ("from b", "b", "b-big", "HEAVY"))
        self.assertEqual([(t["provider"], t["outcome"]) for t in result.trail], [("a", "rate_limit")])
        self.assertIn("provider", result.to_dict())

    def test_stats_count_calls_latency_and_tokens(self):
        self.install(fake("a", [reply(tokens=120)]))
        r.chat([], "q1")
        r.chat([], "q2")
        st = r.quota_status()["a"]["stats"]
        self.assertEqual((st["ok"], st["errors"], st["tokens"], st["calls_last_minute"]), (2, 0, 240, 2))
        self.assertIsNotNone(st["avg_latency_ms"])

    def test_errors_are_counted_per_provider(self):
        self.install(fake("a", [err_429()]), fake("b", [reply()]))
        r.chat([], "q")
        status = r.quota_status()
        self.assertEqual((status["a"]["stats"]["errors"], status["b"]["stats"]["ok"]), (1, 1))

    def test_quota_gauges_are_built_from_ratelimit_headers(self):
        p = fake("a", [reply(headers={
            "x-ratelimit-remaining-tokens": "7000", "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-requests": "900", "x-ratelimit-limit-requests": "1000",
        })], quota_header="x-ratelimit-remaining-tokens")
        p["quota_labels"] = {"tokens": "tokens / min", "requests": "requests / day"}
        self.install(p)
        r.chat([], "q")
        limits = {g["label"]: (g["remaining"], g["limit"]) for g in r.quota_status()["a"]["limits"]}
        self.assertEqual(limits, {"tokens / min": (7000, 8000), "requests / day": (900, 1000)})

    def test_disabled_providers_are_listed_without_key(self):
        self.install(fake("a", [reply()]), disabled=[{
            "name": "cerebras", "label": "Cerebras", "env_key": "CEREBRAS_API_KEY", "supports_tools": True}])
        status = r.quota_status()
        self.assertEqual((status["cerebras"]["status"], status["cerebras"]["configured"]), ("disabled", False))
        self.assertTrue(status["a"]["configured"])

    def test_key_expiry_is_reported_in_days(self):
        p = fake("a", [reply()])
        p["expires_at"] = (date.today() + timedelta(days=10)).isoformat()
        self.install(p)
        st = r.quota_status()["a"]
        self.assertEqual(st["key_expires_in_days"], 10)
        self.assertEqual(st["key_expires_at"], p["expires_at"])

    def test_invalid_key_expiry_is_ignored(self):
        p = fake("a", [reply()])
        p["expires_at"] = "not a date"
        self.install(p)
        self.assertIsNone(r.quota_status()["a"]["key_expires_in_days"])

    def test_events_are_newest_first(self):
        self.install(fake("a", [err_429()]), fake("b", [reply()]))
        r.chat([], "q")
        kinds = [e["kind"] for e in r.recent_events()]
        self.assertEqual(kinds[:2], ["ok", "rate_limit"])

    def test_next_provider_reports_choice_then_error_when_all_down(self):
        self.install(fake("a", [err_429()]), fake("b", [err_429()]))
        self.assertEqual(r.next_provider()["provider"], "a")
        with self.assertRaises(r.LLMRouterError):
            r.chat([], "q")
        nxt = r.next_provider()
        self.assertIsNone(nxt["provider"])
        self.assertIn("No provider", nxt["error"])


class ErrorDescriptionTests(unittest.TestCase):
    def make(self, cls, status, body, message="Error code: x - raw"):
        e = _make_error(cls, message, status)
        e.body = body
        return e

    def test_openrouter_upstream_reason_is_extracted_from_metadata(self):
        e = self.make(openai.RateLimitError, 429, {"error": {
            "message": "Provider returned error", "code": 429,
            "metadata": {"raw": "google/gemma:free is temporarily rate-limited upstream"}}})
        self.assertEqual(r._describe_error(e), "RateLimitError 429: google/gemma:free is temporarily rate-limited upstream")

    def test_gemini_list_body_is_supported(self):
        e = self.make(openai.NotFoundError, 404, [{"error": {"code": 404, "message": "model removed"}}])
        self.assertEqual(r._describe_error(e), "NotFoundError 404: model removed")

    def test_nvidia_detail_field_is_used(self):
        e = self.make(openai.NotFoundError, 404, {"status": 404, "detail": "Function 'x': Not found"})
        self.assertEqual(r._describe_error(e), "NotFoundError 404: Function 'x': Not found")

    def test_falls_back_to_exception_text_without_prefix(self):
        e = _make_error(openai.BadRequestError, "Error code: 400 - boom", 400)
        self.assertEqual(r._describe_error(e), "BadRequestError 400: boom")


class ManualActionTests(RouterTestCase):
    def test_ping_success_lifts_cooldown_and_records_latency(self):
        a, _ = self.install(fake("a", [err_429(), reply()]), fake("b", [reply()]))
        r.chat([], "q")                                           # a → 429 → cooldown
        self.assertEqual(r.quota_status()["a"]["status"], "cooldown")
        result = r.ping_provider("a")
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "a-small")              # the ping uses the light model
        self.assertEqual(r.quota_status()["a"]["status"], "unknown")

    def test_ping_failure_reports_kind_and_error(self):
        self.install(fake("a", [err_status(openai.AuthenticationError, 401)]))
        result = r.ping_provider("a")
        self.assertFalse(result["ok"])
        self.assertEqual(result["kind"], "config")
        self.assertEqual(r.quota_status()["a"]["status"], "cooldown")

    def test_ping_transient_failure_does_not_sleep_and_reports(self):
        self.install(fake("a", [err_connection()]))
        slept = []
        r._sleep = slept.append
        result = r.ping_provider("a")
        self.assertEqual((result["ok"], result["kind"]), (False, "transient"))
        self.assertEqual(slept, [])

    def test_ping_unknown_provider_raises(self):
        self.install(fake("a", [reply()]))
        with self.assertRaises(KeyError):
            r.ping_provider("zzz")

    def test_reset_provider_lifts_cooldown(self):
        self.install(fake("a", [err_429()]), fake("b", [reply()]))
        r.chat([], "q")
        r.reset_provider("a")
        self.assertEqual(r.quota_status()["a"]["status"], "unknown")
        self.assertEqual(r.next_provider()["provider"], "a")


class StripTests(RouterTestCase):
    def test_code_blocks_are_preserved(self):
        text = "Here:\n```python\nprint('x')\n```\nEnd."
        self.assertEqual(r._strip_tool_artifacts(text), text)

    def test_legit_json_is_preserved(self):
        text = 'Example: {"name": "Smith", "city": "London"} there.'
        self.assertEqual(r._strip_tool_artifacts(text), text)

    def test_leaked_tool_call_with_nested_json_is_removed(self):
        text = 'Looking. {"name": "find_customer", "arguments": {"email": "a@b.fr", "opts": {"x": 1}}} Done.'
        self.assertEqual(r._strip_tool_artifacts(text), "Looking.  Done.")

    def test_leaked_type_function_is_removed(self):
        text = '{"type": "function", "function": {"name": "n", "arguments": "{}"}}'
        self.assertEqual(r._strip_tool_artifacts(text), "")

    def test_fenced_leaked_tool_call_is_removed_with_its_fence(self):
        text = 'Ok\n```json\n{"name": "find_customer", "arguments": {}}\n```\nFinished'
        self.assertEqual(r._strip_tool_artifacts(text), "Ok\n\nFinished")

    def test_function_tag_is_removed(self):
        text = 'Hello <function=find_customer>{"email": "a@b.fr"}</function> thanks'
        self.assertEqual(r._strip_tool_artifacts(text), "Hello  thanks")

    def test_json_with_unknown_name_is_kept(self):
        text = '{"name": "Smith", "arguments": "x"}'
        self.assertEqual(r._strip_tool_artifacts(text), text)

    def test_unbalanced_brace_does_not_crash(self):
        self.assertEqual(r._strip_tool_artifacts("a { b"), "a { b")


class ConfigTests(unittest.TestCase):
    def test_providers_without_key_are_ignored(self):
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}, clear=True), \
             mock.patch.object(r, "load_dotenv", lambda: None):
            providers = r._build_providers()
        self.assertEqual([p["name"] for p in providers], ["mistral"])
        self.assertEqual(providers[0]["client"].max_retries, 0)   # the SDK does not retry: the router does the failover

    def test_no_keys_means_no_providers_and_no_crash(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(r._build_providers(), [])


if __name__ == "__main__":
    unittest.main()
