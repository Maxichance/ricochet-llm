"""
`complete()`: OpenAI-format completions with failover, where the caller owns the messages and the tools.
Also the async entry points and the state file that survives a restart. No network call, no quota consumed.
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_router import RouterTestCase, err_429, fake, reply, tool_reply  # noqa: E402

from ricochet_llm import aio  # noqa: E402
from ricochet_llm import router as r  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]


class CompleteTests(RouterTestCase):
    def test_returns_an_openai_style_completion_with_routing_info(self):
        self.install(fake("a", [reply("hello", tokens=9)]))
        out = r.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(out["object"], "chat.completion")
        self.assertEqual(out["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"]["total_tokens"], 9)
        self.assertEqual((out["x_ricochet"]["provider"], out["x_ricochet"]["trail"]), ("a", []))

    def test_fails_over_like_the_other_entry_points(self):
        self.install(fake("a", [err_429(30)]), fake("b", [reply("from b")]))
        out = r.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(out["x_ricochet"]["provider"], "b")
        self.assertEqual(out["x_ricochet"]["trail"][0]["provider"], "a")

    def test_messages_go_through_as_they_are_and_no_system_prompt_is_added(self):
        a = fake("a", [reply("ok")])
        self.install(a)
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        r.complete(messages, max_tokens=77, temperature=0.9, top_p=0.5)
        sent = a["calls"][0]
        self.assertEqual(sent["messages"], messages)
        self.assertEqual((sent["max_tokens"], sent["temperature"], sent["top_p"]), (77, 0.9, 0.5))
        self.assertNotIn("tools", sent)

    def test_tool_calls_come_back_to_the_caller_and_only_tool_capable_providers_are_used(self):
        a = fake("a", [reply("never")], supports_tools=False)
        b = fake("b", [tool_reply("lookup", '{"q": 1}', "call-7")])
        self.install(a, b)
        out = r.complete([{"role": "user", "content": "find"}], tools=TOOLS, tool_choice="auto")
        self.assertEqual(a["calls"], [])
        self.assertEqual(b["calls"][0]["tools"], TOOLS)
        self.assertEqual(b["calls"][0]["tool_choice"], "auto")
        message = out["choices"][0]["message"]
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"], [{"id": "call-7", "type": "function",
                                                  "function": {"name": "lookup", "arguments": '{"q": 1}'}}])
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")

    def test_an_empty_answer_without_tool_calls_still_fails_over(self):
        self.install(fake("a", [reply("")]), fake("b", [reply("full")]))
        out = r.complete([{"role": "user", "content": "hi"}])
        self.assertEqual((out["x_ricochet"]["provider"], out["x_ricochet"]["trail"][0]["outcome"]), ("b", "empty"))

    def test_think_blocks_are_removed_from_the_content(self):
        self.install(fake("a", [reply("<think>hmm</think>Answer")]))
        self.assertEqual(r.complete([{"role": "user", "content": "x"}])["choices"][0]["message"]["content"], "Answer")

    def test_only_restricts_the_providers_and_unknown_parameters_are_refused(self):
        a, b = fake("a", [reply("A")]), fake("b", [reply("B")])
        self.install(a, b)
        self.assertEqual(r.complete([{"role": "user", "content": "x"}], only=["b"])["x_ricochet"]["provider"], "b")
        with self.assertRaises(TypeError):
            r.complete([{"role": "user", "content": "x"}], logit_bias={})

    def test_raises_when_nothing_can_answer(self):
        self.install(fake("a", [err_429(30)]))
        with self.assertRaises(r.LLMRouterError):
            r.complete([{"role": "user", "content": "x"}])


class AsyncTests(RouterTestCase):
    def test_async_entry_points_return_what_the_sync_ones_return(self):
        self.install(fake("a", [reply("one"), reply("two"), reply("three"), reply("four")]))

        async def run():
            history = []
            return (await aio.achat(history, "q"), (await aio.achat_detailed(history, "q")).provider,
                    (await aio.agenerate("p", max_tokens=5)).text,
                    (await aio.acomplete([{"role": "user", "content": "x"}]))["choices"][0]["message"]["content"])

        self.assertEqual(asyncio.run(run()), ("one", "a", "three", "four"))


class StateFileTests(RouterTestCase):
    def setUp(self):
        super().setUp()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "sub", "state.json")
        patcher = mock.patch.dict(os.environ, {r.STATE_ENV_VAR: self.path})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pauses_and_learned_limits_survive_a_restart(self):
        a = fake("a", [reply()])
        self.install(a)
        r._bench_model(a, "a-big", 300, "429")
        r._put_in_cooldown("a", 600, exhausted=True, reason="daily cap")
        with r._quota_lock:
            r._model_limit[("a", "a-big")] = 4000
        self.assertTrue(r.save_state())
        r._reset_state()
        self.assertEqual(r._provider_cooldown_until["a"], 0.0)
        self.assertTrue(r.load_state())
        self.assertAlmostEqual(r._provider_cooldown_until["a"] - r._now(), 600, delta=5)
        self.assertEqual(r._provider_last_error["a"], "daily cap")
        self.assertGreater(r._model_wait(a, "a-big"), 250)
        self.assertEqual(r._model_limit[("a", "a-big")], 4000)

    def test_finished_pauses_and_unknown_providers_are_dropped(self):
        self.install(fake("a", [reply()]))
        os.makedirs(os.path.dirname(self.path))
        past, future = 1.0, 9e12
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"providers": {"a": {"until": past}, "ghost": {"until": future}, "bad": 3},
                       "models": [{"provider": "a", "model": "m", "until": past}, {"oops": 1}],
                       "limits": [{"provider": "ghost", "model": "m", "tokens": 5}, "junk"]}, f)
        self.assertTrue(r.load_state())
        self.assertEqual(r._provider_cooldown_until["a"], 0.0)
        self.assertNotIn("ghost", r._provider_cooldown_until)
        self.assertEqual((r._model_cooldown_until, r._model_limit), ({}, {}))

    def test_a_corrupt_file_is_ignored_and_without_the_variable_nothing_is_written(self):
        self.install(fake("a", [reply()]))
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertFalse(r.load_state())
        with mock.patch.dict(os.environ, {r.STATE_ENV_VAR: ""}):
            self.assertFalse(r.save_state())
            r._put_in_cooldown("a", 60)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not json")

    def test_a_pause_is_written_as_it_happens(self):
        self.install(fake("a", [reply()]))
        r._put_in_cooldown("a", 120, reason="x")
        with open(self.path, encoding="utf-8") as f:
            self.assertIn("a", json.load(f)["providers"])


if __name__ == "__main__":
    unittest.main()
