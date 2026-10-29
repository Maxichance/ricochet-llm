"""
The OpenAI-compatible endpoints (/v1/models, /v1/chat/completions): same failover, caller-owned tools, optional key.
Fake providers, no network. Needs the dev extras: pip install -e ".[dev]".
"""
import json
import os
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
except Exception:           # httpx missing: these tests are skipped
    TestClient = None

from test_router import RouterTestCase, err_429, fake, reply, tool_reply  # noqa: E402

if TestClient is not None:
    from ricochet_llm.server import API_KEY_ENV_VAR, create_app

BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}


def sse_events(text):
    return [line[6:] for line in text.split("\n") if line.startswith("data: ")]


@unittest.skipIf(TestClient is None, "httpx required for fastapi.testclient")
class OpenAIApiTests(RouterTestCase):
    def client(self, *providers):
        self.install(*providers)
        return TestClient(create_app())

    def test_models_lists_auto_and_each_provider(self):
        c = self.client(fake("a", [reply()]), fake("b", [reply()]))
        body = c.get("/v1/models").json()
        self.assertEqual((body["object"], [m["id"] for m in body["data"]]), ("list", ["auto", "a", "b"]))

    def test_completion_has_the_openai_shape_and_fails_over(self):
        c = self.client(fake("a", [err_429(30)]), fake("b", [reply("hello")]))
        res = c.post("/v1/chat/completions", json=BODY)
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(body["x_ricochet"]["provider"], "b")

    def test_model_name_restricts_to_one_provider(self):
        a, b = fake("a", [reply("A")]), fake("b", [reply("B")])
        c = self.client(a, b)
        res = c.post("/v1/chat/completions", json={**BODY, "model": "b"})
        self.assertEqual(res.json()["choices"][0]["message"]["content"], "B")
        self.assertEqual(a["calls"], [])

    def test_tools_and_parameters_go_to_the_provider_and_tool_calls_come_back(self):
        b = fake("b", [tool_reply("lookup", "{}", "id-1")])
        c = self.client(b)
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        res = c.post("/v1/chat/completions", json={**BODY, "tools": tools, "tool_choice": "auto", "top_p": 0.3,
                                                   "max_tokens": 50, "temperature": 0.7})
        sent = b["calls"][0]
        self.assertEqual((sent["tools"], sent["top_p"], sent["max_tokens"], sent["temperature"]), (tools, 0.3, 50, 0.7))
        choice = res.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "lookup")

    def test_stream_sends_the_answer_as_sse_chunks_then_done(self):
        c = self.client(fake("a", [reply("streamed")]))
        res = c.post("/v1/chat/completions", json={**BODY, "stream": True})
        self.assertTrue(res.headers["content-type"].startswith("text/event-stream"))
        events = sse_events(res.text)
        self.assertEqual(events[-1], "[DONE]")
        first, last = json.loads(events[0]), json.loads(events[-2])
        self.assertEqual(first["object"], "chat.completion.chunk")
        self.assertEqual(first["choices"][0]["delta"], {"role": "assistant", "content": "streamed"})
        self.assertEqual(last["choices"][0]["finish_reason"], "stop")

    def test_stream_carries_tool_calls_with_an_index(self):
        c = self.client(fake("a", [tool_reply("lookup", "{}", "id-1")]))
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        res = c.post("/v1/chat/completions", json={**BODY, "stream": True, "tools": tools})
        delta = json.loads(sse_events(res.text)[0])["choices"][0]["delta"]
        self.assertEqual(delta["tool_calls"][0]["index"], 0)
        self.assertEqual(delta["tool_calls"][0]["function"]["name"], "lookup")

    def test_bad_requests_are_refused_in_the_openai_error_format(self):
        c = self.client(fake("a", [reply()]))
        for body in ({}, {"messages": []}, {"messages": ["x"]}, {**BODY, "max_tokens": "many"},
                     {**BODY, "tools": "nope"}):
            res = c.post("/v1/chat/completions", json=body)
            self.assertEqual(res.status_code, 400, body)
            self.assertEqual(res.json()["error"]["type"], "invalid_request_error")
        self.assertEqual(c.post("/v1/chat/completions", content="not json").status_code, 400)

    def test_every_provider_down_is_a_503(self):
        c = self.client(fake("a", [err_429(30)]))
        res = c.post("/v1/chat/completions", json=BODY)
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["error"]["type"], "server_error")

    def test_the_key_is_required_only_when_configured(self):
        c = self.client(fake("a", [reply("ok")]))
        self.assertEqual(c.get("/v1/models").status_code, 200)
        with mock.patch.dict(os.environ, {API_KEY_ENV_VAR: "secret"}):
            self.assertEqual(c.get("/v1/models").status_code, 401)
            self.assertEqual(c.post("/v1/chat/completions", json=BODY,
                                    headers={"Authorization": "Bearer wrong"}).status_code, 401)
            ok = c.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer secret"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(c.get("/api/health").status_code, 200)       # the dashboard API is not touched

    def test_the_official_openai_sdk_reads_both_the_plain_and_the_streamed_answer(self):
        import openai
        c = self.client(fake("a", [reply("sdk answer"), reply("streamed by sdk")]))
        sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=c)
        plain = sdk.chat.completions.create(model="auto", messages=BODY["messages"])
        self.assertEqual(plain.choices[0].message.content, "sdk answer")
        stream = sdk.chat.completions.create(model="auto", messages=BODY["messages"], stream=True)
        self.assertEqual("".join(ch.choices[0].delta.content or "" for ch in stream), "streamed by sdk")


if __name__ == "__main__":
    unittest.main()
