"""
Parallel executor: spreading across providers, caps, pauses, clean stop.

A limit of "40 requests per minute" applies to the START rate, not to concurrency: at 20 s per response, 2
workers make ~5 requests per minute, ~13 are needed to reach 40. No network call.
"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_generate import answer, multi  # noqa: E402
from test_router import RouterTestCase  # noqa: E402

from ricochet_llm import router as r  # noqa: E402
from ricochet_llm.parallel import Interrupted, parse_limits, run_parallel  # noqa: E402


def providers(**conc):
    """Test providers: `a=3` = at most three simultaneous requests; no gap between starts."""
    out = []
    for name, n in conc.items():
        p = multi(name, [answer()], models=("big",))
        p["max_concurrency"], p["min_interval_s"] = n, 0.0
        out.append(p)
    return out


class ParallelTests(RouterTestCase):
    def run_(self, items, work, **kw):
        kw.setdefault("pause_wait", 0.01)
        return run_parallel(items, work, **kw)

    def test_every_item_is_processed_once_and_every_provider_works(self):
        self.install(*providers(a=2, b=2, c=2))
        seen, lock = [], threading.Lock()

        def work(item, name):
            time.sleep(0.005)
            with lock:
                seen.append((item, name))

        summary = self.run_(range(60), work)
        self.assertEqual(sorted(i for i, _ in seen), list(range(60)))
        self.assertEqual((summary["processed"], summary["remaining"], summary["interrupted"], summary["errors"]), (60, 0, False, 0))
        self.assertEqual(set(summary["per_provider"]), {"a", "b", "c"})
        self.assertEqual(sum(summary["per_provider"].values()), 60)

    def test_catalog_concurrency_is_the_default_and_limits_override_it(self):
        self.install(*providers(a=1, b=4))
        active, peak, lock = {}, {}, threading.Lock()

        def work(item, name):
            with lock:
                active[name] = active.get(name, 0) + 1
                peak[name] = max(peak.get(name, 0), active[name])
            time.sleep(0.02)
            with lock:
                active[name] -= 1

        self.run_(range(60), work)
        self.assertEqual(peak["a"], 1)
        self.assertTrue(2 <= peak["b"] <= 4, peak)
        peak.clear()
        self.run_(range(60), work, limits={"a": 5})                   # adjustable without touching the catalog
        self.assertTrue(3 <= peak["a"] <= 5, peak)

    def test_a_paused_provider_takes_no_work(self):
        self.install(*providers(a=2, b=2))
        r._put_in_cooldown("a", 3600)
        who = []
        summary = self.run_(range(20), lambda item, name: who.append(name))
        self.assertEqual(set(who), {"b"})
        self.assertEqual((summary["processed"], summary["remaining"]), (20, 0))

    def test_a_paused_provider_does_not_hold_the_end_of_the_run(self):
        """A provider paused for an hour must not hold up the end of the run once everything is already processed."""
        self.install(*providers(a=2, b=2))
        r._put_in_cooldown("a", 3600)
        start = time.monotonic()
        self.run_(range(6), lambda item, name: None)
        self.assertLess(time.monotonic() - start, 2)

    def test_no_provider_available_stops_cleanly_and_returns_the_items(self):
        self.install(*providers(a=2))
        r._put_in_cooldown("a", 3600)
        called = []
        summary = self.run_(range(5), lambda item, name: called.append(item))
        self.assertEqual((called, summary["remaining"], summary["interrupted"], summary["processed"]), ([], 5, True, 0))

    def test_interrupted_from_work_stops_everything_and_gives_the_item_back(self):
        self.install(*providers(a=2, b=2))
        done, lock = [], threading.Lock()

        def work(item, name):
            if item == 5:
                raise Interrupted("quota")
            time.sleep(0.003)
            with lock:
                done.append(item)

        summary = self.run_(range(40), work)
        self.assertTrue(summary["interrupted"])
        self.assertEqual(summary["processed"] + summary["remaining"], 40)    # the interrupted item is returned to the queue
        self.assertNotIn(5, done)

    def test_router_error_inside_work_counts_as_all_providers_down(self):
        self.install(*providers(a=1))

        def work(item, name):
            raise r.LLMRouterError("No provider available.")

        summary = self.run_(range(4), work)
        self.assertEqual((summary["interrupted"], summary["processed"], summary["remaining"]), (True, 0, 4))

    def test_an_error_on_one_item_does_not_stop_the_others(self):
        self.install(*providers(a=2, b=2))

        def work(item, name):
            if item % 4 == 0:
                raise ValueError("odd response")

        summary = self.run_(range(12), work)
        self.assertEqual((summary["errors"], summary["processed"], summary["remaining"], summary["interrupted"]), (3, 9, 0, False))

    def test_departures_of_one_provider_are_spaced_by_min_interval(self):
        (a,) = providers(a=3)                                  # three workers, but ONE start at a time
        a["min_interval_s"] = 10.0
        self.install(a)
        t, pauses, lock = [0.0], [], threading.Lock()

        def clock():
            return t[0]

        def sleeper(d):
            with lock:
                pauses.append(round(d, 3))
                t[0] += d

        summary = self.run_(range(4), lambda item, name: None, sleep=sleeper, clock=clock)
        self.assertEqual(summary["processed"], 4)
        self.assertEqual(sorted(pauses), [10.0, 10.0, 10.0])   # 4 starts = 3 waits of 10 s (6 requests per minute)

    def test_intervals_argument_overrides_the_catalog(self):
        (a,) = providers(a=1)
        a["min_interval_s"] = 10.0
        self.install(a)
        t, pauses = [0.0], []
        self.run_(range(3), lambda item, name: None, intervals={"a": 2.0},
                  sleep=lambda d: (pauses.append(round(d, 3)), t.__setitem__(0, t[0] + d)), clock=lambda: t[0])
        self.assertEqual(pauses, [2.0, 2.0])

    def test_only_restricts_the_providers_used(self):
        self.install(*providers(a=2, b=2))
        who = []
        self.run_(range(10), lambda item, name: who.append(name), only=["b"])
        self.assertEqual(set(who), {"b"})

    def test_generate_with_prefer_spreads_real_calls_over_providers(self):
        """End to end: `work` calls the router with prefer=provider; each provider gets its share."""
        a, b = providers(a=2, b=2)
        for p in (a, b):                       # realistic latency: without it, the first one's 2 workers drain the queue
            p["client"].chat.completions.with_raw_response.create = (
                lambda create: (lambda **kw: (time.sleep(0.01), create(**kw))[1]))(
                    p["client"].chat.completions.with_raw_response.create)
        self.install(a, b)
        summary = self.run_(range(30), lambda item, name: r.generate(f"doc {item}", prefer=name))
        self.assertEqual(summary["processed"], 30)
        self.assertGreater(len(a["calls"]), 0)
        self.assertGreater(len(b["calls"]), 0)
        self.assertEqual(len(a["calls"]) + len(b["calls"]), 30)


class ParseLimitsTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_limits("nvidia=10,groq=3"), {"nvidia": 10, "groq": 3})
        self.assertEqual(parse_limits(" Nvidia=8 "), {"nvidia": 8})
        self.assertEqual((parse_limits(None), parse_limits("")), ({}, {}))

    def test_a_malformed_setting_is_refused_instead_of_ignored(self):
        for bad in ("nvidia", "nvidia=", "nvidia=abc", "=5", "nvidia=-2", "nvidia=0", "nvidia=31"):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    parse_limits(bad)


if __name__ == "__main__":
    unittest.main()
