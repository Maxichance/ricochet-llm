"""
Exécuteur parallèle : répartition entre providers, plafonds, pauses, arrêt propre.

Une limite « 40 requêtes par minute » porte sur le rythme de DÉPART, pas sur la simultanéité : à 20 s la réponse, 2
travailleurs font ~5 requêtes par minute, il en faut ~13 pour atteindre 40. Aucun appel réseau.
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

from ricochet import router as r  # noqa: E402
from ricochet.parallel import Interrupted, parse_limits, run_parallel  # noqa: E402


def providers(**conc):
    """Providers de test : `a=3` = trois requêtes simultanées au plus ; aucun écart entre départs."""
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

        bilan = self.run_(range(60), work)
        self.assertEqual(sorted(i for i, _ in seen), list(range(60)))
        self.assertEqual((bilan["processed"], bilan["remaining"], bilan["interrupted"], bilan["errors"]), (60, 0, False, 0))
        self.assertEqual(set(bilan["per_provider"]), {"a", "b", "c"})
        self.assertEqual(sum(bilan["per_provider"].values()), 60)

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
        self.run_(range(60), work, limits={"a": 5})                   # réglable sans toucher au catalogue
        self.assertTrue(3 <= peak["a"] <= 5, peak)

    def test_a_paused_provider_takes_no_work(self):
        self.install(*providers(a=2, b=2))
        r._put_in_cooldown("a", 3600)
        who = []
        bilan = self.run_(range(20), lambda item, name: who.append(name))
        self.assertEqual(set(who), {"b"})
        self.assertEqual((bilan["processed"], bilan["remaining"]), (20, 0))

    def test_a_paused_provider_does_not_hold_the_end_of_the_run(self):
        """Un provider en pause une heure ne doit pas retenir la fin du passage quand tout est déjà traité."""
        self.install(*providers(a=2, b=2))
        r._put_in_cooldown("a", 3600)
        debut = time.monotonic()
        self.run_(range(6), lambda item, name: None)
        self.assertLess(time.monotonic() - debut, 2)

    def test_no_provider_available_stops_cleanly_and_returns_the_items(self):
        self.install(*providers(a=2))
        r._put_in_cooldown("a", 3600)
        called = []
        bilan = self.run_(range(5), lambda item, name: called.append(item))
        self.assertEqual((called, bilan["remaining"], bilan["interrupted"], bilan["processed"]), ([], 5, True, 0))

    def test_interrupted_from_work_stops_everything_and_gives_the_item_back(self):
        self.install(*providers(a=2, b=2))
        done, lock = [], threading.Lock()

        def work(item, name):
            if item == 5:
                raise Interrupted("quota")
            time.sleep(0.003)
            with lock:
                done.append(item)

        bilan = self.run_(range(40), work)
        self.assertTrue(bilan["interrupted"])
        self.assertEqual(bilan["processed"] + bilan["remaining"], 40)    # l'élément interrompu est rendu à la file
        self.assertNotIn(5, done)

    def test_router_error_inside_work_counts_as_all_providers_down(self):
        self.install(*providers(a=1))

        def work(item, name):
            raise r.LLMRouterError("Aucun provider disponible.")

        bilan = self.run_(range(4), work)
        self.assertEqual((bilan["interrupted"], bilan["processed"], bilan["remaining"]), (True, 0, 4))

    def test_an_error_on_one_item_does_not_stop_the_others(self):
        self.install(*providers(a=2, b=2))

        def work(item, name):
            if item % 4 == 0:
                raise ValueError("réponse bizarre")

        bilan = self.run_(range(12), work)
        self.assertEqual((bilan["errors"], bilan["processed"], bilan["remaining"], bilan["interrupted"]), (3, 9, 0, False))

    def test_departures_of_one_provider_are_spaced_by_min_interval(self):
        (a,) = providers(a=3)                                  # trois travailleurs, mais UN départ à la fois
        a["min_interval_s"] = 10.0
        self.install(a)
        t, pauses, lock = [0.0], [], threading.Lock()

        def clock():
            return t[0]

        def sleeper(d):
            with lock:
                pauses.append(round(d, 3))
                t[0] += d

        bilan = self.run_(range(4), lambda item, name: None, sleep=sleeper, clock=clock)
        self.assertEqual(bilan["processed"], 4)
        self.assertEqual(sorted(pauses), [10.0, 10.0, 10.0])   # 4 départs = 3 attentes de 10 s (6 requêtes par minute)

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
        """Bout en bout : `work` appelle le routeur avec prefer=provider ; chaque provider reçoit sa part."""
        a, b = providers(a=2, b=2)
        for p in (a, b):                       # une latence réaliste : sans elle, les 2 travailleurs du premier vident la file
            p["client"].chat.completions.with_raw_response.create = (
                lambda create: (lambda **kw: (time.sleep(0.01), create(**kw))[1]))(
                    p["client"].chat.completions.with_raw_response.create)
        self.install(a, b)
        bilan = self.run_(range(30), lambda item, name: r.generate(f"doc {item}", prefer=name))
        self.assertEqual(bilan["processed"], 30)
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
