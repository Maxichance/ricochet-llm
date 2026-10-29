"""Catalogue des providers : validation et surcharge par providers.json."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet import catalog  # noqa: E402
from ricochet.catalog import ConfigError, load_specs  # noqa: E402


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        env = {k: v for k, v in os.environ.items() if k != catalog.CONFIG_ENV_VAR}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        cwd = mock.patch.object(Path, "cwd", return_value=Path(self.dir.name))   # pas de providers.json ambiant
        cwd.start()
        self.addCleanup(cwd.stop)

    def config(self, payload) -> str:
        path = Path(self.dir.name) / "custom.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        return str(path)

    def names(self, **kw):
        return [s["name"] for s in load_specs(**kw)]

    def test_builtin_catalog_is_valid_and_complete(self):
        specs = load_specs()
        self.assertEqual([s["name"] for s in specs][:2], ["groq", "nvidia"])
        for s in specs:
            for key in ("label", "model_light", "quota_min", "supports_tools", "timeout", "quota_labels"):
                self.assertIn(key, s, s["name"])
            self.assertTrue(s["base_url"].startswith("https://"), s["name"])
        self.assertEqual(len({s["name"] for s in specs}), len(specs))

    def test_light_model_defaults_to_heavy_and_label_to_name(self):
        path = self.config({"providers": [{"name": "x", "env_key": "X_KEY", "base_url": "https://x.io/v1", "model_heavy": "m"}]})
        spec = [s for s in load_specs(path) if s["name"] == "x"][0]
        self.assertEqual((spec["model_light"], spec["label"], spec["timeout"], spec["supports_tools"]), ("m", "x", 60, True))

    def test_override_existing_provider_keeps_other_fields(self):
        base = [s for s in load_specs() if s["name"] == "groq"][0]
        path = self.config({"providers": [{"name": "groq", "model_heavy": "autre", "quota_min": 10}]})
        spec = [s for s in load_specs(path) if s["name"] == "groq"][0]
        self.assertEqual((spec["model_heavy"], spec["quota_min"], spec["base_url"]), ("autre", 10, base["base_url"]))

    def test_disable_add_and_order(self):
        path = self.config({
            "providers": [{"name": "cerebras", "enabled": False},
                          {"name": "nouveau", "env_key": "N_KEY", "base_url": "http://localhost:1234/v1", "model_heavy": "m"}],
            "order": ["nouveau", "mistral"],
        })
        names = self.names(path=path)
        self.assertNotIn("cerebras", names)
        self.assertEqual(names[:3], ["nouveau", "mistral", "groq"])

    def test_disabling_unknown_provider_is_ignored(self):
        path = self.config({"providers": [{"name": "inconnu", "enabled": False}]})
        self.assertEqual(self.names(path=path), self.names())

    def test_cwd_file_and_env_var_are_discovered(self):
        (Path(self.dir.name) / catalog.CONFIG_FILENAME).write_text(
            json.dumps({"providers": [{"name": "groq", "enabled": False}]}), encoding="utf-8")
        self.assertNotIn("groq", self.names())
        other = self.config({"providers": [{"name": "nvidia", "enabled": False}]})
        os.environ[catalog.CONFIG_ENV_VAR] = other                    # la variable prime sur le fichier du dossier
        names = self.names()
        self.assertIn("groq", names)
        self.assertNotIn("nvidia", names)

    def test_errors_are_explicit(self):
        cases = {
            "inconnu(s)": {"providers": [{"name": "groq", "model_heavyy": "x"}]},
            "obligatoire": {"providers": [{"name": "new", "env_key": "K"}]},
            "http": {"providers": [{"name": "new", "env_key": "K", "base_url": "ftp://x", "model_heavy": "m"}]},
            "minuscules": {"providers": [{"name": "Nouveau", "env_key": "K", "base_url": "https://x", "model_heavy": "m"}]},
            "quota_min": {"providers": [{"name": "groq", "quota_min": -1}]},
            "supports_tools": {"providers": [{"name": "groq", "supports_tools": "oui"}]},
            "timeout": {"providers": [{"name": "groq", "timeout": 0}]},
            "quota_labels": {"providers": [{"name": "groq", "quota_labels": ["a"]}]},
            "order": {"order": ["fantome"]},
            "name": {"providers": [{"label": "sans nom"}]},
            "providers": {"providers": "pas une liste"},
        }
        for fragment, payload in cases.items():
            with self.subTest(fragment):
                with self.assertRaises(ConfigError) as ctx:
                    load_specs(self.config(payload))
                self.assertIn(fragment, str(ctx.exception))

    def test_bad_json_and_missing_file(self):
        with self.assertRaises(ConfigError):
            load_specs(self.config("{pas du json"))
        with self.assertRaises(ConfigError):
            load_specs(str(Path(self.dir.name) / "absent.json"))


class MeasuredSettingsTests(CatalogTests):
    """Réglages du catalogue : un futur changement ne doit pas passer inaperçu."""

    def spec(self, name):
        return next(s for s in load_specs() if s["name"] == name)

    def models_of(self, spec):
        return {spec["model_heavy"], spec["model_light"], *spec["extra_models"]}

    def test_nemotron_models_that_reason_have_thinking_disabled(self):
        """Sans cela le modèle dépense tout max_tokens à raisonner et rend un `content` vide."""
        nv = self.spec("nvidia")
        off = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        for model in ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-super-vl-120b-a12b"):
            self.assertEqual(nv["model_options"][model], off, model)
            self.assertIn(model, self.models_of(nv))

    def test_nvidia_order_and_the_model_that_returned_empty_json_is_gone(self):
        nv = self.spec("nvidia")
        self.assertEqual(nv["model_heavy"], "nvidia/nemotron-3-ultra-550b-a55b")        # le plus capable en premier
        self.assertEqual(nv["extra_models"][0], "nvidia/nemotron-3.5-super-vl-120b-a12b")
        self.assertNotIn("nvidia/nemotron-3.5-lightning-30b-a3b", self.models_of(nv))   # JSON valide mais VIDE

    def test_nvidia_runs_six_requests_at_once_below_forty_per_minute(self):
        nv = self.spec("nvidia")
        self.assertEqual(nv["max_concurrency"], 6)
        self.assertLessEqual(60 / nv["min_interval_s"], 40)         # 2 s entre départs = 30 par minute au plus

    def test_groq_token_limits_cover_each_of_its_models(self):
        groq = self.spec("groq")
        self.assertEqual(groq["token_limits"], {"openai/gpt-oss-120b": 8000, "qwen/qwen3.8-27b": 7000, "openai/gpt-oss-20b": 8000})
        self.assertTrue(set(groq["token_limits"]) <= self.models_of(groq))

    def test_gonka_is_throttled_to_its_six_requests_per_minute(self):
        gonka = self.spec("gonka")
        self.assertEqual(gonka["max_concurrency"], 1)
        self.assertGreaterEqual(gonka["min_interval_s"], 10)        # 6 requêtes par minute = un départ toutes les 10 s

    def test_cloudflare_needs_the_account_id_and_accepts_known_aliases(self):
        cf = self.spec("cloudflare")
        self.assertEqual(cf["env_required"], ["CLOUDFLARE_ACCOUNT_ID"])
        self.assertIn("{CLOUDFLARE_ACCOUNT_ID}", cf["base_url"])
        self.assertIn("CLOUDFLAIRE_ACCOUNT_ID", cf["env_aliases"]["CLOUDFLARE_ACCOUNT_ID"])   # faute de frappe vue dans un vrai .env

    def test_options_and_limits_only_name_models_of_their_provider(self):
        for spec in load_specs():
            models = self.models_of(spec)
            self.assertTrue(set(spec["model_options"]) <= models, spec["name"])
            self.assertTrue(set(spec["token_limits"]) <= models, spec["name"])
            self.assertNotIn(spec["model_heavy"], spec["extra_models"], spec["name"])
            self.assertEqual(len(spec["extra_models"]), len(set(spec["extra_models"])), spec["name"])

    def test_new_fields_are_validated(self):
        base = {"name": "x", "env_key": "K", "base_url": "https://x.io/v1", "model_heavy": "m"}
        cases = {
            "extra_models": {"extra_models": "pas une liste"},
            "model_options": {"model_options": {"m": "pas un objet"}},
            "réglage(s) inconnu(s)": {"model_options": {"m": {"temperature": 1}}},
            "extra_body": {"model_options": {"m": {"extra_body": "texte"}}},
            "token_limits": {"token_limits": {"m": -5}},
            "max_concurrency": {"max_concurrency": 0},
            "min_interval_s": {"min_interval_s": -1},
            "env_required": {"env_required": "X"},
            "env_aliases": {"env_aliases": {"K": "pas une liste"}},
        }
        for fragment, extra in cases.items():
            with self.subTest(fragment):
                with self.assertRaises(ConfigError) as ctx:
                    load_specs(self.config({"providers": [{**base, **extra}]}))
                self.assertIn(fragment, str(ctx.exception))

    def test_a_providers_json_can_set_thinking_off_for_a_new_model(self):
        path = self.config({"providers": [{
            "name": "monnvidia", "env_key": "K", "base_url": "https://x.io/v1", "model_heavy": "gros",
            "model_options": {"gros": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}},
            "token_limits": {"gros": 8000}, "max_concurrency": 4, "min_interval_s": 1.5}]})
        spec = next(s for s in load_specs(path) if s["name"] == "monnvidia")
        self.assertEqual((spec["max_concurrency"], spec["min_interval_s"], spec["token_limits"]), (4, 1.5, {"gros": 8000}))
        self.assertFalse(spec["model_options"]["gros"]["extra_body"]["chat_template_kwargs"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
