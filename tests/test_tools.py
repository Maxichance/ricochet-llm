"""Registre d'outils et chargement de plugins."""
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import List, Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet import plugins  # noqa: E402
from ricochet.tools import ToolRegistry  # noqa: E402


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

    def schema_of(self, func):
        self.reg.register(func)
        return self.reg.schemas()[0]["function"]

    def test_schema_is_inferred_from_signature_and_docstring(self):
        def chercher(email: str, limite: int = 5, actif: bool = True, tags: Optional[List[str]] = None):
            """Cherche un client.

            Args:
                email: Email du client
                limite (int): Nombre max de résultats,
                    sur plusieurs lignes
            """
        fn = self.schema_of(chercher)
        self.assertEqual(fn["name"], "chercher")
        self.assertEqual(fn["description"], "Cherche un client.")
        params = fn["parameters"]
        self.assertEqual(params["required"], ["email"])
        self.assertEqual(params["properties"]["email"], {"type": "string", "description": "Email du client"})
        self.assertEqual(params["properties"]["limite"]["type"], "integer")
        self.assertIn("sur plusieurs lignes", params["properties"]["limite"]["description"])
        self.assertEqual(params["properties"]["actif"]["type"], "boolean")
        self.assertEqual(params["properties"]["tags"], {"type": "array", "items": {"type": "string"}})

    def test_literal_becomes_enum_and_varargs_are_ignored(self):
        def f(mode: Literal["a", "b"], *args, **kwargs):
            """Doc."""
        props = self.schema_of(f)["parameters"]["properties"]
        self.assertEqual(props, {"mode": {"type": "string", "enum": ["a", "b"]}})

    def test_no_parameters_gives_empty_object_schema(self):
        def f():
            """Doc."""
        self.assertEqual(self.schema_of(f)["parameters"], {"type": "object", "properties": {}})

    def test_explicit_parameters_and_name_override_inference(self):
        custom = {"type": "object", "properties": {"x": {"type": "number"}}}
        self.reg.register(lambda x: x, name="double", description="Double.", parameters=custom)
        fn = self.reg.schemas()[0]["function"]
        self.assertEqual((fn["name"], fn["parameters"]), ("double", custom))


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

    def test_decorator_returns_the_function_unchanged(self):
        @self.reg.register
        def f():
            """Doc."""
            return 1
        self.assertEqual(f(), 1)
        self.assertIn("f", self.reg)

    def test_decorator_with_arguments(self):
        @self.reg.register(name="autre", description="Une description.")
        def f():
            return 1
        self.assertEqual(self.reg.names(), ["autre"])

    def test_invalid_name_duplicate_and_missing_description_are_rejected(self):
        with self.assertRaises(ValueError):
            self.reg.register(lambda: 1, name="nom invalide!", description="d")
        self.reg.register(lambda: 1, name="ok", description="d")
        with self.assertRaises(ValueError):
            self.reg.register(lambda: 1, name="ok", description="d")
        self.reg.register(lambda: 2, name="ok", description="d", replace=True)
        with self.assertRaises(ValueError):
            self.reg.register(lambda: 1, name="sans_doc")


class ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

        @self.reg.register
        def additionner(a: int, b: int = 0):
            """Additionne."""
            return {"somme": a + b}

        @self.reg.register
        def casse():
            """Plante."""
            raise RuntimeError("boum")

    def test_success_returns_json_string(self):
        self.assertEqual(json.loads(self.reg.execute("additionner", '{"a": 2, "b": 3}')), {"somme": 5})

    def test_empty_arguments_are_an_empty_object(self):
        self.assertIn("TypeError", self.reg.execute("additionner", ""))      # a manquant : erreur renvoyée, pas levée

    def test_errors_are_returned_to_the_model_not_raised(self):
        self.assertIn("Outil inconnu", self.reg.execute("nope", "{}"))
        self.assertIn("TypeError", self.reg.execute("additionner", '{"zzz": 1}'))
        self.assertIn("RuntimeError: boum", self.reg.execute("casse", "{}"))
        self.assertIn("error", self.reg.execute("additionner", "{pas du json"))
        self.assertIn("error", self.reg.execute("additionner", "[1, 2]"))

    def test_non_json_results_are_stringified(self):
        self.reg.register(lambda: {1, 2}.__class__, name="t", description="d")
        self.assertIsInstance(json.loads(self.reg.execute("t", "{}")), str)


class PluginTests(unittest.TestCase):
    def setUp(self):
        plugins._loaded.clear()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(plugins._loaded.clear)

    def write(self, name, body):
        path = Path(self.dir.name) / name
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return str(path)

    def test_no_target_loads_nothing(self):
        old = os.environ.pop("RICOCHET_PLUGIN", None)
        try:
            self.assertIsNone(plugins.load_plugin())
        finally:
            if old is not None:
                os.environ["RICOCHET_PLUGIN"] = old

    def test_python_file_is_executed_once(self):
        marker = Path(self.dir.name) / "count.txt"
        path = self.write("mon_plugin.py", f"""
            from pathlib import Path
            p = Path({str(marker)!r})
            p.write_text(p.read_text() + "x" if p.exists() else "x")
        """)
        self.assertEqual(plugins.load_plugin(path), path)
        self.assertIsNone(plugins.load_plugin(path))                  # déjà chargé
        self.assertEqual(marker.read_text(), "x")

    def test_env_var_is_used(self):
        path = self.write("p_env.py", "LOADED = True\n")
        os.environ["RICOCHET_PLUGIN"] = path
        try:
            self.assertEqual(plugins.load_plugin(), path)
        finally:
            del os.environ["RICOCHET_PLUGIN"]

    def test_importable_module_name(self):
        self.write("module_plugin_test.py", "VALUE = 1\n")
        sys.path.insert(0, self.dir.name)
        self.addCleanup(sys.path.remove, self.dir.name)
        self.addCleanup(sys.modules.pop, "module_plugin_test", None)
        self.assertEqual(plugins.load_plugin("module_plugin_test"), "module_plugin_test")

    def test_missing_file_and_broken_plugin_raise_plugin_error(self):
        with self.assertRaises(plugins.PluginError):
            plugins.load_plugin(str(Path(self.dir.name) / "absent.py"))
        broken = self.write("casse.py", "raise ValueError('oups')\n")
        with self.assertRaises(plugins.PluginError) as ctx:
            plugins.load_plugin(broken)
        self.assertIn("oups", str(ctx.exception))
        with self.assertRaises(plugins.PluginError):
            plugins.load_plugin("module_qui_nexiste_pas_xyz")


if __name__ == "__main__":
    unittest.main()
