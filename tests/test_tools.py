"""Tool registry and plugin loading."""
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import List, Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet_llm import plugins  # noqa: E402
from ricochet_llm.tools import ToolRegistry  # noqa: E402


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

    def schema_of(self, func):
        self.reg.register(func)
        return self.reg.schemas()[0]["function"]

    def test_schema_is_inferred_from_signature_and_docstring(self):
        def search(email: str, limit: int = 5, active: bool = True, tags: Optional[List[str]] = None):
            """Looks up a customer.

            Args:
                email: Customer email
                limit (int): Max number of results,
                    over several lines
            """
        fn = self.schema_of(search)
        self.assertEqual(fn["name"], "search")
        self.assertEqual(fn["description"], "Looks up a customer.")
        params = fn["parameters"]
        self.assertEqual(params["required"], ["email"])
        self.assertEqual(params["properties"]["email"], {"type": "string", "description": "Customer email"})
        self.assertEqual(params["properties"]["limit"]["type"], "integer")
        self.assertIn("over several lines", params["properties"]["limit"]["description"])
        self.assertEqual(params["properties"]["active"]["type"], "boolean")
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
        @self.reg.register(name="other", description="A description.")
        def f():
            return 1
        self.assertEqual(self.reg.names(), ["other"])

    def test_invalid_name_duplicate_and_missing_description_are_rejected(self):
        with self.assertRaises(ValueError):
            self.reg.register(lambda: 1, name="invalid name!", description="d")
        self.reg.register(lambda: 1, name="ok", description="d")
        with self.assertRaises(ValueError):
            self.reg.register(lambda: 1, name="ok", description="d")
        self.reg.register(lambda: 2, name="ok", description="d", replace=True)
        with self.assertRaises(ValueError):
            self.reg.register(lambda: 1, name="no_doc")


class ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

        @self.reg.register
        def add(a: int, b: int = 0):
            """Adds."""
            return {"sum": a + b}

        @self.reg.register
        def broken():
            """Crashes."""
            raise RuntimeError("boom")

    def test_success_returns_json_string(self):
        self.assertEqual(json.loads(self.reg.execute("add", '{"a": 2, "b": 3}')), {"sum": 5})

    def test_empty_arguments_are_an_empty_object(self):
        self.assertIn("TypeError", self.reg.execute("add", ""))      # a is missing: error returned, not raised

    def test_errors_are_returned_to_the_model_not_raised(self):
        self.assertIn("Unknown tool", self.reg.execute("nope", "{}"))
        self.assertIn("TypeError", self.reg.execute("add", '{"zzz": 1}'))
        self.assertIn("RuntimeError: boom", self.reg.execute("broken", "{}"))
        self.assertIn("error", self.reg.execute("add", "{not json"))
        self.assertIn("error", self.reg.execute("add", "[1, 2]"))

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
        path = self.write("my_plugin.py", f"""
            from pathlib import Path
            p = Path({str(marker)!r})
            p.write_text(p.read_text() + "x" if p.exists() else "x")
        """)
        self.assertEqual(plugins.load_plugin(path), path)
        self.assertIsNone(plugins.load_plugin(path))                  # already loaded
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
        broken = self.write("broken.py", "raise ValueError('oops')\n")
        with self.assertRaises(plugins.PluginError) as ctx:
            plugins.load_plugin(broken)
        self.assertIn("oops", str(ctx.exception))
        with self.assertRaises(plugins.PluginError):
            plugins.load_plugin("module_that_does_not_exist_xyz")


if __name__ == "__main__":
    unittest.main()
