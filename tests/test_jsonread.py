"""
Tolerant reading of a model's JSON response.

Cases recoverable without a new call: the model sometimes writes its response TWICE
(the complete object, a quote, then a second copy) or follows it with `"extra":[]}`. Two contained a stray `null`
between two key/value pairs. A truncated or looping response stays unreadable and must be re-read.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet_llm import read_json  # noqa: E402

OBJECT = {"items": [{"label": "value A", "quote": "value A", "max": 10}],
          "meta": {"kind": "report", "cta": "See more"}}
GOOD = json.dumps(OBJECT, ensure_ascii=False)


class RecoveredTests(unittest.TestCase):
    def test_a_reply_written_twice_gives_the_first_copy(self):
        self.assertEqual(read_json(GOOD + '"' + GOOD), OBJECT)

    def test_the_second_copy_is_ignored_even_when_cut(self):
        self.assertEqual(read_json(GOOD + '"{"items":[{"label":"Up to 50 on the sel'), OBJECT)

    def test_an_extra_field_after_the_object_is_ignored(self):
        self.assertEqual(read_json(GOOD + '"extra":[]}'), OBJECT)

    def test_a_stray_null_between_two_pairs_is_removed(self):
        raw = '{"items":[{"label":"x","channel":"site","scope":"all",null,"start":null,"max":20}],"meta":{}}'
        self.assertEqual(read_json(raw)["items"][0]["max"], 20)

    def test_the_stray_null_and_the_extra_copy_together(self):
        raw = '{"items":[{"label":"x","scope":"all",null,"max":5}],"meta":{}}"{"items":[{"label":"y"'
        self.assertEqual(read_json(raw)["items"][0]["label"], "x")


class StillUnreadableTests(unittest.TestCase):
    """We never guess: a truncated or looping response must be re-read."""

    def test_cut_loops_and_prose_stay_unreadable(self):
        for raw in (
            GOOD[:-30],                                                           # cut right in the middle
            '{"items":[{"label":"x","notes":"Valid from 17/12/2',     # unterminated string (max_tokens reached)
            '{"items":[],"meta":{"kind":"',                        # cut inside the value
            '{"items":[{"label":"a "guillemet" b"}],"meta":{}}',              # unescaped quote inside a text
            '{"meta": {' + '\n    "urgences": null,' * 200,                    # the model loops: never closed
            "I cannot read this document.",
            "",
            None,
        ):
            self.assertIsNone(read_json(raw), msg=repr(raw)[:60])


class UnchangedTests(unittest.TestCase):
    def test_a_valid_reply_is_read_identically(self):
        self.assertEqual(read_json(GOOD), OBJECT)

    def test_a_legitimate_null_is_untouched(self):
        raw = '{"items":[{"label":"x","end":null,"code":null,"max":10}],"meta":{"urgent":null}}'
        data = read_json(raw)
        self.assertIsNone(data["items"][0]["end"])
        self.assertEqual(data["items"][0]["max"], 10)

    def test_a_null_in_a_number_array_is_kept(self):
        self.assertEqual(read_json('{"items":[],"meta":{"x":[1,null,2]}}')["meta"]["x"], [1, None, 2])

    def test_a_valid_object_inside_an_array_is_read_as_before(self):
        self.assertEqual(read_json('[{"items": []}]'), {"items": []})

    def test_imperfect_replies_already_handled_stay_handled(self):
        self.assertEqual(read_json("```json\n" + GOOD + "\n```"), OBJECT)
        self.assertEqual(read_json("<think>let me think</think>" + GOOD), OBJECT)
        self.assertEqual(read_json(GOOD[:-1] + ",}"), OBJECT)                      # trailing comma
        self.assertEqual(read_json("Here is the result:\n" + GOOD + "\nHave a nice day."), OBJECT)


if __name__ == "__main__":
    unittest.main()
