"""
Obvious personal data must not go to a free provider.

Each pattern is validated by its check key (Luhn, modulo 97) so as not to block an EAN or a product reference.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet_llm import privacy  # noqa: E402
from ricochet_llm.privacy import check_conversation, find_personal_data  # noqa: E402


def _security_key(base13: str) -> str:
    return f"{97 - int(base13) % 97:02d}"


SOCIAL = "1 85 05 78 006 084 " + _security_key("1850578006084")


class DetectionTests(unittest.TestCase):
    def test_obvious_personal_data_is_found(self):
        cases = [
            ("Send the result to jean.dupont@example.fr", [privacy.EMAIL]),
            ("Call me back on 06 12 34 56 78", [privacy.PHONE]),
            ("My number: +33 6 12 34 56 78", [privacy.PHONE]),
            ("Transfer to FR76 3000 6000 0112 3456 7890 189", [privacy.IBAN]),
            ("card 4111 1111 1111 1111 expires 12/28", [privacy.CARD]),
            ("card 4111111111111111", [privacy.CARD]),
            (f"social security {SOCIAL}", [privacy.SOCIAL_SECURITY]),
            ("mail a@b.fr and phone 0612345678", [privacy.EMAIL, privacy.PHONE]),
            ("carte 4111 1111 1111 1111", [privacy.CARD]),                # no-break spaces
        ]
        for text, expected in cases:
            self.assertEqual(find_personal_data(text), expected, msg=text)

    def test_an_ordinary_text_is_not_blocked(self):
        for text in (
            "What is the average price of this product?",
            "Is product 3017620422003 available?",                       # EAN-13
            "Reference 1850578006084 at this supplier",                                    # 13 digits starting with 1: an EAN
            "How many products between 10 and 20 euros, i.e. 1 234 567 items?",
            "The first 4111 1111 1111 1112?",                                   # 16 digits but wrong Luhn key
            "Price 12,50 € and 30 % discount",
            "FR76 is not an IBAN",
            "report of 26/10/2026 at 13:33",
            "",
            None,
        ):
            self.assertEqual(find_personal_data(text), [], msg=repr(text))


class ConversationTests(unittest.TestCase):
    def test_a_clean_message_with_a_clean_history_may_leave(self):
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        self.assertEqual(check_conversation("what price?", history), [])

    def test_a_replayed_user_message_blocks_the_following_turns(self):
        history = [{"role": "user", "content": "my email is a@b.fr"}, {"role": "assistant", "content": "ok"}]
        self.assertEqual(check_conversation("and now?", history), [privacy.EMAIL])

    def test_the_models_own_replies_do_not_count(self):
        history = [{"role": "assistant", "content": "write to contact@example.fr"}]
        self.assertEqual(check_conversation("thanks", history), [])

    def test_only_the_last_turns_are_replayed(self):
        history = [{"role": "user", "content": "a@b.fr"}] + [{"role": "user", "content": "ok"}] * 6
        self.assertEqual(check_conversation("next", history, turns=6), [])


if __name__ == "__main__":
    unittest.main()
