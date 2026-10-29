"""
Une donnée personnelle évidente ne doit pas partir chez un provider gratuit.

Chaque motif est validé par sa clé de contrôle (Luhn, modulo 97) pour ne pas bloquer un EAN ou une référence produit.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ricochet import privacy  # noqa: E402
from ricochet.privacy import check_conversation, find_personal_data  # noqa: E402


def _security_key(base13: str) -> str:
    return f"{97 - int(base13) % 97:02d}"


SOCIAL = "1 85 05 78 006 084 " + _security_key("1850578006084")


class DetectionTests(unittest.TestCase):
    def test_obvious_personal_data_is_found(self):
        cases = [
            ("Envoie le résultat à jean.dupont@exemple.fr", [privacy.EMAIL]),
            ("Rappelez-moi au 06 12 34 56 78", [privacy.PHONE]),
            ("Mon numéro : +33 6 12 34 56 78", [privacy.PHONE]),
            ("Virement sur FR76 3000 6000 0112 3456 7890 189", [privacy.IBAN]),
            ("carte 4111 1111 1111 1111 expire 12/28", [privacy.CARD]),
            ("carte 4111111111111111", [privacy.CARD]),
            (f"sécu {SOCIAL}", [privacy.SOCIAL_SECURITY]),
            ("mail a@b.fr et tel 0612345678", [privacy.EMAIL, privacy.PHONE]),
            ("carte 4111 1111 1111 1111", [privacy.CARD]),                # espaces insécables
        ]
        for text, expected in cases:
            self.assertEqual(find_personal_data(text), expected, msg=text)

    def test_an_ordinary_text_is_not_blocked(self):
        for text in (
            "Quel est le prix moyen de ce produit ?",
            "Le produit 3017620422003 est-il disponible ?",                       # EAN-13
            "Référence 1850578006084 chez ce fournisseur",                                    # 13 chiffres commençant par 1 : un EAN
            "Combien de produits entre 10 et 20 euros, soit 1 234 567 articles ?",
            "Les 4111 1111 1111 1112 premiers ?",                                   # 16 chiffres mais clé de Luhn fausse
            "Prix 12,50 € et réduction de 30 %",
            "FR76 n'est pas un IBAN",
            "rapport du 26/10/2026 à 13:33",
            "",
            None,
        ):
            self.assertEqual(find_personal_data(text), [], msg=repr(text))


class ConversationTests(unittest.TestCase):
    def test_a_clean_message_with_a_clean_history_may_leave(self):
        history = [{"role": "user", "content": "bonjour"}, {"role": "assistant", "content": "salut"}]
        self.assertEqual(check_conversation("quel prix ?", history), [])

    def test_a_replayed_user_message_blocks_the_following_turns(self):
        history = [{"role": "user", "content": "mon mail est a@b.fr"}, {"role": "assistant", "content": "ok"}]
        self.assertEqual(check_conversation("et maintenant ?", history), [privacy.EMAIL])

    def test_the_models_own_replies_do_not_count(self):
        history = [{"role": "assistant", "content": "écrivez à contact@exemple.fr"}]
        self.assertEqual(check_conversation("merci", history), [])

    def test_only_the_last_turns_are_replayed(self):
        history = [{"role": "user", "content": "a@b.fr"}] + [{"role": "user", "content": "ok"}] * 6
        self.assertEqual(check_conversation("suite", history, turns=6), [])


if __name__ == "__main__":
    unittest.main()
