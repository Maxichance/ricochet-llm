"""
Repère les données personnelles ÉVIDENTES dans ce qu'un utilisateur saisit, avant de l'envoyer à un provider.

Les providers gratuits sont hors de l'Europe pour la plupart, et plusieurs peuvent journaliser ou réutiliser les échanges
(voir la colonne « note » du catalogue). Demander aux utilisateurs de ne rien saisir de personnel ne les en empêche pas :
ce module est le filet. Adresse e-mail, téléphone (France), IBAN, carte bancaire, numéro de sécurité sociale.

    from ricochet import find_personal_data, chat_detailed

    found = find_personal_data(message)
    if found:
        return f"Cette question semble contenir une donnée personnelle ({', '.join(found)}). Reformulez-la."
    chat_detailed(history, message)

Chaque motif est validé par sa clé de contrôle (Luhn pour la carte, modulo 97 pour l'IBAN et la sécurité sociale) pour ne
pas bloquer un code-barres, une référence produit ou un numéro de commande. Limite assumée : un nom, une situation
confidentielle ou un contexte personnel écrits en texte libre ne se repèrent pas. Fonctions pures, aucun appel réseau.
"""
import re
from typing import List, Optional

EMAIL = "adresse e-mail"
PHONE = "numéro de téléphone"
IBAN = "IBAN"
CARD = "numéro de carte bancaire"
SOCIAL_SECURITY = "numéro de sécurité sociale"

# Les classes « [ <nbsp><espace fine insécable>-] » acceptent les séparateurs que le français met dans les nombres.
_RE_EMAIL = re.compile(r"[\w.+%\-]+@[\w\-]+(?:\.[\w\-]+)*\.[A-Za-z]{2,}", re.UNICODE)
_RE_PHONE = re.compile(r"(?<![\w+])(?:0[1-9]|(?:\+33|0033)[ . -]?[1-9])(?:[ . -]?\d{2}){4}(?!\w)")
_RE_IBAN = re.compile(r"\b[A-Za-z]{2}\d{2}(?:[ \xa0 -]?[A-Za-z0-9]){11,30}\b")
_RE_DIGITS = re.compile(r"(?<![\d.,])(?:\d[ \xa0 -]?){14,18}\d(?![\d.,]|[ \xa0 -]?\d)")
_RE_SOCIAL = re.compile(
    r"(?<!\d)[12][ \xa0 .]?\d{2}[ \xa0 .]?(?:0[1-9]|1[0-2]|20)[ \xa0 .]?(?:\d{2}|2[AB])[ \xa0 .]?\d{3}"
    r"[ \xa0 .]?\d{3}(?:[ \xa0 .]?\d{2})?(?!\d)", re.IGNORECASE)


def _luhn(digits: str) -> bool:
    total = 0
    for rank, c in enumerate(reversed(digits)):
        n = int(c)
        if rank % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _valid_iban(text: str) -> bool:
    compact = re.sub(r"[ \xa0 -]", "", text).upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def _valid_social_security_key(text: str) -> bool:
    compact = re.sub(r"[ \xa0 .]", "", text).upper()
    if len(compact) != 15:
        return False    # 13 chiffres sans clé : indiscernable d'un code-barres EAN, on ne bloque pas
    base, key = compact[:13].replace("2A", "19").replace("2B", "18"), compact[13:]
    return base.isdigit() and key.isdigit() and (97 - int(base) % 97) == int(key)


def find_personal_data(text: Optional[str]) -> List[str]:
    """Types de données personnelles évidentes, dans un ordre fixe ; liste vide si rien."""
    if not text:
        return []
    found: List[str] = []
    if _RE_EMAIL.search(text):
        found.append(EMAIL)
    if _RE_PHONE.search(text):
        found.append(PHONE)
    if any(_valid_iban(m.group()) for m in _RE_IBAN.finditer(text)):
        found.append(IBAN)
    if any(_valid_social_security_key(m.group()) for m in _RE_SOCIAL.finditer(text)):
        found.append(SOCIAL_SECURITY)
    for m in _RE_DIGITS.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 15 <= len(digits) <= 19 and _luhn(digits):
            found.append(CARD)
            break
    return found


def check_conversation(message: str, history: Optional[List[dict]] = None, turns: int = 6) -> List[str]:
    """Types trouvés dans le message OU dans les `turns` derniers messages de l'utilisateur de l'historique rejoué
    (la réponse du modèle ne compte pas : un historique envoyé à chaque tour ferait repartir la donnée). Liste vide si
    rien : le message peut partir."""
    found = find_personal_data(message)
    if found:
        return found
    for entry in (history or [])[-turns:]:
        if entry.get("role") == "user":
            found = find_personal_data(entry.get("content"))
            if found:
                return found
    return []
