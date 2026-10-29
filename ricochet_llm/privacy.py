"""
Spots OBVIOUS personal data in what a user types, before it is sent to a provider.

Most free providers are outside Europe, and several may log or reuse the exchanges (see the "note" field of the
catalog). Asking users not to type anything personal does not stop them: this module is the safety net. Email address,
phone number (France), IBAN, bank card, social security number (France).

    from ricochet_llm import find_personal_data, chat_detailed

    found = find_personal_data(message)
    if found:
        return f"This question seems to contain personal data ({', '.join(found)}). Please rephrase it."
    chat_detailed(history, message)

Each pattern is validated by its check key (Luhn for the card, modulo 97 for the IBAN and the social security number) so
as not to block a barcode, a product reference or an order number. Accepted limit: a name, a confidential situation or
a personal context written in free text is not detected. Pure functions, no network calls.
"""
import re
from typing import List, Optional

EMAIL = "email address"
PHONE = "phone number"
IBAN = "IBAN"
CARD = "bank card number"
SOCIAL_SECURITY = "social security number"

# The classes "[ <nbsp><narrow no-break space>-]" accept the separators French puts in numbers.
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
        return False    # 13 digits without a key: indistinguishable from an EAN barcode, we do not block
    base, key = compact[:13].replace("2A", "19").replace("2B", "18"), compact[13:]
    return base.isdigit() and key.isdigit() and (97 - int(base) % 97) == int(key)


def find_personal_data(text: Optional[str]) -> List[str]:
    """Types of obvious personal data, in a fixed order; empty list if none."""
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
    """Types found in the message OR in the last `turns` user messages of the replayed history
    (the model's reply does not count: a history sent on every turn would send the data out again). Empty list if
    none: the message may go out."""
    found = find_personal_data(message)
    if found:
        return found
    for entry in (history or [])[-turns:]:
        if entry.get("role") == "user":
            found = find_personal_data(entry.get("content"))
            if found:
                return found
    return []
