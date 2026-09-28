import re
from decimal import Decimal
from unicodedata import lookup

# A whole part (optional before a decimal point, as in ".5"), a fraction, and a scale word.
_NUMBER = re.compile(
    r"(?<!\w)(\d{1,3}(?:,\d{3})+|\d+|(?=\.\d))(\.\d+)?(?:\s*(k|thousand|m|million))?\b",
    re.IGNORECASE,
)
_SCALE = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}
_TYPOGRAPHY = str.maketrans(
    {
        lookup("LEFT SINGLE QUOTATION MARK"): "'",
        lookup("RIGHT SINGLE QUOTATION MARK"): "'",
        lookup("LEFT DOUBLE QUOTATION MARK"): '"',
        lookup("RIGHT DOUBLE QUOTATION MARK"): '"',
        lookup("EN DASH"): "-",
        lookup("EM DASH"): "-",
    }
)


_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def numbers_in(text: str) -> set[Decimal]:
    """Every number in the text, without thousands separators and with k/m suffixes applied."""
    return {_value(match) for match in _NUMBER.finditer(text)}


def unsupported_numbers(text: str, allowed: set[Decimal]) -> list[str]:
    """The numbers in the text, as written, whose values are not allowed. List markers such as
    "1." at the start of a line are not numbers."""
    written = _NUMBER.finditer(_LIST_MARKER.sub("", text))
    return list(dict.fromkeys(m.group(0) for m in written if _value(m) not in allowed))


def _value(match: re.Match[str]) -> Decimal:
    whole, fraction, scale = match.groups(default="")
    return Decimal(whole.replace(",", "") + fraction) * _SCALE.get(scale.lower(), 1)


def normalise(text: str) -> str:
    """Case, spacing and curly punctuation do not count when comparing a quote to its source."""
    return " ".join(text.translate(_TYPOGRAPHY).split()).casefold()
