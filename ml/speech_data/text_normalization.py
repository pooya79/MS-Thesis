from __future__ import annotations

import string
import unicodedata


SKIP = set(
    list(string.ascii_letters)
    + [
        "=",  # occurs only 2x in utterance (transl.): "twenty = xx"
        "ā",  # occurs only 4x together with "š"
        "š",
        # Arabic letters
        "ة",  # TEH MARBUTA
    ]
)

DISCARD = [
    # "(laughter)" in Farsi
    "(خنده)",
    # ASCII
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    ":",
    ";",
    # Unicode punctuation?
    "–",
    "¬",
    "“",
    "”",
    "…",
    "؟",
    "،",
    "؛",
    "ـ",
    # Unicode whitespace?
    "ً",
    "ٌ",
    "َ",
    "ُ",
    "ِ",
    "ّ",
    "ْ",
    "ٔ",
    # Other
    "«",
    "»",
]

REPLACEMENTS = {
    "أ": "ا",
    "ۀ": "ە",
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ﯽ": "ی",
    "ﻮ": "و",
    "ے": "ی",
    "ﺒ": "ب",
    "ﻢ": "ﻡ",
    "٬": " ",
    "ە": "ه",
}


def normalize_persian_asr_text(text: str) -> str | None:
    if set(text) & SKIP:
        return None

    text = " ".join(w for w in text.split() if not w.startswith("#"))

    for lhs, rhs in REPLACEMENTS.items():
        text = text.replace(lhs, rhs)

    for tok in DISCARD:
        text = text.replace(tok, "")

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ء", "")
    text = remove_punctuation(text)

    return " ".join(t for t in text.split() if t)


def remove_punctuation(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return " ".join(t for t in text.split() if t)
