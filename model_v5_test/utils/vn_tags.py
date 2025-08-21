"""
Vietnamese decomposition/compose helper.

Functions:
- decompose_str(s) -> (base_str, mod_ids, tone_ids)
- compose_str(base_str, mod_preds, tone_preds) -> composed_str

Notes:
- Modifiers: none, breve, circ, horn
- Tones: level, acute, grave, hook, tilde, dot
- Treat đ/Đ as base letters (no tags)
- Handles both lowercase and uppercase
"""
from typing import List, Tuple

# Tag id mapping
MODIFIERS = ["none", "breve", "circ", "horn"]
TONES = ["level", "acute", "grave", "hook", "tilde", "dot"]
MOD2ID = {m: i for i, m in enumerate(MODIFIERS)}
TONE2ID = {t: i for i, t in enumerate(TONES)}

# Vietnamese base letters (as characters used without diacritics)
# We consider 'đ' and 'Đ' as distinct base letters; they carry no modifier/tone.
# For other letters, base is one of a, e, i, o, u, y, A, E, I, O, U, Y, and plain consonants.

# Build decomposition/compose maps for both cases.
# Each entry maps a (base, modifier, tone) -> composed character and reverse.

# Helper: generate tables for a, e, o (which can take breve, circumflex), and u (horn), o (horn), a (no horn), e (no horn), i, y (no modifier), etc.
# To keep it robust and explicit, we enumerate all standard precomposed Vietnamese letters in Unicode NFC.

# Data source: Unicode Vietnamese precomposed letters (not copied from any single source; compiled by enumerating standard combinations)
# The mapping below lists the composed character for key vowels across modifiers and tones.

# For readability, split by lowercase and uppercase.

# Lowercase vowels maps: dict[(base, mod, tone)] = char
LOWER_MAP = {}
UPPER_MAP = {}

# Define helper to register entries

def reg(base: str, mod: str, tone: str, lower: str, upper: str):
    key = (base, mod, tone)
    LOWER_MAP[key] = lower
    UPPER_MAP[(base.upper(), mod, tone)] = upper

# Vowels that can have circumflex: a, e, o
# Vowels that can have breve: a
# Vowels that can have horn: o, u
# Vowels i, y, u, o, e, a can have tones (except tone marks don't apply to ă/â/ê/ô/ơ/ư base w.r.t positions but are encoded as precomposed)

# We will enumerate the standard Vietnamese set from common usage.
# a
reg('a', 'none', 'level', 'a', 'A')
reg('a', 'none', 'acute', 'á', 'Á')
reg('a', 'none', 'grave', 'à', 'À')
reg('a', 'none', 'hook', 'ả', 'Ả')
reg('a', 'none', 'tilde', 'ã', 'Ã')
reg('a', 'none', 'dot', 'ạ', 'Ạ')
# a with breve (ă)
reg('a', 'breve', 'level', 'ă', 'Ă')
reg('a', 'breve', 'acute', 'ắ', 'Ắ')
reg('a', 'breve', 'grave', 'ằ', 'Ằ')
reg('a', 'breve', 'hook', 'ẳ', 'Ẳ')
reg('a', 'breve', 'tilde', 'ẵ', 'Ẵ')
reg('a', 'breve', 'dot', 'ặ', 'Ặ')
# a with circumflex (â)
reg('a', 'circ', 'level', 'â', 'Â')
reg('a', 'circ', 'acute', 'ấ', 'Ấ')
reg('a', 'circ', 'grave', 'ầ', 'Ầ')
reg('a', 'circ', 'hook', 'ẩ', 'Ẩ')
reg('a', 'circ', 'tilde', 'ẫ', 'Ẫ')
reg('a', 'circ', 'dot', 'ậ', 'Ậ')

# e and ê
reg('e', 'none', 'level', 'e', 'E')
reg('e', 'none', 'acute', 'é', 'É')
reg('e', 'none', 'grave', 'è', 'È')
reg('e', 'none', 'hook', 'ẻ', 'Ẻ')
reg('e', 'none', 'tilde', 'ẽ', 'Ẽ')
reg('e', 'none', 'dot', 'ẹ', 'Ẹ')
reg('e', 'circ', 'level', 'ê', 'Ê')
reg('e', 'circ', 'acute', 'ế', 'Ế')
reg('e', 'circ', 'grave', 'ề', 'Ề')
reg('e', 'circ', 'hook', 'ể', 'Ể')
reg('e', 'circ', 'tilde', 'ễ', 'Ễ')
reg('e', 'circ', 'dot', 'ệ', 'Ệ')

# i
reg('i', 'none', 'level', 'i', 'I')
reg('i', 'none', 'acute', 'í', 'Í')
reg('i', 'none', 'grave', 'ì', 'Ì')
reg('i', 'none', 'hook', 'ỉ', 'Ỉ')
reg('i', 'none', 'tilde', 'ĩ', 'Ĩ')
reg('i', 'none', 'dot', 'ị', 'Ị')

# o and ô and ơ (horn)
reg('o', 'none', 'level', 'o', 'O')
reg('o', 'none', 'acute', 'ó', 'Ó')
reg('o', 'none', 'grave', 'ò', 'Ò')
reg('o', 'none', 'hook', 'ỏ', 'Ỏ')
reg('o', 'none', 'tilde', 'õ', 'Õ')
reg('o', 'none', 'dot', 'ọ', 'Ọ')
reg('o', 'circ', 'level', 'ô', 'Ô')
reg('o', 'circ', 'acute', 'ố', 'Ố')
reg('o', 'circ', 'grave', 'ồ', 'Ồ')
reg('o', 'circ', 'hook', 'ổ', 'Ổ')
reg('o', 'circ', 'tilde', 'ỗ', 'Ỗ')
reg('o', 'circ', 'dot', 'ộ', 'Ộ')
reg('o', 'horn', 'level', 'ơ', 'Ơ')
reg('o', 'horn', 'acute', 'ớ', 'Ớ')
reg('o', 'horn', 'grave', 'ờ', 'Ờ')
reg('o', 'horn', 'hook', 'ở', 'Ở')
reg('o', 'horn', 'tilde', 'ỡ', 'Ỡ')
reg('o', 'horn', 'dot', 'ợ', 'Ợ')

# u and ư (horn)
reg('u', 'none', 'level', 'u', 'U')
reg('u', 'none', 'acute', 'ú', 'Ú')
reg('u', 'none', 'grave', 'ù', 'Ù')
reg('u', 'none', 'hook', 'ủ', 'Ủ')
reg('u', 'none', 'tilde', 'ũ', 'Ũ')
reg('u', 'none', 'dot', 'ụ', 'Ụ')
reg('u', 'horn', 'level', 'ư', 'Ư')
reg('u', 'horn', 'acute', 'ứ', 'Ứ')
reg('u', 'horn', 'grave', 'ừ', 'Ừ')
reg('u', 'horn', 'hook', 'ử', 'Ử')
reg('u', 'horn', 'tilde', 'ữ', 'Ữ')
reg('u', 'horn', 'dot', 'ự', 'Ự')

# y
reg('y', 'none', 'level', 'y', 'Y')
reg('y', 'none', 'acute', 'ý', 'Ý')
reg('y', 'none', 'grave', 'ỳ', 'Ỳ')
reg('y', 'none', 'hook', 'ỷ', 'Ỷ')
reg('y', 'none', 'tilde', 'ỹ', 'Ỹ')
reg('y', 'none', 'dot', 'ỵ', 'Ỵ')

# Plain consonants and vowels without modifier remain with tone 'level'
# We'll support letters: b c d đ g h k l m n p q r s t v x w f j z (w/f/j/z rarely in VN but keep as-is)
# These don't have modifier/tone; we encode as (char, 'none', 'level') and only 'level' tone.
PLAIN_CHARS_LOWER = list("bcdfghjklmnpqrstvwxz")
PLAIN_CHARS_UPPER = [c.upper() for c in PLAIN_CHARS_LOWER]
for ch in PLAIN_CHARS_LOWER:
    LOWER_MAP[(ch, 'none', 'level')] = ch
for ch in PLAIN_CHARS_UPPER:
    UPPER_MAP[(ch, 'none', 'level')] = ch

# Special: đ/Đ as base, no modifier/tone
LOWER_MAP[('đ', 'none', 'level')] = 'đ'
UPPER_MAP[('Đ', 'none', 'level')] = 'Đ'

# Build reverse maps: char -> (base, mod, tone)
REV_MAP = {}
for (base, mod, tone), ch in LOWER_MAP.items():
    REV_MAP[ch] = (base, mod, tone)
for (base, mod, tone), ch in UPPER_MAP.items():
    REV_MAP[ch] = (base, mod, tone)


def decompose_char(ch: str) -> Tuple[str, int, int]:
    """Return (base_char, mod_id, tone_id) for a single character.
    Unknown chars are treated as themselves with 'none' modifier and 'level' tone.
    """
    if ch in REV_MAP:
        base, mod, tone = REV_MAP[ch]
        return base, MOD2ID.get(mod, 0), TONE2ID.get(tone, 0)
    # Default: keep char as base
    return ch, MOD2ID['none'], TONE2ID['level']


def compose_char(base: str, mod_id: int, tone_id: int) -> str:
    mod = MODIFIERS[mod_id] if 0 <= mod_id < len(MODIFIERS) else 'none'
    tone = TONES[tone_id] if 0 <= tone_id < len(TONES) else 'level'
    # Uppercase detection: if base is uppercase, choose upper map
    is_upper = base.isupper()
    key = (base, mod, tone)
    if is_upper:
        return UPPER_MAP.get(key, base)
    else:
        return LOWER_MAP.get(key, base)


def decompose_str(s: str) -> Tuple[str, List[int], List[int]]:
    """Decompose an input string into base_str, mod_ids, tone_ids.
    - base_str: string of base characters (same length as s)
    - mod_ids: list of modifier ids per character
    - tone_ids: list of tone ids per character
    """
    base_chars: List[str] = []
    mod_ids: List[int] = []
    tone_ids: List[int] = []
    for ch in s:
        base, mid, tid = decompose_char(ch)
        base_chars.append(base)
        mod_ids.append(mid)
        tone_ids.append(tid)
    return ''.join(base_chars), mod_ids, tone_ids


def compose_str(base_str: str, mod_preds: List[int], tone_preds: List[int]) -> str:
    """Compose a string from base_str and predicted modifier/tone ids.
    Lengths must match; if lists are shorter, missing entries default to 'none'/'level'.
    """
    n = len(base_str)
    out_chars: List[str] = []
    for i in range(n):
        base = base_str[i]
        mid = mod_preds[i] if i < len(mod_preds) else MOD2ID['none']
        tid = tone_preds[i] if i < len(tone_preds) else TONE2ID['level']
        out_chars.append(compose_char(base, mid, tid))
    return ''.join(out_chars)


__all__ = [
    'MODIFIERS', 'TONES', 'MOD2ID', 'TONE2ID',
    'decompose_str', 'compose_str', 'decompose_char', 'compose_char',
    'is_vowel'
]

# Vowel helper for TER computation (base vowels + extended Vietnamese vowels)
VOWELS_BASE = set(list("aAeEiIoOuUyY"))
VOWELS_EXT = set(list("ăĂâÂêÊôÔơƠưƯ"))

def is_vowel(ch: str) -> bool:
    """Return True if base character is a Vietnamese vowel (including extended forms)."""
    return ch in VOWELS_BASE or ch in VOWELS_EXT
