"""
Vietnamese tone-only decomposition/compose helper for model_v6.

Modified from model_v5_mask to only handle tones, not modifiers.
Base characters now include modifier forms (ă â ê ô ơ ư) as distinct base characters.

Functions:
- decompose_str(s) -> (base_str, tone_ids)
- compose_str(base_str, tone_preds) -> composed_str

Notes:
- Tones: level, acute, grave, hook, tilde, dot
- Base characters include: a ă â e ê i o ô ơ u ư y đ (plus uppercase)
- Each Vietnamese character with modifier+tone is mapped to its base form + tone
"""
from typing import List, Tuple

# Tone id mapping (same as model_v5_mask)
TONES = ["level", "acute", "grave", "hook", "tilde", "dot"]
TONE2ID = {t: i for i, t in enumerate(TONES)}

# Extended base characters that include modifier forms
BASE_CHARS_LOWER = list("abcdefghijklmnopqrstuvwxyzăâêôơưđ")
BASE_CHARS_UPPER = [c.upper() for c in BASE_CHARS_LOWER]

# Maps: dict[composed_char] = (base_char, tone_id)
DECOMPOSE_MAP = {}
# Maps: dict[(base_char, tone_id)] = composed_char
COMPOSE_MAP = {}

def register_tone_mapping(base_lower: str, base_upper: str, 
                         level: str, acute: str, grave: str, 
                         hook: str, tilde: str, dot: str):
    """Register tone mappings for a base character."""
    tone_chars = [level, acute, grave, hook, tilde, dot]
    
    # Lowercase mappings
    for tone_id, char in enumerate(tone_chars):
        if char:  # Some tones might not exist for certain characters
            DECOMPOSE_MAP[char] = (base_lower, tone_id)
            COMPOSE_MAP[(base_lower, tone_id)] = char
    
    # Uppercase mappings
    tone_chars_upper = [level.upper(), acute.upper(), grave.upper(), 
                       hook.upper(), tilde.upper(), dot.upper()]
    for tone_id, char in enumerate(tone_chars_upper):
        if char and char != char.lower():  # Valid uppercase char
            DECOMPOSE_MAP[char] = (base_upper, tone_id)
            COMPOSE_MAP[(base_upper, tone_id)] = char

# Register all Vietnamese tone mappings with new base characters
# a (base 'a')
register_tone_mapping('a', 'A', 'a', 'á', 'à', 'ả', 'ã', 'ạ')

# ă (base 'ă' - breve is now part of base)
register_tone_mapping('ă', 'Ă', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ')

# â (base 'â' - circumflex is now part of base)
register_tone_mapping('â', 'Â', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ')

# e (base 'e')
register_tone_mapping('e', 'E', 'e', 'é', 'è', 'ẻ', 'ẽ', 'ẹ')

# ê (base 'ê' - circumflex is now part of base)
register_tone_mapping('ê', 'Ê', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ')

# i (base 'i')
register_tone_mapping('i', 'I', 'i', 'í', 'ì', 'ỉ', 'ĩ', 'ị')

# o (base 'o')
register_tone_mapping('o', 'O', 'o', 'ó', 'ò', 'ỏ', 'õ', 'ọ')

# ô (base 'ô' - circumflex is now part of base)
register_tone_mapping('ô', 'Ô', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ')

# ơ (base 'ơ' - horn is now part of base)
register_tone_mapping('ơ', 'Ơ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ')

# u (base 'u')
register_tone_mapping('u', 'U', 'u', 'ú', 'ù', 'ủ', 'ũ', 'ụ')

# ư (base 'ư' - horn is now part of base)
register_tone_mapping('ư', 'Ư', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự')

# y (base 'y')
register_tone_mapping('y', 'Y', 'y', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ')

# đ and Đ (special cases - always level tone)
DECOMPOSE_MAP['đ'] = ('đ', 0)  # level tone
DECOMPOSE_MAP['Đ'] = ('Đ', 0)  # level tone
COMPOSE_MAP[('đ', 0)] = 'đ'
COMPOSE_MAP[('Đ', 0)] = 'Đ'

# All other characters (consonants, numbers, punctuation) map to themselves with level tone
OTHER_CHARS = (
    'bcdfghjklmnpqrstvwxz'
    'BCDFGHJKLMNPQRSTVWXZ'
    '0123456789'
    '.,!?;: "#&\'()*+-/%=<>@[]^_`{|}~'
)
for char in OTHER_CHARS:
    DECOMPOSE_MAP[char] = (char, 0)  # level tone
    COMPOSE_MAP[(char, 0)] = char

def is_vowel(char: str) -> bool:
    """Check if character is a Vietnamese vowel (including modified forms)."""
    vowels = set('aăâeêioôơuưyAĂÂEÊIOÔƠUƯY')
    return char in vowels

def decompose_char(ch: str) -> Tuple[str, int]:
    """Return (base_char, tone_id) for a single character.
    Unknown chars are treated as themselves with level tone.
    """
    if ch in DECOMPOSE_MAP:
        return DECOMPOSE_MAP[ch]
    # Default: keep char as base with level tone
    return ch, 0

def compose_char(base: str, tone_id: int) -> str:
    """Compose a character from base and tone_id."""
    if tone_id < 0 or tone_id >= len(TONES):
        tone_id = 0  # Default to level tone
    
    key = (base, tone_id)
    if key in COMPOSE_MAP:
        return COMPOSE_MAP[key]
    
    # Default: return base character
    return base

def decompose_str(s: str) -> Tuple[str, List[int]]:
    """Decompose an input string into base_str and tone_ids.
    - base_str: string of base characters (same length as s)  
    - tone_ids: list of tone ids per character
    """
    base_chars: List[str] = []
    tone_ids: List[int] = []
    
    for ch in s:
        base, tid = decompose_char(ch)
        base_chars.append(base)
        tone_ids.append(tid)
    
    return ''.join(base_chars), tone_ids

def compose_str(base_str: str, tone_preds: List[int]) -> str:
    """Compose a string from base_str and predicted tone ids.
    Lengths must match; if lists are shorter, missing entries default to level tone.
    """
    n = len(base_str)
    out_chars: List[str] = []
    
    for i in range(n):
        base = base_str[i]
        tid = tone_preds[i] if i < len(tone_preds) else 0  # default to level
        out_chars.append(compose_char(base, tid))
    
    return ''.join(out_chars)

__all__ = [
    'TONES', 'TONE2ID', 'BASE_CHARS_LOWER', 'BASE_CHARS_UPPER',
    'decompose_str', 'compose_str', 'decompose_char', 'compose_char',
    'is_vowel'
]
