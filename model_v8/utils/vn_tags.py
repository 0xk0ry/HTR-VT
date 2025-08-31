"""Tone-only Vietnamese decomposition utilities (based on v6 reference).

Provides:
	decompose_str(text) -> (base_str, tone_ids)
	compose_str(base_str, tone_ids) -> reconstructed string

Tones:
	0: level, 1: acute, 2: grave, 3: hook, 4: tilde, 5: dot

Rules:
 - Modifier forms (ă â ê ô ơ ư) are treated as distinct base characters.
 - Non-vowel characters map to themselves with level tone (0).
"""
from typing import List, Tuple

TONES = ["level", "acute", "grave", "hook", "tilde", "dot"]
TONE2ID = {t: i for i, t in enumerate(TONES)}

DECOMPOSE_MAP = {}
COMPOSE_MAP = {}


def _reg(base_lower: str, base_upper: str, level: str, acute: str, grave: str, hook: str, tilde: str, dot: str):
	lowers = [level, acute, grave, hook, tilde, dot]
	uppers = [c.upper() for c in lowers]
	for tid, ch in enumerate(lowers):
		DECOMPOSE_MAP[ch] = (base_lower, tid)
		COMPOSE_MAP[(base_lower, tid)] = ch
	for tid, ch in enumerate(uppers):
		if ch != lowers[tid]:  # has uppercase distinct
			DECOMPOSE_MAP[ch] = (base_upper, tid)
			COMPOSE_MAP[(base_upper, tid)] = ch


# Vowel registrations (modifier variants are bases themselves)
_reg('a', 'A', 'a', 'á', 'à', 'ả', 'ã', 'ạ')
_reg('ă', 'Ă', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ')
_reg('â', 'Â', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ')
_reg('e', 'E', 'e', 'é', 'è', 'ẻ', 'ẽ', 'ẹ')
_reg('ê', 'Ê', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ')
_reg('i', 'I', 'i', 'í', 'ì', 'ỉ', 'ĩ', 'ị')
_reg('o', 'O', 'o', 'ó', 'ò', 'ỏ', 'õ', 'ọ')
_reg('ô', 'Ô', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ')
_reg('ơ', 'Ơ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ')
_reg('u', 'U', 'u', 'ú', 'ù', 'ủ', 'ũ', 'ụ')
_reg('ư', 'Ư', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự')
_reg('y', 'Y', 'y', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ')

# đ / Đ
DECOMPOSE_MAP['đ'] = ('đ', 0)
DECOMPOSE_MAP['Đ'] = ('Đ', 0)
COMPOSE_MAP[('đ', 0)] = 'đ'
COMPOSE_MAP[('Đ', 0)] = 'Đ'

# Others map to themselves level tone
OTHER = (
	'bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ'
	'0123456789'
	'.,!?;: "#&\'()*+-/%=<>@[]^_`{|}~'
)
for ch in OTHER:
	DECOMPOSE_MAP[ch] = (ch, 0)
	COMPOSE_MAP[(ch, 0)] = ch

_VOWELS = set('aăâeêioôơuưyAĂÂEÊIOÔƠUƯY')


def is_vowel(ch: str) -> bool:
	return ch in _VOWELS


def decompose_char(ch: str) -> Tuple[str, int]:
	return DECOMPOSE_MAP.get(ch, (ch, 0))


def compose_char(base: str, tone_id: int) -> str:
	return COMPOSE_MAP.get((base, tone_id if 0 <= tone_id < len(TONES) else 0), base)


def decompose_str(s: str) -> Tuple[str, List[int]]:
	bases, tones = [], []
	for ch in s:
		b, t = decompose_char(ch)
		bases.append(b)
		tones.append(t)
	return ''.join(bases), tones


def compose_str(base_str: str, tone_preds: List[int]) -> str:
	out = []
	for i, b in enumerate(base_str):
		t = tone_preds[i] if i < len(tone_preds) else 0
		out.append(compose_char(b, t))
	return ''.join(out)


__all__ = [
	'TONES', 'TONE2ID', 'decompose_str', 'compose_str', 'decompose_char', 'compose_char', 'is_vowel'
]

