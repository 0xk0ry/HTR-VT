"""Vietnamese decomposition and composition utilities.

Provides functions to decompose a Vietnamese Unicode string into:
 - base characters (no diacritics)
 - modifier ids (none, breve, circ, horn)
 - tone ids (level, acute, grave, hook, tilde, dot)

Also supports recomposition from predicted tag ids.
This is a trimmed version tailored for dual-head training in model_v7.
"""
from typing import List, Tuple

MODIFIERS = ["none", "breve", "circ", "horn"]
TONES = ["level", "acute", "grave", "hook", "tilde", "dot"]
MOD2ID = {m: i for i, m in enumerate(MODIFIERS)}
TONE2ID = {t: i for i, t in enumerate(TONES)}

LOWER_MAP = {}
UPPER_MAP = {}

def _reg(base: str, mod: str, tone: str, lower: str, upper: str):
    LOWER_MAP[(base, mod, tone)] = lower
    UPPER_MAP[(base.upper(), mod, tone)] = upper

# a
_reg('a','none','level','a','A'); _reg('a','none','acute','á','Á'); _reg('a','none','grave','à','À'); _reg('a','none','hook','ả','Ả'); _reg('a','none','tilde','ã','Ã'); _reg('a','none','dot','ạ','Ạ')
_reg('a','breve','level','ă','Ă'); _reg('a','breve','acute','ắ','Ắ'); _reg('a','breve','grave','ằ','Ằ'); _reg('a','breve','hook','ẳ','Ẳ'); _reg('a','breve','tilde','ẵ','Ẵ'); _reg('a','breve','dot','ặ','Ặ')
_reg('a','circ','level','â','Â'); _reg('a','circ','acute','ấ','Ấ'); _reg('a','circ','grave','ầ','Ầ'); _reg('a','circ','hook','ẩ','Ẩ'); _reg('a','circ','tilde','ẫ','Ẫ'); _reg('a','circ','dot','ậ','Ậ')
# e
_reg('e','none','level','e','E'); _reg('e','none','acute','é','É'); _reg('e','none','grave','è','È'); _reg('e','none','hook','ẻ','Ẻ'); _reg('e','none','tilde','ẽ','Ẽ'); _reg('e','none','dot','ẹ','Ẹ')
_reg('e','circ','level','ê','Ê'); _reg('e','circ','acute','ế','Ế'); _reg('e','circ','grave','ề','Ề'); _reg('e','circ','hook','ể','Ể'); _reg('e','circ','tilde','ễ','Ễ'); _reg('e','circ','dot','ệ','Ệ')
# i
_reg('i','none','level','i','I'); _reg('i','none','acute','í','Í'); _reg('i','none','grave','ì','Ì'); _reg('i','none','hook','ỉ','Ỉ'); _reg('i','none','tilde','ĩ','Ĩ'); _reg('i','none','dot','ị','Ị')
# o / ô / ơ
_reg('o','none','level','o','O'); _reg('o','none','acute','ó','Ó'); _reg('o','none','grave','ò','Ò'); _reg('o','none','hook','ỏ','Ỏ'); _reg('o','none','tilde','õ','Õ'); _reg('o','none','dot','ọ','Ọ')
_reg('o','circ','level','ô','Ô'); _reg('o','circ','acute','ố','Ố'); _reg('o','circ','grave','ồ','Ồ'); _reg('o','circ','hook','ổ','Ổ'); _reg('o','circ','tilde','ỗ','Ỗ'); _reg('o','circ','dot','ộ','Ộ')
_reg('o','horn','level','ơ','Ơ'); _reg('o','horn','acute','ớ','Ớ'); _reg('o','horn','grave','ờ','Ờ'); _reg('o','horn','hook','ở','Ở'); _reg('o','horn','tilde','ỡ','Ỡ'); _reg('o','horn','dot','ợ','Ợ')
# u / ư
_reg('u','none','level','u','U'); _reg('u','none','acute','ú','Ú'); _reg('u','none','grave','ù','Ù'); _reg('u','none','hook','ủ','Ủ'); _reg('u','none','tilde','ũ','Ũ'); _reg('u','none','dot','ụ','Ụ')
_reg('u','horn','level','ư','Ư'); _reg('u','horn','acute','ứ','Ứ'); _reg('u','horn','grave','ừ','Ừ'); _reg('u','horn','hook','ử','Ử'); _reg('u','horn','tilde','ữ','Ữ'); _reg('u','horn','dot','ự','Ự')
# y
_reg('y','none','level','y','Y'); _reg('y','none','acute','ý','Ý'); _reg('y','none','grave','ỳ','Ỳ'); _reg('y','none','hook','ỷ','Ỷ'); _reg('y','none','tilde','ỹ','Ỹ'); _reg('y','none','dot','ỵ','Ỵ')

PLAIN_LOWER = list("bcdfghjklmnpqrstvwxz")
for ch in PLAIN_LOWER:
    LOWER_MAP[(ch,'none','level')] = ch
    UPPER_MAP[(ch.upper(),'none','level')] = ch.upper()
LOWER_MAP[('đ','none','level')] = 'đ'
UPPER_MAP[('Đ','none','level')] = 'Đ'

REV_MAP = {v:k for k,v in LOWER_MAP.items()}
REV_MAP.update({v:k for k,v in UPPER_MAP.items()})

def decompose_char(ch: str):
    if ch in REV_MAP:
        base, mod, tone = REV_MAP[ch]
        return base, MOD2ID.get(mod,0), TONE2ID.get(tone,0)
    return ch, MOD2ID['none'], TONE2ID['level']

def decompose_str(s: str) -> Tuple[str, List[int], List[int]]:
    bases=[]; mods=[]; tones=[]
    for ch in s:
        b,m,t = decompose_char(ch)
        bases.append(b); mods.append(m); tones.append(t)
    return ''.join(bases), mods, tones

def compose_char(base: str, mod_id: int, tone_id: int) -> str:
    mod = MODIFIERS[mod_id] if 0 <= mod_id < len(MODIFIERS) else 'none'
    tone = TONES[tone_id] if 0 <= tone_id < len(TONES) else 'level'
    key = (base, mod, tone)
    if base.isupper():
        return UPPER_MAP.get(key, base)
    return LOWER_MAP.get(key, base)

def compose_str(base_str: str, mod_ids: List[int], tone_ids: List[int]) -> str:
    out=[]
    for i,b in enumerate(base_str):
        mid = mod_ids[i] if i < len(mod_ids) else 0
        tid = tone_ids[i] if i < len(tone_ids) else 0
        out.append(compose_char(b, mid, tid))
    return ''.join(out)

VOWELS_BASE = set(list("aAeEiIoOuUyY"))
VOWELS_EXT = set(list("ăĂâÂêÊôÔơƠưƯ"))
def is_vowel(ch: str) -> bool:
    return ch in VOWELS_BASE or ch in VOWELS_EXT

__all__ = [
    'MODIFIERS','TONES','MOD2ID','TONE2ID','decompose_str','compose_str','is_vowel'
]