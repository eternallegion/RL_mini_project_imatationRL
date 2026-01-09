# effect_parser.py
# 간단한 정규식 기반 미니 파서: 한국어/영문 키워드 → effect_vm DSL
import re
from typing import Dict, Any

DESTROY_PATTERNS = [
    r"(세트된|세트된\s*카드|마법/함정|세트 카드).*(파괴|없앤다|제거)",
    r"(set|spell|trap).*(destroy)"
]
DRAW_PATTERNS = [
    r"(카드\s*)?(\d+)\s*장\s*드로우",
    r"draw\s*(\d+)"
]
BUFF_PATTERNS = [
    r"공격력\s*\+?\s*(\d+)",
    r"attack\s*\+?\s*(\d+)"
]
REBLOLCK_PATTERNS = [
    r"(다시|재)세트|reblock|set\s*again"
]
NEGATE_PATTERNS = [
    r"(무효|negate)\s*\(?(?:spd|speed)?\s*(\d)\)?",
]

def parse_text_to_program(text: str) -> Dict[str, Any]:
    t = (text or "").lower()

    # 1) negate(speed)
    m = None
    for pat in NEGATE_PATTERNS:
        m = re.search(pat, t)
        if m:
            spd = int(m.group(1))
            return {"effect": [{"op": "negate", "speed": spd}]}

    # 2) destroy set/spell/trap
    for pat in DESTROY_PATTERNS:
        if re.search(pat, t):
            return {"effect": [{"op": "destroy", "target": {"owner": "opponent", "zone": "spelltrap"}}]}

    # 3) draw(n)
    for pat in DRAW_PATTERNS:
        m = re.search(pat, t)
        if m:
            n = int(m.group(1))
            return {"effect": [{"op": "draw", "amount": n}]}

    # 4) add_buff(+X)
    for pat in BUFF_PATTERNS:
        m = re.search(pat, t)
        if m:
            x = int(m.group(1))
            return {"effect": [{"op": "add_buff", "target": {"owner":"self"}, "amount": x, "until": "end_of_turn"}]}

    # 5) reblock
    for pat in REBLOLCK_PATTERNS:
        if re.search(pat, t):
            return {"effect": [{"op": "reblock", "target": {"owner": "opponent"}}]}

    # fallback: 아무 것도 못 알아들으면 no-op
    return {"effect": []}
