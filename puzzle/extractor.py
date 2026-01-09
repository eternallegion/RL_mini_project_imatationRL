import re
from typing import Optional
from effect_dsl import EffectProgram

NUM = r"(\d{1,4})"

PATTERNS = [
    # 파괴
    (re.compile(r"(상대|적).*(마법|함정|세트|세트된|백로우).*(1|한)\s*장.*파괴"),
     lambda m: {"effect": [{"op": "destroy", "target": {"owner": "opponent", "zone": "spelltrap", "count": 1}}]}),
    (re.compile(r"destroy.*?(opponent|their).*(spell|trap|set|backrow).*(1|one)"),
     lambda m: {"effect": [{"op": "destroy", "target": {"owner": "opponent", "zone": "spelltrap", "count": 1}}]}),

    # 공격력 상승
    (re.compile(r"(자신|내).*공격력.*?(\+|상승|오른다)\s*"+NUM+r".*?(턴 종료|턴 끝|엔드)"),
     lambda m: {"effect": [{"op": "add_buff", "target": {"owner": "self", "zone": "monster"}, "amount": int(m.group(3)), "until": "end_of_turn"}]}),

    # 드로우
    (re.compile(r"(카드|패).*(1|한)\s*장.*드로우"),
     lambda m: {"effect": [{"op": "draw", "target": {"owner": "self"}, "amount": 1}]}),

    # 상대가 다시 블록 킨다
    (re.compile(r"(상대|적).*(다시|재차).*(세트|백로우|블록).*(준비|설치|세팅)"),
     lambda m: {"effect": [{"op": "reblock", "target": {"owner": "opponent", "zone": "spelltrap"}}]}),
]

def normalize(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def extract_effect(text: str) -> Optional[EffectProgram]:
    t = normalize(text.lower())
    for pat, builder in PATTERNS:
        m = pat.search(t)
        if m:
            return builder(m)
    return None
