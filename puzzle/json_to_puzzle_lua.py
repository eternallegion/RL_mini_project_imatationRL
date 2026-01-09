# json_to_puzzle_lua.py
import json
import argparse
from pathlib import Path
from textwrap import dedent

# ✅ 네 환경의 kind들을 실제 카드 코드로 매핑 (원하면 자유롭게 교체)
#  - 안전한 기본값들 (OCG에 기본 포함된 카드)
KIND_TO_CARD = {
    "ATK": 89631139,         # Blue-Eyes White Dragon (예: 공격 대표)
    "BUFF": 40619825,        # Axe of Despair (장착 버프)
    "BLOCK_BREAK": 5318639, # Mystical Space Typhoon (마/함 파괴)
    "HEAL": 73915051,        # Dian Keto the Cure Master (회복)
    "DRAW": 53129443,        # Pot of Greed (드로우) - 금지지만 퍼즐 데모용
    "SCRIPT": 5318639,       # Monster Reborn (임의로 '스크립트' 자리표시)
}

def _lua_header(name, my_lp, opp_lp, constraints, notes):
    hint_lines = []
    if constraints:
        if constraints.get("must_break_before_first_atk"):
            hint_lines.append("- 먼저 차단 해제 후 공격하세요.")
        if constraints.get("must_end_with") == "ATK":
            hint_lines.append("- 마지막 액션은 ATK여야 합니다.")
        forbid = constraints.get("forbid_kinds") or []
        if forbid:
            hint_lines.append(f"- 금지된 액션: {', '.join(forbid)}")
        lim = constraints.get("limit_kind_counts", {})
        if lim.get("ATK"):
            hint_lines.append(f"- ATK 횟수 제한: {lim['ATK']}회")

    if notes:
        hint_lines.append(f"- NOTE: {notes}")

    hints = "\\n".join(hint_lines) if hint_lines else "이 퍼즐을 해결하세요."

    return dedent(f"""
    -- Auto-generated from puzzles.json: {name}
    -- LP/제약/힌트를 포함한 퍼즐 템플릿입니다. (카드코드는 아래 매핑을 변경해 커스터마이즈 가능)
    Debug.SetAIName("RL-PUZZLE")
    Debug.ReloadFieldBegin(DUEL_ATTACK_FIRST_TURN+DUEL_SIMPLE_AI,4)
    Debug.SetPlayerInfo(0,{my_lp},0,0)  -- player
    Debug.SetPlayerInfo(1,{opp_lp},0,0) -- opponent
    Debug.ShowHint("{hints}")
    """).lstrip()

def _lua_footer():
    return dedent("""
    Debug.ReloadFieldEnd()
    aux.BeginPuzzle()
    """).lstrip()

def _emit_hand(hand, controller=0):
    """
    LOCATION_HAND에 순서대로 카드 배치. 손패 인덱스 = RL의 action 인덱스와 일치하도록 함.
    """
    lines = []
    for i, card in enumerate(hand):
        kind = card.get("kind")
        code = KIND_TO_CARD.get(kind, 89631139)
        # AddCard(code, owner, controller, location, sequence, position)
        lines.append(f"Debug.AddCard({code}, {controller}, {controller}, LOCATION_HAND, {i}, POS_FACEDOWN)")
    return "\n".join(lines)

def _emit_opp_block(opp_block):
    # 간단히 상대 필드에 "차단"을 상징하는 카드(예: Mirror Force) 하나 오픈 세트
    if opp_block:
        return dedent("""
        -- 상대의 '차단' 상황을 상징 (예: 함정 세트)
        Debug.AddCard(44095762,1,1,LOCATION_SZONE,2,POS_FACEDOWN) -- Mirror Force
        """).strip()
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=str, default="puzzles.json")
    ap.add_argument("--name", type=str, required=True, help="생성할 퍼즐 이름")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    data = json.loads(Path(args.puzzles).read_text(encoding="utf-8"))
    pzl = next(p for p in data if p["name"] == args.name)

    my_lp = int(pzl.get("my_lp", 8000))
    opp_lp = int(pzl.get("opp_lp", 8000))
    hand = pzl.get("hand", [])
    constraints = pzl.get("constraints", {})
    opp_block = bool(pzl.get("opp_block", False))

    out_path = args.out or f"puzzle_{args.name}.lua"

    lua = []
    lua.append(_lua_header(args.name, my_lp, opp_lp, constraints, args.note))
    lua.append("-- 손패 구성 (RL action 인덱스와 동일한 순서)")
    lua.append(_emit_hand(hand, controller=0))
    if opp_block:
        lua.append("\n" + _emit_opp_block(True))

    # (필요시) 필드/묘지/덱 초기화 추가 지점 —— 원하는 만큼 확장 가능
    lua.append(_lua_footer())

    Path(out_path).write_text("\n".join(lua), encoding="utf-8")
    print(f"[ok] wrote {out_path}")

if __name__ == "__main__":
    main()
