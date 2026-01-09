from typing import Dict, Any

def apply_op(state: Dict[str, Any], op: str) -> None:
    """
    퍼즐별 스크립트 효과 정의.
    state: {'opp_lp','my_buff','opp_block','flags':{'trio','zero','pill'}, ...}
    """
    op = (op or "").upper()
    if op == "BLOCK_BREAK":
        state["opp_block"] = False
    elif op == "OJAMA_TRIO":
        state["flags"]["trio"] = True
    elif op == "ZERO_GRAVITY":
        state["flags"]["zero"] = True
    elif op == "BIG_EVOLUTION_PILL":
        state["flags"]["pill"] = True
    else:
        # 정의 없는 스크립트는 무시(보수적)
        pass

def apply_atk(state: Dict[str, Any], base: int) -> None:
    dmg = max(0, int(base) + int(state["my_buff"]))
    if state["opp_block"]:
        dmg = 0
    # OJAMA 콤보 보정(데모용): TRIO&ZERO&PILL 모두면 피니셔 관대하게 +6000
    if state["flags"].get("trio") and state["flags"].get("zero") and state["flags"].get("pill"):
        dmg = max(dmg, 9000)  # 원턴킬 유도용
    state["opp_lp"] = max(0, state["opp_lp"] - dmg)
