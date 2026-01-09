# puzzle_env.py
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

@dataclass(frozen=True)
class Card:
    name: str
    kind: str            # "ATK","BUFF","BLOCK_BREAK","HEAL","DRAW","SCRIPT" 등
    value: int = 0
    script: dict | None = None   # SCRIPT 카드의 DSL 저장용

@dataclass(frozen=True)
class PuzzleState:
    my_lp: int = 4000
    opp_lp: int = 4000
    hand: Tuple[Card, ...] = ()
    my_buff: int = 0
    opp_block: bool = False
    used: Tuple[int, ...] = ()
    scripted_draw: Tuple[Card, ...] = ()
    last_damage: int = 0
    # 🔹 체인 스택 (가벼운 DSL 스텝 dict들의 튜플)
    stack: Tuple[dict, ...] = ()

    def is_terminal(self) -> bool:
        return (self.opp_lp <= 0) or (len(self.used) == len(self.hand))

    def legal_actions(self) -> List[int]:
        if self.is_terminal():
            return []
        return [i for i in range(len(self.hand)) if i not in self.used]

def _apply_draw(hand: Tuple[Card, ...], scripted_draw: Tuple[Card, ...]) -> Tuple[Tuple[Card, ...], Tuple[Card, ...]]:
    if not scripted_draw:
        return hand, scripted_draw
    return hand + (scripted_draw[0],), scripted_draw[1:]

def _to_card(d: Dict[str, Any]) -> Card:
    return Card(
        name=d.get("name",""),
        kind=d.get("kind",""),
        value=d.get("value",0),
        script=d.get("script",None),
    )

def make_state_from_dict(pzl: Dict[str, Any]) -> PuzzleState:
    hand = tuple(_to_card(c) for c in pzl.get("hand", []))
    draw = tuple(_to_card(c) for c in pzl.get("scripted_draw", []))
    return PuzzleState(
        my_lp=pzl.get("my_lp", 4000),
        opp_lp=pzl.get("opp_lp", 4000),
        hand=hand,
        my_buff=pzl.get("my_buff", 0),
        opp_block=pzl.get("opp_block", False),
        scripted_draw=draw,
        stack=()
    )

def terminal_reward(state: PuzzleState) -> float:
    return 1.0 if state.opp_lp == 0 else 0.0

def step(state: PuzzleState, action: int) -> PuzzleState:
    c = state.hand[action]

    # SCRIPT
    if c.script:
        from effect_vm import run_effect_vm
        new_state = run_effect_vm(state, c.script)
        return PuzzleState(
            my_lp=new_state.my_lp,
            opp_lp=new_state.opp_lp,
            hand=new_state.hand,
            my_buff=new_state.my_buff,
            opp_block=new_state.opp_block,
            used=tuple(sorted(state.used + (action,))),
            scripted_draw=new_state.scripted_draw,
            last_damage=new_state.last_damage,
            stack=new_state.stack,  # 🔹 스택 유지
        )

    if c.kind == "ATK":
        dmg = c.value + state.my_buff
        if state.opp_block:
            new_lp = state.opp_lp
            new_block = False
        else:
            new_lp = max(0, state.opp_lp - dmg)
            new_block = state.opp_block
        return PuzzleState(
            my_lp=state.my_lp,
            opp_lp=new_lp,
            hand=state.hand,
            my_buff=state.my_buff,
            opp_block=new_block,
            used=tuple(sorted(state.used + (action,))),
            scripted_draw=state.scripted_draw,
            last_damage=dmg,
            stack=state.stack
        )

    if c.kind == "BUFF":
        new_buff = state.my_buff + c.value
        return PuzzleState(**{**state.__dict__, "my_buff": new_buff, "used": tuple(sorted(state.used + (action,)))})

    if c.kind == "BLOCK_BREAK":
        return PuzzleState(**{**state.__dict__, "opp_block": False, "used": tuple(sorted(state.used + (action,)))})

    if c.kind == "HEAL":
        healed_lp = min(4000, state.my_lp + c.value)
        return PuzzleState(**{**state.__dict__, "my_lp": healed_lp, "used": tuple(sorted(state.used + (action,)))})

    if c.kind == "DRAW":
        hand, rest = _apply_draw(state.hand, state.scripted_draw)
        return PuzzleState(**{**state.__dict__, "hand": hand, "scripted_draw": rest, "used": tuple(sorted(state.used + (action,)))})

    return state
