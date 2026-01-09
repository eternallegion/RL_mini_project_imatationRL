from dataclasses import dataclass
from typing import Tuple, List, Dict
from effect_vm import run_effect_vm


@dataclass(frozen=True)
class Card:
    name: str
    kind: str   # "ATK","HEAL","BUFF","BLOCK_BREAK","DOUBLE_NEXT_ATK","DRAW","SCRIPT"
    value: int = 0
    script: dict | None = None

@dataclass(frozen=True)
class PuzzleState:
    my_lp: int
    opp_lp: int
    hand: Tuple[Card, ...]
    my_buff: int = 0
    opp_block: bool = False
    used: Tuple[int, ...] = ()
    scripted_draw: Tuple[Card, ...] = ()
    last_damage: int = 0

    def is_terminal(self) -> bool:
        return self.opp_lp <= 0 or len(self.used) == len(self.hand)

    def legal_actions(self) -> List[int]:
        if self.is_terminal():
            return []
        return [i for i in range(len(self.hand)) if i not in self.used]

def _apply_draw(hand: Tuple[Card, ...], scripted: Tuple[Card, ...]) -> Tuple[Tuple[Card, ...], Tuple[Card, ...]]:
    if not scripted:
        return hand, scripted
    top, rest = scripted[0], scripted[1:]
    return hand + (top,), rest

def step(state: PuzzleState, action: int) -> PuzzleState:
    c = state.hand[action]
    my_lp, opp_lp = state.my_lp, state.opp_lp
    my_buff, opp_block = state.my_buff, state.opp_block
    hand, scripted = state.hand, state.scripted_draw
    last_damage = state.last_damage

    if c.script:
        new_state = run_effect_vm(state, c.script)
        return PuzzleState(
            my_lp=new_state.my_lp, opp_lp=new_state.opp_lp, hand=new_state.hand,
            my_buff=new_state.my_buff, opp_block=new_state.opp_block,
            used=tuple(sorted(state.used + (action,))),
            scripted_draw=new_state.scripted_draw, last_damage=new_state.last_damage
        )
    if c.kind == "ATK":
        double = (my_buff >= 10**6)
        base_buff = my_buff - (10**6 if double else 0)
        dmg = c.value + base_buff
        if double:
            dmg *= 2
        if opp_block:
            dmg = 0
            opp_block = False
        opp_lp = max(0, opp_lp - dmg)
        last_damage = dmg
        if double:
            my_buff = base_buff

    elif c.kind == "HEAL":
        my_lp += c.value

    elif c.kind == "BUFF":
        my_buff += c.value

    elif c.kind == "DOUBLE_NEXT_ATK":
        my_buff += 10**6  # mark “double next ATK”

    elif c.kind == "BLOCK_BREAK":
        if opp_block:
            opp_block = False
        else:
            opp_lp = max(0, opp_lp - c.value)

    elif c.kind == "DRAW":
        hand, scripted = _apply_draw(hand, scripted)

    return PuzzleState(
        my_lp=my_lp, opp_lp=opp_lp, hand=hand,
        my_buff=my_buff, opp_block=opp_block,
        used=tuple(sorted(state.used + (action,))),
        scripted_draw=scripted, last_damage=last_damage
    )

def terminal_reward(state: PuzzleState) -> float:
    return 1.0 if state.opp_lp == 0 else 0.0

def _card_from_dict(cd: Dict) -> Card:
    return Card(name=cd["name"], kind=cd["kind"], value=cd.get("value", 0))

def make_state_from_dict(d: Dict) -> PuzzleState:
    hand = tuple(_card_from_dict(c) for c in d["hand"])
    scripted = tuple(_card_from_dict(c) for c in d.get("scripted_draw", []))
    return PuzzleState(
        my_lp=d.get("my_lp", 2000),
        opp_lp=d["opp_lp"],
        hand=hand,
        my_buff=d.get("my_buff", 0),
        opp_block=d.get("opp_block", False),
        scripted_draw=scripted
    )
