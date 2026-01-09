# effect_vm.py
from typing import Dict, Any, List
from effect_parser import parse_text_to_program  # ⬅️ 신규
from effect_dsl import EffectProgram

class PuzzleGameFacade:
    def __init__(self, s, debug: bool=False):
        self.s = s
        self.debug = debug

    def _log(self, msg: str):
        if self.debug:
            print(f"[VM] {msg}")

    # --- 기본 동작 ---
    def destroy(self, target: Dict[str, Any]):
        self._log("destroy(target=opponent.spelltrap)")
        if target.get("owner") == "opponent" and target.get("zone") in ("spelltrap","block","spell/trap"):
            if self.s.opp_block:
                self.s = self.s.__class__(**{**self.s.__dict__, "opp_block": False})
                self._log(" -> opponent block destroyed")

    def add_buff(self, target: Dict[str, Any], amount: int, until: str | None):
        self._log(f"add_buff(+{amount})")
        if target.get("owner") == "self":
            self.s = self.s.__class__(**{**self.s.__dict__, "my_buff": self.s.my_buff + int(amount)})
            self._log(f" -> my_buff={self.s.my_buff}")

    def draw(self, amount: int):
        self._log(f"draw({amount})")
        from puzzle_env import _apply_draw
        hand, rest = self.s.hand, self.s.scripted_draw
        for _ in range(int(amount)):
            hand, rest = _apply_draw(hand, rest)
        self.s = self.s.__class__(**{**self.s.__dict__, "hand": hand, "scripted_draw": rest})
        self._log(f" -> hand_size={len(self.s.hand)} scripted_draw={len(self.s.scripted_draw)}")

    def reblock(self, target: Dict[str, Any]):
        self._log("reblock()")
        if not self.s.opp_block:
            self.s = self.s.__class__(**{**self.s.__dict__, "opp_block": True})
            self._log(" -> opponent block set")

    # --- 체인 스택 ---
    def _push(self, payload: List[Dict[str, Any]], speed: int):
        self._log(f"activate(speed={speed})")
        stack = list(self.s.stack)
        stack.append({"payload": payload, "speed": int(speed), "negated": False})
        self.s = self.s.__class__(**{**self.s.__dict__, "stack": tuple(stack)})
        self._log(f" -> stack size={len(stack)}")

    def _negate_top(self, speed: int):
        self._log(f"negate(speed={speed})")
        stack = list(self.s.stack)
        if not stack:
            self._log(" -> no target on stack")
            return
        top = stack[-1]
        if int(speed) >= int(top.get("speed", 1)):
            top["negated"] = True
            self._log(" -> top effect NEGATED")
        else:
            self._log(" -> negate failed (lower speed)")
        stack[-1] = top
        self.s = self.s.__class__(**{**self.s.__dict__, "stack": tuple(stack)})

    def _resolve_once(self):
        stack = list(self.s.stack)
        if not stack:
            self._log("resolve() -> empty stack")
            return
        top = stack.pop()
        self._log(f"resolve() -> pop, negated={top.get('negated', False)}")
        if not top.get("negated", False):
            for step in top.get("payload", []):
                self._exec(step)
        self.s = self.s.__class__(**{**self.s.__dict__, "stack": tuple(stack)})
        self._log(f" -> stack size={len(self.s.stack)}")

    # --- 더미 오퍼레이터 ---
    def banish(self, target: Dict[str, Any]): return
    def send_grave(self, target: Dict[str, Any]): return

    # --- 실행 디스패처 ---
    def _exec(self, step: Dict[str, Any]):
        op = step.get("op")
        if op == "destroy":
            self.destroy(step.get("target", {}))
        elif op == "add_buff":
            self.add_buff(step.get("target", {}), step.get("amount", 0), step.get("until"))
        elif op == "draw":
            self.draw(step.get("amount", 1))
        elif op == "reblock":
            self.reblock(step.get("target", {}))
        elif op == "banish":
            self.banish(step.get("target", {}))
        elif op == "send_grave":
            self.send_grave(step.get("target", {}))
        elif op == "activate":
            payload = step.get("payload", [])
            speed = int(step.get("speed", 1))
            self._push(payload, speed)
        elif op == "negate":
            speed = int(step.get("speed", 2))
            self._negate_top(speed)
        elif op == "resolve":
            self._resolve_once()

    def commit(self):
        return self.s

def _ensure_program(program: Dict[str, Any]) -> Dict[str, Any]:
    # program에 effect가 없고 text만 있으면 파싱
    if isinstance(program, dict):
        if "effect" in program:
            return program
        if "text" in program:
            return parse_text_to_program(program.get("text",""))
    # 비정형 입력은 no-op
    return {"effect": []}

def run_effect_vm(state, program: EffectProgram, debug: bool=False):
    prog = _ensure_program(program)
    g = PuzzleGameFacade(state, debug=debug)
    for step in prog.get("effect", []):
        g._exec(step)
    return g.commit()
