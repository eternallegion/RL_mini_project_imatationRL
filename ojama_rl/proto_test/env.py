# env.py
from dataclasses import dataclass
import json
import numpy as np

@dataclass
class State:
    trio: bool = False
    gravity: bool = False
    pill: bool = False
    dino: bool = False
    t: int = 0
    opp_lp: int = 4000

class OjamaEnv:
    def __init__(self, puzzle_path="puzzle.json"):
        with open(puzzle_path, "r", encoding="utf-8") as f:
            pzl = json.load(f)
        self.name = pzl["name"]
        self.ACTS = pzl["actions"]
        self.idx = {a: i for i, a in enumerate(self.ACTS)}
        self.max_steps = 8

        rules = pzl.get("rules", {})
        self.init_opp_lp = int(rules.get("opp_lp", 4000))
        self.attack_damage = int(rules.get("attack_damage", 4000))
        self.must_order = rules.get("must_order", [])
        self.stop_after = bool(rules.get("stop_after", True))

        self.flags_init = pzl.get("flags_init", {})

    def reset(self, seed: int | None = None):
        self.s = State()
        self.s.opp_lp = self.init_opp_lp
        # 초기 플래그(예: has_stego) 참고할 일이 생기면 self.flags_init 사용
        return self.obs(), {"action_mask": self.action_mask()}

    def obs(self):
        s = self.s
        # 관측에 opp_lp 포함(정규화 없이 단순 노출)
        return np.array(
            [float(s.trio), float(s.gravity), float(s.pill), float(s.dino),
             float(s.t), float(s.opp_lp)],
            dtype=np.float32,
        )

    def _order_ok(self, act: str) -> bool:
        """must_order 순서를 어겼는지 체크(어기면 False)."""
        if not self.must_order:
            return True
        # 현재까지의 진행 단계
        needed = ["OJAMA_TRIO", "ZERO_GRAVITY", "BIG_EVOLUTION_PILL", "ATTACK_ALL"]
        stage = 0
        if self.s.trio: stage = 1
        if self.s.gravity: stage = 2
        if self.s.pill: stage = 3
        if self.s.dino: stage = 4
        # 다음에 와야 할 액션
        expect = needed[stage] if stage < len(needed) else "STOP"
        if act == "STOP":
            return True
        return act == expect

    def action_mask(self):
        s = self.s
        m = np.zeros(len(self.ACTS), np.float32)

        # 순서 기반 마스크 (must_order를 충실히 반영)
        if not s.trio:
            m[self.idx["OJAMA_TRIO"]] = 1.0
        elif not s.gravity:
            m[self.idx["ZERO_GRAVITY"]] = 1.0
        elif not s.pill:
            m[self.idx["BIG_EVOLUTION_PILL"]] = 1.0
        elif not s.dino:
            m[self.idx["ATTACK_ALL"]] = 1.0
        else:
            m[self.idx["STOP"]] = 1.0

        return m

    def _terminal_reward(self) -> float:
        """STOP에서의 보상."""
        # 정답 조건: 모든 플래그 완료 + 상대 LP <= 0
        if self.s.trio and self.s.gravity and self.s.pill and self.s.dino and self.s.opp_lp <= 0:
            return 1.0
        # 진행 미완료 혹은 LP가 남았으면 패널티
        return -0.5

    def step(self, a: int):
        assert hasattr(self, "s"), "call reset() first"
        act = self.ACTS[a]

        mask = self.action_mask()
        if a < 0 or a >= len(self.ACTS) or mask[a] == 0.0:
            # 불법 액션은 즉시 종료 + 패널티
            return self.obs(), -1.0, True, {"invalid": True, "action_mask": self.action_mask()}

        r, done = 0.0, False

        # 순서 위반 보호(마스크가 막아주지만, 이중 방어)
        if act not in ("STOP",) and not self._order_ok(act):
            return self.obs(), -1.0, True, {"order_violation": True, "action_mask": self.action_mask()}

        if act == "OJAMA_TRIO":
            if not self.s.trio:
                self.s.trio = True
                r += 0.1

        elif act == "ZERO_GRAVITY":
            if self.s.trio and not self.s.gravity:
                self.s.gravity = True
                r += 0.1
            else:
                # 선행조건 미충족이면 약한 패널티
                r -= 0.1

        elif act == "BIG_EVOLUTION_PILL":
            # (단순화) 선행: TRIO & GRAVITY가 끝난 후에만 의미있게 적용
            if self.s.trio and self.s.gravity and not self.s.pill:
                self.s.pill = True
                r += 0.1
            else:
                r -= 0.1

        elif act == "ATTACK_ALL":
            # 핵심: 앞선 3개가 모두 True일 때만 대미지 적용
            if self.s.trio and self.s.gravity and self.s.pill and not self.s.dino:
                self.s.dino = True
                prev_lp = self.s.opp_lp
                self.s.opp_lp = max(0, self.s.opp_lp - self.attack_damage)
                # 대미지에 비례한 추가 보상(선형, 선택)
                dmg_gain = (prev_lp - self.s.opp_lp) / max(1, self.init_opp_lp)
                r += 0.5 + 0.2 * dmg_gain
            else:
                # 조건 미충족 공격은 무효 처리(가벼운 패널티)
                r -= 0.2

        elif act == "STOP":
            r += self._terminal_reward()
            done = True

        self.s.t += 1
        if self.s.t >= self.max_steps:
            done = True

        return self.obs(), r, done, {"action_mask": self.action_mask()}
