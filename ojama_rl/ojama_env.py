# ojama_env.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any
import logging
import numpy as np
import os
import random

# ====== 로깅 공통 설정 ======
def _make_logger(name: str, logfile: str):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # 중복 핸들러 방지
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_ojama_log", False)
               for h in logger.handlers):
        fh = logging.FileHandler(logfile)
        fh._ojama_log = True  # 마킹
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@dataclass
class State:
    turn: int = 0
    used: Tuple[int, ...] = field(default_factory=tuple)
    opp_lp: int = 4000
    flags: Dict[str, bool] = field(default_factory=lambda: {"trio": False, "zero": False, "pill": False})


class OjamaEnv:
    """
    퍼즐 듀얼(오자마) 환경 - 순서 강제 버전
    액션 인덱스: 0=TRIO, 1=ZERO, 2=PILL, 3=ATK
    require_order=True면 반드시 TRIO→ZERO→PILL→ATK 순으로만 진행 가능
    """
    def __init__(self, seed: int | None = None, require_order: bool = True, randomize_lp: bool = False):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed if seed is not None else 2025)
        self.require_order = require_order
        self.randomize_lp = randomize_lp

        self.action_dim = 4
        self.state: State | None = None

        self.solve_logger = _make_logger("solve", "logs/solve.log")
        self.eval_logger = _make_logger("eval", "logs/eval.log")

        self.reset()

    # ========== 유틸 ==========
    @staticmethod
    def _op_name(a: int) -> str:
        return ["TRIO", "ZERO", "PILL", "ATK"][a]

    def _obs(self) -> np.ndarray:
        """간단한 관측: [turn, trio, zero, pill, opp_lp/4000]"""
        s = self.state
        trio = 1.0 if s.flags["trio"] else 0.0
        zero = 1.0 if s.flags["zero"] else 0.0
        pill = 1.0 if s.flags["pill"] else 0.0
        return np.array([s.turn, trio, zero, pill, s.opp_lp / 4000.0], dtype=np.float32)

    def _reward(self, done: bool) -> float:
        s = self.state
        if done:
            return 1.0 if s.opp_lp <= 0 else -1.0
        # 진행 보상(약함)
        return 0.0

    # ========== 인터페이스 ==========
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.RandomState(seed)
        lp = 4000
        if self.randomize_lp:
            lp = int(self.np_rng.choice([3000, 3500, 4000]))

        self.state = State(turn=0, used=tuple(), opp_lp=lp, flags={"trio": False, "zero": False, "pill": False})
        return self._obs(), {"action_mask": self.action_mask()}

    def action_mask(self) -> np.ndarray:
        """
        사용 가능 액션 마스크
        require_order=True:
          - 아직 trio 안썼으면 TRIO(0)만 1
          - trio만 썼으면 ZERO(1)만 1
          - trio+zero 썼으면 PILL(2)만 1
          - trio+zero+pill 썼으면 ATK(3)만 1
        """
        s = self.state
        assert s is not None, "call reset() first"
        mask = np.zeros(self.action_dim, dtype=np.float32)

        if self.require_order:
            if not s.flags["trio"]:
                mask[0] = 1.0
            elif not s.flags["zero"]:
                mask[1] = 1.0
            elif not s.flags["pill"]:
                mask[2] = 1.0
            else:
                mask[3] = 1.0
            return mask

        # 자유 순서(옵션)
        # 기본: 아직 안 쓴 액션만 열기
        used_set = set(s.used)
        for i in range(3):  # TRIO,ZERO,PILL
            if i not in used_set:
                mask[i] = 1.0
        # 모든 플래그가 켜졌다면 ATK 허용
        if s.flags["trio"] and s.flags["zero"] and s.flags["pill"]:
            mask[3] = 1.0
        return mask

    def step(self, a: int):
        assert self.state is not None, "call reset() first"
        s = self.state

        mask = self.action_mask()
        if a < 0 or a >= self.action_dim or mask[a] <= 0.0:
            # 순서 위반 등 불법 액션은 즉시 실패 처리
            self.solve_logger.debug(
                f"[BLOCK] t={s.turn+1} tried={self._op_name(a)} "
                f"flags={s.flags} LP={s.opp_lp}"
            )
            obs = self._obs()
            return obs, -1.0, True, {"invalid_action": True, "action_mask": self.action_mask()}

        name = self._op_name(a)
        prev_lp = s.opp_lp

        # 효과 적용
        if name == "TRIO":
            s.flags["trio"] = True
        elif name == "ZERO":
            s.flags["zero"] = True
        elif name == "PILL":
            s.flags["pill"] = True
        elif name == "ATK":
            # 세 플래그가 모두 켜졌다면 원턴킬
            if s.flags["trio"] and s.flags["zero"] and s.flags["pill"]:
                s.opp_lp = 0
            else:
                # 이 경로는 마스크 상 열리지 않지만, 안전차원에서 처리
                s.opp_lp = max(0, s.opp_lp - 500)

        s.used = tuple(list(s.used) + [a])
        s.turn += 1

        # 로그(풀이 과정)
        self.solve_logger.debug(
            f"  t={s.turn:02d}  act={name}(idx={a})"
            f"  LP:{prev_lp}→{s.opp_lp}  flags={s.flags}"
        )

        done = (s.opp_lp <= 0) or (s.turn >= 4)  # 4스텝 내 성공/실패
        reward = self._reward(done)

        obs = self._obs()
        return obs, reward, done, {"action_mask": self.action_mask()}
