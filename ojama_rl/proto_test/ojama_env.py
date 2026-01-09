# ojama_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
import random


@dataclass(frozen=True)
class Card:
    name: str
    kind: str              # "SCRIPT" | "ATK"
    value: int = 0
    script: str = ""       # "OJAMA_TRIO" | "ZERO_GRAVITY" | "BIG_EVOLUTION_PILL"


@dataclass(frozen=True)
class State:
    opp_lp: int
    my_buff: int
    opp_block: bool
    hand: Tuple[Card, ...]
    used: Tuple[int, ...]
    stack: Tuple[Dict[str, Any], ...]


class OjamaEnv:
    """
    [GX_Spirit_Caller]B03_Oh_Jama 퍼즐 전용 간소화 환경.
    정답 시퀀스: TRIO(0) -> ZERO(1) -> PILL(2) -> ATK(3)
    """

    def __init__(self, seed: int | None = 0) -> None:
        self._seed = 0 if seed is None else seed
        self.rng = random.Random(self._seed)
        self._np_rng = np.random.RandomState(self._seed)

        self._t = 0
        self.state: State | None = None
        self._script_flags: Dict[str, bool] = {}
        self._last_flags: Dict[str, Any] = {}

        self.reset()

    # ---------- 고정 규격 ----------
    @property
    def max_hand(self) -> int:
        return 4  # 카드 4장

    @property
    def stop_index(self) -> int:
        return self.max_hand  # STOP 액션 인덱스

    # ---------- 초기 상태 ----------
    def _make_initial_state(self) -> State:
        hand = (
            Card("Ojama Trio", "SCRIPT", 1, "OJAMA_TRIO"),
            Card("Zero Gravity", "SCRIPT", 1, "ZERO_GRAVITY"),
            Card("Big Evolution Pill", "SCRIPT", 1, "BIG_EVOLUTION_PILL"),
            Card("Finisher ATK", "ATK", 3000, ""),
        )
        return State(
            opp_lp=4000,
            my_buff=0,
            opp_block=False,
            hand=hand,
            used=tuple(),
            stack=tuple(),
        )

    # ---------- API ----------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = seed
            self.rng = random.Random(self._seed)
            self._np_rng = np.random.RandomState(self._seed)

        self._t = 0
        self._script_flags = {"TRIO": False, "ZERO": False, "PILL": False}
        self._last_flags = {}
        self.state = self._make_initial_state()
        return self._obs(), {"action_mask": self.action_mask()}

    def _obs(self) -> np.ndarray:
        assert self.state is not None
        s = self.state
        # 아주 단순한 상태 특징(학습에 충분)
        used_ratio = len(s.used) / self.max_hand
        return np.array(
            [
                s.opp_lp / 10000.0,
                s.my_buff / 10000.0,
                1.0 if s.opp_block else 0.0,
                used_ratio,
            ],
            dtype=np.float32,
        )

    # 정답 순서만 허용하는 보수적 마스크(실패·탐색 흔들림 제거)
    def action_mask(self) -> np.ndarray:
        assert self.state is not None, "call reset() before action_mask()"
        s = self.state
        A = self.max_hand + 1
        mask = np.zeros(A, dtype=np.float32)
        STOP = self.stop_index

        # 이미 쓴 건 전부 금지
        used = set(s.used)

        # 다음에 허용할 정확한 인덱스
        expected_next = len(s.used)
        if expected_next < self.max_hand:
            # 아직 안 쓴 카드 중 정확히 다음 카드 1장만 허용
            if expected_next not in used:
                mask[expected_next] = 1.0

        # STOP은 마지막(ATK 수행 후)만 허용
        if len(s.used) == 4 or s.opp_lp <= 0:
            mask[STOP] = 1.0
        else:
            mask[STOP] = 0.0
        return mask

    def legal_actions(self) -> Tuple[int, ...]:
        mask = self.action_mask()
        return tuple(int(i) for i, m in enumerate(mask) if m > 0.5)

    def step(self, action: int):
        assert self.state is not None, "call reset() first"
        STOP = self.stop_index
        s = self.state

        # STOP
        if action == STOP:
            self._t += 1
            done = True
            reward = self._reward(done)
            return self._obs(), reward, done, {"stopped": True, "action_mask": self.action_mask()}

        # 합법 체크
        legal = self.legal_actions()
        if action not in legal or action >= self.max_hand:
            # 보수적으로 즉시 종료 패널티
            return self._obs(), -1.0, True, {"invalid_action": True, "action_mask": self.action_mask()}

        # 전이
        ns = self._env_step(s, action)
        if not isinstance(ns, State):
            return self._obs(), -1.0, True, {"invalid_transition": True, "action_mask": self.action_mask()}

        self.state = ns
        self._t += 1

        done = self._done()
        reward = self._reward(done)
        return self._obs(), reward, done, {"action_mask": self.action_mask()}

    # ---------- 전이/보상 ----------
    def _env_step(self, s: State, action: int) -> State:
        self._last_flags = {}
        c = s.hand[action]

        opp_lp = s.opp_lp
        my_buff = s.my_buff
        opp_block = s.opp_block
        used = tuple(list(s.used) + [action])

        # 스크립트 처리
        if c.kind == "SCRIPT":
            op = c.script.strip().upper()
            if op == "OJAMA_TRIO":
                # 체인 토큰 추가 정도로만 표기(실제 LP 영향은 ZERO 이후)
                self._script_flags["TRIO"] = True
                self._last_flags["TRIO"] = True

            elif op == "ZERO_GRAVITY":
                # ZERO가 발동되면 오자마 토큰이 전부 DEF→ATK 취급되는 느낌으로
                # 여기서는 다음 PILL 버프가 유효하게 적용되는 플래그만 켬
                if self._script_flags.get("TRIO", False):
                    self._script_flags["ZERO"] = True
                    self._last_flags["ZERO"] = True

            elif op == "BIG_EVOLUTION_PILL":
                # PILL이 발동되면 큰 버프를 건다고 가정(원턴킬 만들기)
                if self._script_flags.get("TRIO", False) and self._script_flags.get("ZERO", False):
                    my_buff = 7000  # 3000 ATK + 7000 = 10000 → 4000 원킬
                    self._script_flags["PILL"] = True
                    self._last_flags["PILL"] = True

        elif c.kind == "ATK":
            dmg = max(0, c.value + my_buff)
            if opp_block:
                dmg = 0
            opp_lp = max(0, opp_lp - dmg)

        # hand/stack는 고정
        return State(
            opp_lp=opp_lp,
            my_buff=my_buff,
            opp_block=opp_block,
            hand=s.hand,
            used=used,
            stack=s.stack,
        )

    def _done(self) -> bool:
        assert self.state is not None
        s = self.state
        # 성공 or 4장 다 사용
        if s.opp_lp <= 0:
            return True
        if len(s.used) >= self.max_hand:
            return True
        return False

    def _reward(self, done: bool) -> float:
        assert self.state is not None
        s = self.state
        if done and s.opp_lp <= 0:
            return 2.0  # 성공 보상
        # 진행 중에는 작은 shaping은 생략(보수적)
        return 0.0
