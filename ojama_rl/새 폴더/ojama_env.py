# ojama_env.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict, Any

# 고정 퍼즐 정의
CARD_OPS = ["TRIO", "ZERO", "PILL", "ATK"]  # 인덱스 0~3
STOP_IDX = 4
INIT_OPP_LP = 4000

@dataclass(frozen=True)
class State:
    opp_lp: int
    trio: bool
    zero: bool
    pill: bool
    used: Tuple[int, ...]  # 사용한 카드 인덱스들
    turn: int

class OjamaEnv:
    """
    [GX_Spirit_Caller] B03_Oh_Jama 퍼즐 전용 간단 Env
    - 카드 4장: TRIO, ZERO, PILL, ATK
    - 목표: TRIO -> ZERO -> PILL -> ATK 로 1턴 킬
    - 규칙(간소화):
      * 매 카드 최대 1회 사용
      * STOP은 마지막 사용 카드가 ATK일 때만 허용(must_end_with=ATK)
      * ATK는 TRIO, ZERO, PILL이 모두 선행된 경우에만 상대 LP=0 (그 외 ATK는 피해 0)
    """
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.max_hand = 4
        self.cards = CARD_OPS[:]  # index -> op
        self.state: State | None = None
        self._t = 0
        self.require_order = require_order
        # 에피소드별 풀이 과정(trace) 저장용
        self.trace: List[Dict[str, Any]] = []
        self.reset()

    # ----------- 필수 API -----------
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._t = 0
        self.state = State(
            opp_lp=INIT_OPP_LP,
            trio=False,
            zero=False,
            pill=False,
            used=tuple(),
            turn=0
        )
        self.trace = []  # 새 에피소드 trace 초기화
        obs = self._obs()
        info = {"action_mask": self.action_mask()}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        mask = self.action_mask()
        if mask[a] <= 0.0:
            return self._obs(), -1.0, True, {"invalid_action": True}
        assert self.state is not None, "reset() 먼저 호출하세요."
        s = self.state
        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # 잘못된 범위
        if not (0 <= action <= self.max_hand):
            return self._obs(), -1.0, True, {"invalid_action": True, "action_mask": self.action_mask()}

        # STOP 처리(마지막 카드 ATK였을 때만 허용)
        if action == STOP_IDX:
            if self._last_used_is_atk() and len(self._unused_indices()) == 0:
                done = True
                reward = self._terminal_reward()
                # trace 기록
                self.trace.append({
                    "t": self._t, "action": "STOP",
                    "lp_before": s.opp_lp, "lp_after": s.opp_lp,
                    "flags": {"trio": s.trio, "zero": s.zero, "pill": s.pill}
                })
                obs = self._obs()
                info["action_mask"] = self.action_mask()
                return obs, reward, done, info
            else:
                # 허용 안 되는 STOP
                return self._obs(), -1.0, True, {"invalid_stop": True, "action_mask": self.action_mask()}

        # 이미 쓴 카드 금지
        if action in s.used:
            return self._obs(), -1.0, True, {"invalid_action": True, "action_mask": self.action_mask()}

        op = self.cards[action]  # "TRIO"/"ZERO"/"PILL"/"ATK"
        lp_before = s.opp_lp

        # 효과 적용
        trio, zero, pill = s.trio, s.zero, s.pill
        opp_lp = s.opp_lp

        if op == "TRIO":
            trio = True
        elif op == "ZERO":
            zero = True
        elif op == "PILL":
            pill = True
        elif op == "ATK":
            # 선행 3장 모두 사용되었다면 원턴킬
            if trio and zero and pill:
                opp_lp = 0  # 킬
            else:
                # 선행 조건 부족하면 피해 없음(학습 안내 목적)
                opp_lp = opp_lp

        used = tuple(list(s.used) + [action])
        self.state = State(
            opp_lp=opp_lp,
            trio=trio,
            zero=zero,
            pill=pill,
            used=used,
            turn=s.turn + 1
        )
        self._t += 1

        # shaping: ATK로 LP를 0으로 만들면 큰 보상, 그 외에는 소량의 진행 보상
        if opp_lp == 0:
            reward = 1.0
            done = True  # 킬 즉시 종료
        else:
            reward = 0.01  # 소폭 진행 보상

        # trace 기록
        self.trace.append({
            "t": self._t,
            "action": f"{op}(idx={action})",
            "lp_before": lp_before,
            "lp_after": opp_lp,
            "flags": {"trio": trio, "zero": zero, "pill": pill},
            "used": used
        })

        # 모든 카드 사용 후 마지막이 ATK면 STOP 가능, 아니라면 더 진행 필요
        if len(used) == self.max_hand and not (self._last_used_is_atk()):
            # 마지막이 ATK가 아니면 실패로 종료(불가능한 수열)
            done = True
            reward = 0.0

        obs = self._obs()
        info["action_mask"] = self.action_mask()
        return obs, reward, done, info

    #def action_mask(self) -> np.ndarray:
        """
        사용 가능 액션: [4개 카드 + STOP] 길이의 마스크.
        - 아직 사용하지 않은 카드는 1
        - STOP은 '마지막 사용 카드가 ATK'이고 '더 이상 사용할 카드가 없을 때'만 1
        """
     #   s = self.state
     #   assert s is not None
     #   mask = np.zeros(self.max_hand + 1, dtype=np.float32)

        # 아직 안쓴 카드 1
     #   for i in range(self.max_hand):
     #       if i not in s.used:
     #           mask[i] = 1.0

        # STOP 규칙: 모든 카드를 다 썼고, 마지막 카드가 ATK일 때만
     #   if len(s.used) == self.max_hand and self._last_used_is_atk():
     #       mask[STOP_IDX] = 1.0
     #   else:
     #       mask[STOP_IDX] = 0.0
     #   return mask


    def action_mask(self):
        # 인덱스: 0=TRIO, 1=ZERO, 2=PILL, 3=ATK, 4=STOP(있다면)
        mask = np.zeros(self.action_dim, dtype=np.float32)
        t = self.state.turn   # 0-based

        if self.require_order:
            if not self.state.flags["trio"]:
                mask[0] = 1.0  # TRIO만 허용
            elif not self.state.flags["zero"]:
                mask[1] = 1.0  # ZERO만 허용
            elif not self.state.flags["pill"]:
                mask[2] = 1.0  # PILL만 허용
            else:
                mask[3] = 1.0  # ATK만 허용
            return mask

    # ----------- 헬퍼 -----------
    def _obs(self) -> np.ndarray:
        s = self.state
        assert s is not None
        # LP를 대략 구간화(1000 단위), 불리언은 0/1, used는 4비트
        lp_bucket = min(s.opp_lp // 1000, 4)
        used_bits = [1 if i in s.used else 0 for i in range(self.max_hand)]
        return np.array(
            [lp_bucket, int(s.trio), int(s.zero), int(s.pill)] + used_bits,
            dtype=np.int32
        )

    def _terminal_reward(self) -> float:
        s = self.state
        assert s is not None
        return 1.0 if s.opp_lp == 0 else 0.0

    def _last_used_is_atk(self) -> bool:
        s = self.state
        if s is None or len(s.used) == 0:
            return False
        last_idx = s.used[-1]
        return self.cards[last_idx] == "ATK"

    def _unused_indices(self) -> List[int]:
        s = self.state
        return [i for i in range(self.max_hand) if i not in s.used]

    # 사람이 보기 좋은 상태/트레이스 출력
    def render(self) -> str:
        assert self.state is not None
        s = self.state
        used_ops = [self.cards[i] for i in s.used]
        return (
            f"Turn: {s.turn}  OppLP: {s.opp_lp}\n"
            f"Flags: TRIO={s.trio}, ZERO={s.zero}, PILL={s.pill}\n"
            f"Used: {s.used} ({used_ops})\n"
        )
