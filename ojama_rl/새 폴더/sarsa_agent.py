# sarsa_agent.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Any

def _obs_key(obs: np.ndarray) -> Tuple[int, ...]:
    # 탭룰러 키로 쓰기 쉬운 튜플
    return tuple(int(x) for x in obs.tolist())

class SarsaAgent:
    """
    매우 단순한 탭룰러 SARSA(ε-greedy).
    상태: 관측 obs 튜플
    행동: 0~4 (4는 STOP)
    """
    def __init__(self, n_actions: int, eps: float = 0.2, alpha: float = 0.2, gamma: float = 0.9, seed: int = 0):
        self.nA = n_actions
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.Q: Dict[Tuple[int, ...], np.ndarray] = {}

    def _ensure_Q(self, key: Tuple[int, ...]):
        if key not in self.Q:
            self.Q[key] = np.zeros(self.nA, dtype=np.float32)

    def select_action(self, obs: np.ndarray, mask: np.ndarray | None) -> int:
        key = _obs_key(obs)
        self._ensure_Q(key)

        # 마스크 처리: 불가능한 액션은 -inf 처리
        q = self.Q[key].copy()
        if mask is not None:
            for a in range(self.nA):
                if mask[a] <= 0.0:
                    q[a] = -1e9

        # ε-greedy
        if self.rng.rand() < self.eps:
            legal = np.where((mask > 0.0) if mask is not None else np.ones(self.nA))[0]
            return int(self.rng.choice(legal)) if len(legal) else 0
        return int(np.argmax(q))

    def update(self, obs, a, r, obs2, a2, done):
        k1 = _obs_key(obs)
        k2 = _obs_key(obs2)
        self._ensure_Q(k1)
        self._ensure_Q(k2)
        q1 = self.Q[k1][a]
        target = r if done else (r + self.gamma * self.Q[k2][a2])
        self.Q[k1][a] = q1 + self.alpha * (target - q1)
