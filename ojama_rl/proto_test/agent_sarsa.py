# agent_sarsa.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

class SarsaAgent:
    def __init__(self, n_features: int, n_actions: int, eps: float = 0.2, alpha: float = 0.2, gamma: float = 0.95):
        self.n_features = n_features
        self.n_actions = n_actions
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        # 테이블: (discrete_state_tuple) -> Q[a]
        self.Q: Dict[Tuple[int, ...], np.ndarray] = {}

    def _discretize(self, obs: np.ndarray) -> Tuple[int, ...]:
        """
        관측을 간단히 이산화:
          - 앞 4개(used mask)는 round
          - 버프는 0/1(있음) 플래그
          - opp_lp/1000은 floor
          - stack_len은 int
        """
        used = tuple(int(round(x)) for x in obs[:4])
        buff_flag = int(obs[4] > 0.5)
        lp_bin = int(np.floor(obs[5]))  # 4.0 -> 4, 3.8->3 등
        stack_len = int(round(obs[6]))
        return (*used, buff_flag, lp_bin, stack_len)

    def _ensure(self, key: Tuple[int, ...]):
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions, dtype=np.float32)

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        key = self._discretize(obs)
        self._ensure(key)

        # 마스크된 액션만 후보
        legal = np.flatnonzero(mask > 0.0)
        if len(legal) == 0:
            return 0  # fallback

        if np.random.rand() < self.eps:
            return int(np.random.choice(legal))
        # greedy on masked Q
        q = self.Q[key].copy()
        q[mask == 0.0] = -1e9
        return int(np.argmax(q))

    def greedy_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        eps_bak = self.eps
        self.eps = 0.0
        a = self.select_action(obs, mask)
        self.eps = eps_bak
        return a

    def update(self, obs, a, r, obs2, a2, done):
        k1 = self._discretize(obs)
        k2 = self._discretize(obs2)
        self._ensure(k1)
        self._ensure(k2)
        q1 = self.Q[k1][a]
        target = r if done else r + self.gamma * self.Q[k2][a2]
        self.Q[k1][a] += self.alpha * (target - q1)
