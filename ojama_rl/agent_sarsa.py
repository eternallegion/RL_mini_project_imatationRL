# agent_sarsa.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

class SarsaAgent:
    def __init__(self, action_dim: int, eps: float = 0.2, eps_min: float = 0.05, gamma: float = 0.98, alpha: float = 0.15):
        self.action_dim = action_dim
        self.eps = eps
        self.eps_min = eps_min
        self.gamma = gamma
        self.alpha = alpha
        self.Q: Dict[Tuple[bytes, int], float] = {}

    def _q(self, s_bytes: bytes, a: int) -> float:
        return self.Q.get((s_bytes, a), 0.0)

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.where(mask > 0.5)[0]
        if len(legal) == 0:
            return 0
        if np.random.rand() < self.eps:
            return int(np.random.choice(legal))
        qvals = np.array([self._q(obs.tobytes(), int(a)) for a in legal])
        return int(legal[int(np.argmax(qvals))])

    def select_action_greedy(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.where(mask > 0.5)[0]
        if len(legal) == 0:
            return 0
        qvals = np.array([self._q(obs.tobytes(), int(a)) for a in legal])
        return int(legal[int(np.argmax(qvals))])

    def update(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, a2: int, done: bool):
        s_b = s.tobytes()
        s2_b = s2.tobytes()
        q = self._q(s_b, a)
        target = r if done else (r + self.gamma * self._q(s2_b, a2))
        self.Q[(s_b, a)] = q + self.alpha * (target - q)

    def decay_eps(self, rate: float = 0.995):
        self.eps = max(self.eps_min, self.eps * rate)
