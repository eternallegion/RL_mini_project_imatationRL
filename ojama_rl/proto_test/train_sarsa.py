from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Any, List
from ojama_env import OjamaEnv
from torch.utils.tensorboard import SummaryWriter
import logging, os



def make_env() -> OjamaEnv:
    return OjamaEnv(seed=0)


# 간단한 SARSA 에이전트(탭룰 Q)
class SarsaAgent:
    def __init__(self, n_actions: int, eps: float = 0.2, alpha: float = 0.3, gamma: float = 0.95):
        self.n_actions = n_actions
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.Q: Dict[Tuple[Any, ...], np.ndarray] = {}

    def _ensure(self, key: Tuple[Any, ...]):
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions, dtype=np.float32)

    def select_action(self, key: Tuple[Any, ...], mask: np.ndarray, rng: np.random.RandomState) -> int:
        self._ensure(key)
        q = self.Q[key].copy()

        # 불법 액션 -inf 처리
        illegal = (mask < 0.5)
        q[illegal] = -1e9

        if rng.rand() < self.eps:
            # 무작위 합법 액션 중 선택
            legal_idx = np.where(mask > 0.5)[0]
            return int(rng.choice(legal_idx))
        else:
            # 최대 Q
            return int(np.argmax(q))

    def update(self, key, a, r, key2, a2, done):
        self._ensure(key)
        self._ensure(key2)

        q_sa = self.Q[key][a]
        if done:
            target = r
        else:
            target = r + self.gamma * self.Q[key2][a2]
        self.Q[key][a] += self.alpha * (target - q_sa)


# 교사 데모: 정답 인덱스 [0,1,2,3]
TEACHER_SEQ = [0, 1, 2, 3]


def state_key(env: OjamaEnv) -> Tuple[Any, ...]:
    s = env.state
    assert s is not None
    # 퍼즐이 완전 결정적이라 used 시퀀스만으로 충분
    return (tuple(s.used), s.opp_lp, s.my_buff)


def warmstart(agent: SarsaAgent, env: OjamaEnv, demos: int = 200, seed: int = 2025):
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        mask = info["action_mask"]
        k = state_key(env)
        # 교사 시퀀스 실행
        for i, a in enumerate(TEACHER_SEQ + [env.stop_index]):  # 마지막 STOP까지
            # 다음 상태 예측 위해 행동
            obs2, r, done, info2 = env.step(a)
            k2 = state_key(env)
            mask2 = info2["action_mask"]
            # 다음 행동도 교사
            if not done:
                if i + 1 < len(TEACHER_SEQ) + 1:
                    a2 = TEACHER_SEQ[i + 1] if (i + 1) < len(TEACHER_SEQ) else env.stop_index
                else:
                    a2 = env.stop_index
            else:
                a2 = env.stop_index

            agent.update(k, a, r, k2, a2, done)
            k, mask = k2, mask2
            if done:
                break
        # 성공 여부 집계
        if env.state is not None and env.state.opp_lp == 0:
            ok += 1
    print(f"[warmstart] teacher demos={demos} ok={ok}")


def eval_success(env: OjamaEnv, agent: SarsaAgent, rollouts: int = 40, seed: int = 7) -> float:
    rng = np.random.RandomState(seed)
    success = 0
    for _ in range(rollouts):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        mask = info["action_mask"]
        k = state_key(env)
        # 결정적 실행(탐험 끔)
        old_eps = agent.eps
        agent.eps = 0.0
        for _step in range(8):
            a = agent.select_action(k, mask, rng)
            obs2, r, done, info2 = env.step(a)
            k = state_key(env)
            mask = info2["action_mask"]
            if done:
                if env.state is not None and env.state.opp_lp == 0:
                    success += 1
                break
        agent.eps = old_eps
    return success / rollouts


def main():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/train.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    rng = np.random.RandomState(1234)
    env = make_env()
    agent = SarsaAgent(n_actions=env.max_hand + 1, eps=0.2, alpha=0.3, gamma=0.95)
    writer = SummaryWriter(log_dir="runs/ojama_train")
    # 1) 교사 데모로 워밍업
    warmstart(agent, env, demos=200, seed=2025)

    # 2) 자체 학습
    episodes = 2000
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        mask = info["action_mask"]
        k = state_key(env)
        a = agent.select_action(k, mask, rng)

        for _ in range(8):
            obs2, r, done, info2 = env.step(a)
            k2 = state_key(env)
            mask2 = info2["action_mask"]
            a2 = agent.select_action(k2, mask2, rng)
            agent.update(k, a, r, k2, a2, done)

            k, a, mask = k2, a2, mask2
            if done:
                break

        # 약간씩 탐험률 감소(너무 낮추지는 않음)
#        if ep % 200 == 0:
#            writer.add_scalar("success_rate", sr, ep)
#            writer.add_scalar("epsilon", agent.eps, ep)
#            agent.eps = max(0.05, agent.eps * 0.98)
#            sr = eval_success(env, agent, rollouts=40, seed=42)
#            print(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")
#            print(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")
#            logging.info(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")
        if ep % 200 == 0:
            agent.eps = max(0.05, agent.eps * 0.98)
            sr = eval_success(env, agent, rollouts=40, seed=42)
            print(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")
            writer.add_scalar("success_rate", sr, ep)
            writer.add_scalar("epsilon", agent.eps, ep)
            print(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")
            logging.info(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")

    # 최종 평가(강건성)
    sr = eval_success(env, agent, rollouts=40, seed=2026)
    print(f"[Eval-Robust] success={sr:.2f}")


if __name__ == "__main__":
    main()
