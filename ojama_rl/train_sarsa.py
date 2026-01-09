# train_sarsa.py
from __future__ import annotations
import os
import numpy as np
import logging

from ojama_env import OjamaEnv
from agent_sarsa import SarsaAgent
from utils import setup_root_logger

def warmstart(agent: SarsaAgent, env: OjamaEnv, demos: int = 200, seed: int = 2025):
    """
    교사 시연: 올바른 인덱스 시퀀스 [0,1,2,3] 으로 SARSA 업데이트
    """
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        seq = [0, 1, 2, 3]  # TRIO, ZERO, PILL, ATK
        # 첫 a
        mask = info["action_mask"]
        a = seq[0]
        for t in range(len(seq)):
            obs2, r, done, info2 = env.step(a)
            if t + 1 < len(seq) and not done:
                a2 = seq[t + 1]
            else:
                a2 = a
            agent.update(obs, a, r, obs2, a2, done)
            obs, a, info = obs2, a2, info2
            if done:
                break
        if env.state.opp_lp == 0:
            ok += 1
    print(f"[warmstart] teacher demos={demos} ok={ok}")

def eval_success(env: OjamaEnv, agent: SarsaAgent, rollouts: int = 100) -> float:
    ok = 0
    for _ in range(rollouts):
        obs, info = env.reset()
        for _ in range(4):
            mask = info["action_mask"]
            a = agent.select_action_greedy(obs, mask)  # ε=0
            obs, r, done, info = env.step(a)
            if done:
                break
        ok += int(env.state.opp_lp == 0)
    return ok / rollouts

def main():
    setup_root_logger()
    logger = logging.getLogger(__name__)

    # 환경(순서 강제 + LP 랜덤평가는 False)
    env = OjamaEnv(seed=2025, require_order=True, randomize_lp=False)

    agent = SarsaAgent(action_dim=4, eps=0.20, eps_min=0.05, gamma=0.98, alpha=0.12)

    # 워ーム스타트
    warmstart(agent, env, demos=200, seed=2025)

    # 학습
    episodes = 2000
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=2025 + ep)
        mask = info["action_mask"]
        a = agent.select_action(obs, mask)

        for _ in range(4):
            obs2, r, done, info2 = env.step(a)
            a2 = agent.select_action(obs2, info2["action_mask"])
            agent.update(obs, a, r, obs2, a2, done)
            obs, a, mask, info = obs2, a2, info2["action_mask"], info2
            if done:
                break

        if ep % 200 == 0:
            agent.decay_eps(0.99)
            sr = eval_success(env, agent, rollouts=50)
            print(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")
            logging.info(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")

    # 최종 평가(견고성)
    env.randomize_lp = True  # 약간의 변화를 줘도 성공 확인
    sr = eval_success(env, agent, rollouts=100)
    print(f"[Eval-Robust] success={sr:.2f}")

    # 체크포인트 저장
    os.makedirs("ckpt", exist_ok=True)
    np.save("ckpt/q_table_keys.npy", np.array(list(agent.Q.keys()), dtype=object))
    np.save("ckpt/q_table_vals.npy", np.array(list(agent.Q.values()), dtype=np.float32))

if __name__ == "__main__":
    main()
