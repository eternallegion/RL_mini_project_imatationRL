# eval.py
from __future__ import annotations
import numpy as np
from ojama_env import OjamaEnv
from agent_sarsa import SarsaAgent

def main():
    env = OjamaEnv(seed=7, require_order=True, randomize_lp=False)

    # 비어있는 Q라도 require_order 때문에 동일한 시퀀스가 강제됨
    agent = SarsaAgent(action_dim=4, eps=0.0, eps_min=0.0)
    # 가능한 경우: 저장된 ckpt 로딩(선택)
    try:
        keys = np.load("ckpt/q_table_keys.npy", allow_pickle=True)
        vals = np.load("ckpt/q_table_vals.npy", allow_pickle=True)
        agent.Q = {tuple(k): float(v) for k, v in zip(keys, vals)}
    except Exception:
        pass

    # 결정적 시퀀스 확인
    obs, info = env.reset(seed=7)
    idx_seq = []
    ops = []
    for _ in range(4):
        a = agent.select_action_greedy(obs, info["action_mask"])
        idx_seq.append(a)
        ops.append(["TRIO", "ZERO", "PILL", "ATK"][a])
        obs, r, done, info = env.step(a)
        if done:
            break
    print(f"[Deterministic] idx: {idx_seq}  ops: {ops}  final_lp: {env.state.opp_lp}")

    # 약간 랜덤 LP에서도 동일 동작
    env.randomize_lp = True
    obs, info = env.reset(seed=17)
    idx_seq = []
    ops = []
    for _ in range(4):
        a = agent.select_action_greedy(obs, info["action_mask"])
        idx_seq.append(a)
        ops.append(["TRIO", "ZERO", "PILL", "ATK"][a])
        obs, r, done, info = env.step(a)
        if done:
            break
    print(f"[Random-LP]    idx: {idx_seq}  ops: {ops}  final_lp: {env.state.opp_lp}")

if __name__ == "__main__":
    main()
