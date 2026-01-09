# eval.py
from __future__ import annotations
from ojama_env import OjamaEnv, CARD_OPS, STOP_IDX
from sarsa_agent import SarsaAgent

def eval_deterministic():
    env = OjamaEnv(seed=42)
    # 더미 에이전트(안 씀). 시퀀스 고정 실행.
    seq = [0, 1, 2, 3, STOP_IDX]
    obs, info = env.reset()
    for a in seq:
        obs, r, done, info = env.step(a)
        if done:
            break

    used_ops = [CARD_OPS[i] for i in env.state.used] if env.state else []
    print(f"[Deterministic] idx: {env.state.used}  ops: {used_ops}  final_lp: {env.state.opp_lp}")

def eval_random_lp():
    # 여기선 랜덤 요소가 거의 없지만, seed만 다르게 해서 확인
    env = OjamaEnv(seed=7)
    seq = [0, 1, 2, 3, STOP_IDX]
    obs, info = env.reset(seed=123)
    for a in seq:
        obs, r, done, info = env.step(a)
        if done:
            break
    used_ops = [CARD_OPS[i] for i in env.state.used] if env.state else []
    print(f"[Random-LP]    idx: {env.state.used}  ops: {used_ops}  final_lp: {env.state.opp_lp}")

if __name__ == "__main__":
    eval_deterministic()
    eval_random_lp()
