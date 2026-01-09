# eval.py
from __future__ import annotations
from ojama_env import OjamaEnv

def run_once(env: OjamaEnv):
    obs, info = env.reset(seed=42)
    # 정답 시퀀스
    seq = [0, 1, 2, 3, env.stop_index]
    names = [env.state.hand[i].script if i < env.max_hand else "STOP" for i in seq[:-1]]  # type: ignore
    for a in seq:
        obs, r, done, info = env.step(a)
        if done:
            break
    s = env.state
    print(f"[Deterministic] idx: {seq[:-1]}  ops: {['TRIO','ZERO','PILL','ATK']}  final_lp: {s.opp_lp}")  # type: ignore

def main():
    env = OjamaEnv(seed=0)
    run_once(env)

if __name__ == "__main__":
    main()
