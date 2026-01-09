# train_sarsa.py
from __future__ import annotations
import os, logging
from logging.handlers import RotatingFileHandler
import numpy as np
from typing import Tuple
from ojama_env import OjamaEnv, CARD_OPS, STOP_IDX
from sarsa_agent import SarsaAgent

# ------------------- 로깅 세팅 -------------------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("ojama")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    fh = RotatingFileHandler("logs/train.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
logger.propagate = False

# 풀이 trace 전용 파일 로거
solve_logger = logging.getLogger("ojama.solve")
solve_logger.setLevel(logging.INFO)
if not solve_logger.handlers:
    fh2 = RotatingFileHandler("logs/solve.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh2.setLevel(logging.INFO)
    fh2.setFormatter(logging.Formatter("%(asctime)s [TRACE] %(message)s"))
    solve_logger.addHandler(fh2)
solve_logger.propagate = False

# ------------------- Teacher demo -------------------
TEACHER_SEQ = [0, 1, 2, 3]  # TRIO, ZERO, PILL, ATK

def teacher_rollout(env: OjamaEnv) -> Tuple[float, bool]:
    """
    고정 정답 시퀀스를 1회 재현하여 Q 업데이트를 도와주는 warmstart용.
    """
    obs, info = env.reset()
    total_r = 0.0
    done = False

    # 4장 순차 사용
    for a in TEACHER_SEQ:
        mask = info["action_mask"]
        obs2, r, done, info = env.step(a)
        total_r += r
        obs = obs2
        if done:
            break

    # 조건 맞으면 STOP
    if not done:
        obs2, r, done, info = env.step(STOP_IDX)
        total_r += r
        obs = obs2

    # 트레이스 로그 남기기
    solve_logger.info("TEACHER TRACE START")
    for t in env.trace:
        solve_logger.info(f"  t={t['t']:02d}  act={t['action']:<14}  LP:{t['lp_before']}→{t['lp_after']}  flags={t['flags']}")
    solve_logger.info("TEACHER TRACE END\n")
    return total_r, done

def warmstart(agent: SarsaAgent, env: OjamaEnv, demos=200, seed=2025):
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        done = False

        # 4장 순차 사용
        for i, a in enumerate(TEACHER_SEQ):
            mask = info["action_mask"]
            a2 = TEACHER_SEQ[i + 1] if (i + 1 < len(TEACHER_SEQ)) else STOP_IDX
            obs2, r, done, info = env.step(a)
            agent.update(obs, a, r, obs2, a2, done)
            obs = obs2
            if done:
                break

        if not done:
            obs2, r, done, info = env.step(STOP_IDX)
            agent.update(obs, STOP_IDX, r, obs2, STOP_IDX, done)
        if done:
            ok += 1
    logger.info(f"[warmstart] teacher demos={demos} ok={ok}")

# ------------------- 학습/평가 유틸 -------------------
def success_rate(env: OjamaEnv, agent: SarsaAgent, rollouts=50, max_steps=6, seed=7) -> float:
    rng = np.random.RandomState(seed)
    succ = 0
    for _ in range(rollouts):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        for _ in range(max_steps):
            a = agent.select_action(obs, info["action_mask"])
            obs2, r, done, info = env.step(a)
            obs = obs2
            if done:
                succ += int(env.state.opp_lp == 0)  # 성공: LP=0
                break
    return succ / rollouts

def log_one_trace(env: OjamaEnv, header="TRACE"):
    solve_logger.info(f"{header} START")
    solve_logger.info(env.render().rstrip())
    for t in env.trace:
        solve_logger.info(f"  t={t['t']:02d}  act={t['action']:<14}  LP:{t['lp_before']}→{t['lp_after']}  flags={t['flags']}")
    solve_logger.info(f"{header} END\n")

# ------------------- 메인 -------------------
def main():
    env = OjamaEnv(seed=0)
    agent = SarsaAgent(n_actions=env.max_hand + 1, eps=0.2, alpha=0.25, gamma=0.9, seed=0)

    # 교사 예제 1회(로그용)
    teacher_rollout(env)

    # 워름스타트
    warmstart(agent, env, demos=200, seed=2025)

    # 학습
    episodes = 2000
    max_steps = 6
    rng = np.random.RandomState(1234)
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        a = agent.select_action(obs, info["action_mask"])
        total_r = 0.0

        for _ in range(max_steps):
            obs2, r, done, info = env.step(a)
            a2 = agent.select_action(obs2, info["action_mask"])
            agent.update(obs, a, r, obs2, a2, done)
            total_r += r
            obs, a = obs2, a2
            if done:
                break

        if ep % 200 == 0:
            sr = success_rate(env, agent, rollouts=50, max_steps=max_steps, seed=ep)
            logger.info(f"[Ep {ep}] eps={agent.eps:.3f} success={sr:.2f}")
            # 대표 1에피소드 trace 로깅
            if len(env.trace) > 0:
                log_one_trace(env, header=f"TRAIN EP{ep}")

            # 약간씩 epsilon 감소(원하면)
            agent.eps = max(0.05, agent.eps * 0.99)

    # 최종 평가(좀 더 강건)
    sr = success_rate(env, agent, rollouts=200, max_steps=max_steps, seed=999)
    logger.info(f"[Eval-Robust] success={sr:.2f}")

    # 최종 Q 저장(선택)
    os.makedirs("ckpt", exist_ok=True)
    # 간단 저장: key들을 문자열로 직렬화
    try:
        import pickle
        with open("ckpt/q_table.pkl", "wb") as f:
            pickle.dump(agent.Q, f)
    except Exception:
        pass

if __name__ == "__main__":
    main()
