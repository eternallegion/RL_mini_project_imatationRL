# dump_policy_trace.py
import os, json, argparse
import numpy as np

from puzzle_rl_env import CardPuzzleEnv
from train_sarsa_masked import LinearSARSASA, extract_card_feats

def load_puzzles(path="puzzles.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_env_for(puzzles, name):
    p = next(p for p in puzzles if p["name"] == name)
    # 학습에서 쓴 환경 파라미터와 동일하게
    return CardPuzzleEnv(p, max_hand=8, length_penalty=0.004, progress_coef=1.0)

def load_agent(checkpoint, probe_env):
    # card_feat_dim = 8 (|kinds|(6) + value(1) + stop_flag(1))
    agent = LinearSARSASA(
        base_obs_dim=probe_env.observation_size,
        act_dim=probe_env.action_size,
        card_feat_dim=8,
        alpha=3e-3, gamma=0.99, eps=0.05  # 평가 시 eps 낮춤
    )
    agent.load(checkpoint)
    return agent

def run_one(env, agent, seed=42, max_steps=16):
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    card_feats = extract_card_feats(env)

    actions = []
    total_r = 0.0
    STOP = env.stop_index

    for t in range(max_steps):
        a = agent.select_action(obs, mask, card_feats)
        actions.append(int(a) if a != STOP else "STOP")
        obs2, r, done, info = env.step(a)
        total_r += float(r)
        if done:
            return True, actions, total_r
        obs, mask, card_feats = obs2, info["action_mask"], extract_card_feats(env)
    return False, actions, total_r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=str, default="puzzles.json")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="예: checkpoints/sarsa_stage4.npz")
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--outdir", type=str, default="replays")
    args = ap.parse_args()

    puzzles = load_puzzles(args.puzzles)
    env = make_env_for(puzzles, args.name)

    # probe env로 agent 초기화
    agent = load_agent(args.checkpoint, env)

    ok, seq, R = run_one(env, agent, seed=args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{args.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"puzzle": args.name, "seed": args.seed, "reward": R, "actions": seq}, f, ensure_ascii=False, indent=2)
    print(f"[{'OK' if ok else 'NG'}] wrote {out_path} | reward={R:.3f} | seq={seq}")

if __name__ == "__main__":
    main()
