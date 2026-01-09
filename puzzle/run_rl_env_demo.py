# run_rl_env_demo.py
import json
from puzzle_rl_env import CardPuzzleEnv

def play_one(pzl, seed=0):
    env = CardPuzzleEnv(pzl, max_hand=8, length_penalty=0.004, progress_coef=1.0)
    obs, info = env.reset(seed=seed)
    total = 0.0
    for _ in range(12):
        mask = info["action_mask"]
        valid = [i for i, m in enumerate(mask) if m > 0.5]
        if not valid:
            break
        a = valid[0]  # 데모: 항상 첫 합법행동
        obs, r, done, info = env.step(int(a))
        total += r
        if done:
            break
    return total, env.state.opp_lp

if __name__ == "__main__":
    with open("puzzles.json","r",encoding="utf-8") as f:
        puzzles = json.load(f)

    names = ["seq_block_buff_kill", "no_heal_cap_atk2", "must_break_before_first_atk"]
    for nm in names:
        pzl = next(p for p in puzzles if p["name"] == nm)
        R, lp = play_one(pzl, seed=42)
        print(f"[{nm}] return={R:.3f}  opp_lp={lp}")
