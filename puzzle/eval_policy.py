# eval_policy.py
import json, numpy as np
from puzzle_rl_env import CardPuzzleEnv
from train_sarsa_masked import LinearSARSASA, extract_card_feats

def make_env_for(pzl):
    return CardPuzzleEnv(pzl, max_hand=8, length_penalty=0.006, progress_coef=1.0)

def success_rate(env, agent, rollouts=200, max_steps=12, seed=123):
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(rollouts):
        obs, info = env.reset(seed=int(rng.randint(1e9)))
        mask = info["action_mask"]
        for t in range(max_steps):
            feats = extract_card_feats(env)
            a = agent.select_action(obs, mask, feats)
            obs, r, done, info = env.step(a)
            mask = info["action_mask"]
            if done:
                ok += float(env.state.opp_lp <= 0)
                break
    return ok/rollouts

if __name__ == "__main__":
    with open("puzzles.json","r",encoding="utf-8") as f:
        puzzles = json.load(f)

    pzl_map = {p["name"]: p for p in puzzles}
    probe = make_env_for(pzl_map["seq_block_buff_kill"])
    agent = LinearSARSASA(base_obs_dim=probe.observation_size,
                          act_dim=probe.action_size,
                          card_feat_dim=8, alpha=3e-3, gamma=0.99, eps=0.0)  # 평가 때 eps=0
    agent.load("checkpoints/sarsa_stage3.npz")

    for name in ["seq_block_buff_kill","no_heal_cap_atk2","must_break_before_first_atk"]:
        env = make_env_for(pzl_map[name])
        rate = success_rate(env, agent, rollouts=500)
        print(f"{name}: {rate*100:.1f}%")
