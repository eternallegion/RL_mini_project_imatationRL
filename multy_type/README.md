# Ojama RL Multi-Puzzle
- Train: `python multy_type/train_ppo.py`
- Eval:  `python multy_type/eval.py`  → `exports/trajectories.jsonl`에 액션 시퀀스 저장
- Puzzles: `puzzles/*.json` (템플릿 아래)

## Puzzle JSON 템플릿
{
  "name": "sample_puzzle",
  "opp_lp_init": 4000,
  "my_lp_init": 4000,
  "hand": ["ZERO","TRIO","PILL","ATK","ATK"],
  "atk_value": 2000,
  "constraints": {
    "required_order": ["ZERO","TRIO","PILL","ATK"],
    "forbid_kinds": [],
    "limit_kind_counts": {"ATK": 2},
    "must_break_before_first_atk": true
  }
}
