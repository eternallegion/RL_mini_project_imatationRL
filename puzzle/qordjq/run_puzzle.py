import json, argparse, os, csv
from typing import Dict, List
from puzzle_env import make_state_from_dict, step, PuzzleState
from mcts_solver import mcts_plan, filtered_legal_actions, improved_rollout_policy

def check_constraints(seq_names: List[str], seq_kinds: List[str], final_state: PuzzleState, cons: Dict) -> bool:
    if not cons:
        return True
    # 기본 개수
    if "exact_cards" in cons and len(seq_names) != int(cons["exact_cards"]): return False
    if "min_cards"   in cons and len(seq_names) <  int(cons["min_cards"]):   return False
    if "max_cards"   in cons and len(seq_names) >  int(cons["max_cards"]):   return False
    # 종류/이름 필수·금지
    if "must_end_with" in cons:
        if not seq_kinds or seq_kinds[-1] != cons["must_end_with"]: return False
    if "forbid_kinds" in cons and any(k in set(cons["forbid_kinds"]) for k in seq_kinds): return False
    if "require_names" in cons and not set(cons["require_names"]).issubset(set(seq_names)): return False
    if "require_kinds" in cons and not set(cons["require_kinds"]).issubset(set(seq_kinds)): return False
    if "forbid_names"  in cons and any(n in set(cons["forbid_names"]) for n in seq_names):  return False
    # 종류별 상한
    if "limit_kind_counts" in cons:
        from collections import Counter
        c = Counter(seq_kinds)
        for k, mx in cons["limit_kind_counts"].items():
            if c.get(k,0) > int(mx): return False
    # 순서/패턴
    if "require_before" in cons:
        for a, b in cons["require_before"]:
            try:
                ia = seq_kinds.index(a); ib = seq_kinds.index(b)
                if ia >= ib: return False
            except ValueError:
                return False
    if "require_sequence" in cons:
        req = cons["require_sequence"]; it = iter(seq_kinds)
        if not all(any(x==y for x in it) for y in req): return False
    # 남은 손패
    if "min_unused_cards" in cons:
        total = len(final_state.hand); used = len(seq_kinds)
        if (total - used) < int(cons["min_unused_cards"]): return False
    # 첫 ATK 전에 반드시 BLOCK_BREAK
    if cons.get("must_break_before_first_atk"):
        first_atk_idx = next((i for i,k in enumerate(seq_kinds) if k=="ATK"), None)
        first_break_idx = next((i for i,k in enumerate(seq_kinds) if k=="BLOCK_BREAK"), None)
        if first_atk_idx is None or first_break_idx is None or not (first_break_idx < first_atk_idx):
            return False
    # 마지막 일격 데미지 하한
    if "min_last_attack" in cons:
        if not seq_kinds or seq_kinds[-1] != "ATK": return False
        if final_state.last_damage < int(cons["min_last_attack"]): return False
    return True

def solve_puzzle(pzl, sims=800):
    s = make_state_from_dict(pzl)
    constraints = pzl.get("constraints", {})
    name = pzl.get("name", "(no name)")

    print(f"\n=== Puzzle: {name} ===")
    cur = s
    seq_names, seq_kinds = [], []

    max_cards = constraints.get("max_cards", None)
    while (not cur.is_terminal()) and (max_cards is None or len(seq_names) < int(max_cards)):
        # 제약-aware 후보군
        legal = filtered_legal_actions(cur, constraints)
        # MCTS에도 제약 전달
        a, q = mcts_plan(cur, simulations=sims, constraints=constraints)
        # 세이프가드: 만약 a가 필터 밖이면 휴리스틱으로 교정
        if a not in legal:
            a = improved_rollout_policy(cur, legal)

        card = cur.hand[a]
        seq_names.append(card.name); seq_kinds.append(card.kind)
        cur = step(cur, a)
        print(f"pick '{card.name}' (a={a}), Q≈{q:.3f} -> opp_lp={cur.opp_lp}, my_buff={cur.my_buff}, used={cur.used}, last_damage={cur.last_damage}")

    base_success = (cur.opp_lp == 0)
    constraints_ok = check_constraints(seq_names, seq_kinds, cur, constraints)
    success = bool(base_success and constraints_ok)

    print("Result:",
          "SUCCESS" if success else ("FAIL (constraint)" if (base_success and not constraints_ok) else "FAIL"),
          "| Sequence:", " -> ".join(seq_names))

    return {
        "name": name,
        "success": int(success),
        "base_success": int(base_success),
        "constraints_ok": int(constraints_ok),
        "used_cards": len(seq_kinds),
        "remain_opp_lp": cur.opp_lp,
        "last_damage": cur.last_damage,
        "sims": sims,
        "sequence": " -> ".join(seq_names),
        "constraints": json.dumps(constraints, ensure_ascii=False)
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", default="puzzles.json")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--metrics", default="../results/metrics.csv")
    args = ap.parse_args()

    with open(args.puzzles, "r", encoding="utf-8") as f:
        puzzles = json.load(f)

    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)

    rows = []; solved = 0
    for pzl in puzzles:
        res = solve_puzzle(pzl, sims=args.sims)
        rows.append(res); solved += res["success"]

    print(f"\nSolved {solved}/{len(puzzles)} puzzles with {args.sims} simulations each.")

    write_header = not os.path.exists(args.metrics)
    with open(args.metrics, "a", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "name","success","base_success","constraints_ok",
            "used_cards","remain_opp_lp","last_damage","sims","sequence","constraints"
        ])
        if write_header: writer.writeheader()
        for r in rows: writer.writerow(r)
    print(f"Saved metrics -> {args.metrics}")
