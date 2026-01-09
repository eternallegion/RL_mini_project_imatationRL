import json
import argparse, json, random
from mcts_solver import solve_puzzle
from puzzle_env import make_state_from_dict, step
from effect_vm import run_effect_vm
import csv, os
from puzzle_env import make_state_from_dict
from mcts_solver import solve_puzzle
from extractor import extract_effect


def _auto_compile_scripts(pzl: dict) -> dict:
    for c in pzl.get("hand", []):
        if c.get("kind", "").upper() == "SCRIPT" and "script" not in c:
            prog = extract_effect(c.get("text", ""))
            if prog:
                c["script"] = prog
    for c in pzl.get("scripted_draw", []):
        if c.get("kind", "").upper() == "SCRIPT" and "script" not in c:
            prog = extract_effect(c.get("text", ""))
            if prog:
                c["script"] = prog
    return pzl

def replay_with_debug(puzzle, seq_names):
    print("\n[DEBUG REPLAY]")
    state = make_state_from_dict(puzzle)
    used = set()
    for name in seq_names:
        # 같은 이름의 미사용 카드 인덱스 선택
        idx = next(i for i,c in enumerate(state.hand) if c.name == name and i not in used)
        card = state.hand[idx]
        if card.script:
            print(f" -> {card.name} (SCRIPT)")
            state = run_effect_vm(state, card.script, debug=True)
            used.add(idx)
        else:
            print(f" -> {card.name} ({card.kind})")
            state = step(state, idx)
            used.add(idx)
    print(f"[END] opp_lp={state.opp_lp}, my_buff={state.my_buff}, opp_block={state.opp_block}\n")


def _append_metrics(name, sims, seed, success, seq):
    os.makedirs("../results", exist_ok=True)
    path = "../results/metrics.csv"
    newfile = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["puzzle", "sims", "seed", "success", "sequence"])
        w.writerow([name, sims, seed if seed is not None else "", int(success), " | ".join(seq)])


def main():
    import argparse, json, random
    from mcts_solver import solve_puzzle
    from puzzle_env import make_state_from_dict, step
    from effect_vm import run_effect_vm


    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
#    parser.add_argument("--seed", type=int, default=None)
#    args = parser.parse_args()

    with open("puzzles.json","r",encoding="utf-8") as f:
        puzzles = json.load(f)

    ok = 0
    puzzles = [_auto_compile_scripts(p) for p in puzzles]

    success = 0
    for pzl in puzzles:
        print(f"\n=== Puzzle: {pzl['name']} ===")
        success, seq = solve_puzzle(pzl, sims=args.sims, seed=args.seed)
        result = "SUCCESS" if success else "FAIL"
        print(f"Result: {result} | Sequence: " + " -> ".join(seq) if seq else "(no sequence)")
        if args.debug and seq:
            replay_with_debug(pzl, seq)
        if success: ok += 1
    print(f"\nSolved {ok}/{len(puzzles)} puzzles with {args.sims} simulations each.")
if __name__ == "__main__":
    main()
