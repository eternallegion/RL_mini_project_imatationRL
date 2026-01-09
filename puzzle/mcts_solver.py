# -*- coding: utf-8 -*-
import math, random
from typing import Any, Dict, List, Optional, Tuple

from puzzle_env import PuzzleState, make_state_from_dict, step

# ----------------------------
# 액션 이름 복원 (드로우로 인덱스 변동 대비)
# ----------------------------
def _seq_names_from_actions(root_state: PuzzleState, seq_actions: List[int]) -> List[str]:
    names: List[str] = []
    s = root_state
    for a in seq_actions:
        if a < 0 or a >= len(s.hand):
            break
        names.append(s.hand[a].name)
        s = step(s, a)
    return names

# ----------------------------
# 퍼즐 제약 관련 유틸
# ----------------------------
def _hand_kinds(puzzle: Dict[str, Any]) -> List[str]:
    return [c.get("kind", "") for c in puzzle.get("hand", [])]

def _action_kind_at(puzzle: Dict[str, Any], idx: int) -> str:
    kinds = _hand_kinds(puzzle)
    return kinds[idx] if 0 <= idx < len(kinds) else ""

def _seq_kinds(puzzle: Dict[str, Any], seq: List[int]) -> List[str]:
    kinds = _hand_kinds(puzzle)
    return [kinds[i] for i in seq if 0 <= i < len(kinds)]

def _count_kinds(puzzle: Dict[str, Any], seq: List[int]) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for k in _seq_kinds(puzzle, seq):
        cnt[k] = cnt.get(k, 0) + 1
    return cnt

def _prefix_ok(require_seq: List[str], kinds_so_far: List[str]) -> bool:
    L = len(kinds_so_far)
    return kinds_so_far == require_seq[:L]

def _constraints_ok(puzzle: Dict[str, Any], state: PuzzleState, seq: List[int], final: bool=False) -> bool:
    cons = puzzle.get("constraints", {}) or {}
    kinds_so_far = _seq_kinds(puzzle, seq)
    counts = _count_kinds(puzzle, seq)

    # forbid_kinds
    forbid = cons.get("forbid_kinds", [])
    if any(counts.get(fk, 0) > 0 for fk in forbid):
        return False

    # limit_kind_counts
    lim = cons.get("limit_kind_counts", {})
    for k, m in lim.items():
        if counts.get(k, 0) > int(m):
            return False

    # must_break_before_first_atk
    if cons.get("must_break_before_first_atk", False):
        if "ATK" in kinds_so_far:
            first = kinds_so_far.index("ATK")
            if "BLOCK_BREAK" not in kinds_so_far[:first]:
                return False
        # 추가: 현재 보드에 블록이 남아있고 아직 BLOCK_BREAK를 안 썼다면
        # 첫 ATK는 허용하지 않음 (선택 단계에서 막기용)
        if state.opp_block:
            if len(kinds_so_far) > 0 and kinds_so_far[-1] == "ATK":
                if "BLOCK_BREAK" not in kinds_so_far:
                    return False

    # require_sequence (prefix 강제)
    req = cons.get("require_sequence", [])
    if req and (not _prefix_ok(req, kinds_so_far)):
        return False

    # max_cards
    mc = cons.get("max_cards", None)
    if mc is not None and len(seq) > int(mc):
        return False

    # final 전용: must_end_with
    if final and cons.get("must_end_with"):
        if len(kinds_so_far) == 0 or kinds_so_far[-1] != cons["must_end_with"]:
            return False

    return True

def _filter_legal_actions_by_constraints(puzzle: Dict[str, Any], state: PuzzleState, seq: List[int]) -> List[int]:
    base = state.legal_actions()
    filtered: List[int] = []
    for a in base:
        trial = seq + [a]
        if not _constraints_ok(puzzle, state, trial, final=False):
            continue
        # 특수: 블록이 있고 아직 BLOCK_BREAK를 쓰지 않았으면 첫 ATK 금지
        if state.opp_block and _action_kind_at(puzzle, a) == "ATK":
            if "BLOCK_BREAK" not in _seq_kinds(puzzle, seq):
                continue
        filtered.append(a)
    return filtered

def _is_success_terminal(puzzle: Dict[str, Any], state: PuzzleState, seq: List[int]) -> bool:
    if state.opp_lp != 0:
        return False
    return _constraints_ok(puzzle, state, seq, final=True)

def _is_fail_terminal(puzzle: Dict[str, Any], state: PuzzleState, seq: List[int]) -> bool:
    if state.opp_lp == 0:
        return False
    # 손패 소진 또는 max_cards 도달로 더 진행 못하면 실패
    cons = puzzle.get("constraints", {}) or {}
    mc = cons.get("max_cards", None)
    if state.is_terminal():
        return True
    if mc is not None and len(seq) >= int(mc):
        return True
    # 더 이상 제약을 만족하며 고를 수 있는 액션이 없으면 실패
    if not _filter_legal_actions_by_constraints(puzzle, state, seq):
        return True
    return False

# ----------------------------
# MCTS 노드
# ----------------------------
class Node:
    def __init__(self, puzzle: Dict[str, Any], state: PuzzleState, parent=None, action_from_parent: Optional[int]=None, seq: Optional[List[int]]=None):
        self.puzzle = puzzle
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.seq = list(seq) if seq is not None else []
        self.children: List["Node"] = []
        self._untried: Optional[List[int]] = None
        self.visits = 0
        self.value = 0.0

    def untried_actions(self) -> List[int]:
        if self._untried is None:
            self._untried = _filter_legal_actions_by_constraints(self.puzzle, self.state, self.seq)
        return self._untried

    def uct_select_child(self, c_param: float=1.41421356) -> "Node":
        best, best_score = None, -1e9
        for ch in self.children:
            if ch.visits == 0:
                score = 1e9
            else:
                q = ch.value / ch.visits
                score = q + c_param * math.sqrt(math.log(self.visits + 1e-9) / ch.visits)
            if score > best_score:
                best, best_score = ch, score
        return best

    def expand(self) -> Optional["Node"]:
        ua = self.untried_actions()
        if not ua:
            return None
        a = random.choice(ua)
        new_state = step(self.state, a)
        child = Node(self.puzzle, new_state, parent=self, action_from_parent=a, seq=self.seq + [a])
        self.children.append(child)
        ua.remove(a)
        return child

    def update(self, reward: float):
        self.visits += 1
        self.value += reward

    def is_terminal_node(self) -> bool:
        return _is_success_terminal(self.puzzle, self.state, self.seq) or _is_fail_terminal(self.puzzle, self.state, self.seq)

# ----------------------------
# 롤아웃 정책
# ----------------------------
def _rollout(puzzle: Dict[str, Any], state: PuzzleState, seq: List[int], rollout_limit: int=20) -> float:
    s = state
    tr = list(seq)
    # 즉시 판정
    if _is_success_terminal(puzzle, s, tr): return 1.0
    if _is_fail_terminal(puzzle, s, tr):    return 0.0

    steps = 0
    while steps < rollout_limit:
        acts = _filter_legal_actions_by_constraints(puzzle, s, tr)
        if not acts:
            break
        a = random.choice(acts)
        s = step(s, a)
        tr.append(a)

        if _is_success_terminal(puzzle, s, tr): return 1.0
        if _is_fail_terminal(puzzle, s, tr):    return 0.0
        steps += 1

    # 부분 보상: LP 진행도 - 길이 패널티
    init_lp = puzzle.get("opp_lp", 4000)
    progress = max(0.0, (init_lp - s.opp_lp) / max(1, init_lp))
    penalty = 0.01 * len(tr)
    return max(0.0, progress - penalty)

# ----------------------------
# 퍼즐 해결
# ----------------------------
def solve_puzzle(puzzle: Dict[str, Any], sims: int=400, seed: Optional[int]=None) -> Tuple[bool, List[str]]:
    if seed is not None:
        random.seed(seed)

    root_state = make_state_from_dict(puzzle)
    root = Node(puzzle, root_state, None, None, [])

    for _ in range(max(1, sims)):
        # Selection
        node = root
        while (not node.is_terminal_node()) and (not node.untried_actions() and node.children):
            node = node.uct_select_child()

        # Expansion
        if (not node.is_terminal_node()) and node.untried_actions():
            node = node.expand() or node

        # Rollout
        reward = _rollout(puzzle, node.state, node.seq, rollout_limit=20)

        # Backprop
        while node is not None:
            node.update(reward)
            node = node.parent

    # 루트에서 greedy로 최상 자식 따라가며 시퀀스 뽑기
    seq_actions: List[int] = []
    cur = root
    while cur.children:
        cur = max(cur.children, key=lambda ch: 0.0 if ch.visits == 0 else ch.value / ch.visits)
        if cur.action_from_parent is not None:
            seq_actions.append(cur.action_from_parent)
        if _is_success_terminal(puzzle, cur.state, cur.seq):
            break

    success = _is_success_terminal(puzzle, cur.state, cur.seq)
    seq_names = _seq_names_from_actions(root_state, seq_actions)
    return success, seq_names
