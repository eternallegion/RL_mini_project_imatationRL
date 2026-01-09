import math, random
from typing import Optional, Dict, Tuple, List
from collections import Counter
from puzzle_env import PuzzleState, step, terminal_reward, Card

# ---------- 유틸 ----------
def seq_kinds_from_state(state: PuzzleState) -> List[str]:
    return [state.hand[i].kind for i in state.used]

def progress_on_required_subseq(done: List[str], req: List[str]) -> int:
    j = 0
    for k in done:
        if j < len(req) and k == req[j]:
            j += 1
    return j

def state_key(s: PuzzleState) -> tuple:
    # 동일 상태 공유를 위한 캐시 키
    hand_sig = tuple((c.name, c.kind, c.value) for c in s.hand)
    scripted_sig = tuple((c.name, c.kind, c.value) for c in s.scripted_draw)
    return (s.my_lp, s.opp_lp, hand_sig, s.my_buff, s.opp_block, s.used, scripted_sig)

# 트랜스포지션 테이블(상태 캐시): key -> (N, W)
TT: Dict[tuple, Tuple[int, float]] = {}

# ---------- 제약 반영: 합법 액션 필터 ----------
def filtered_legal_actions(state: PuzzleState, constraints: Optional[Dict]) -> List[int]:
    legal = state.legal_actions()
    if not constraints:
        return legal

    kinds_so_far = seq_kinds_from_state(state)

    # (1) 금지/상한
    forbid = set(constraints.get("forbid_kinds", []))
    limits = constraints.get("limit_kind_counts", {})
    cnt = Counter(kinds_so_far)

    # (2) 순서 제약
    must_break_before_first_atk = constraints.get("must_break_before_first_atk", False)
    first_atk_seen = any(k == "ATK" for k in kinds_so_far)
    break_seen     = any(k == "BLOCK_BREAK" for k in kinds_so_far)

    require_before = constraints.get("require_before", [])
    req_seq        = constraints.get("require_sequence", [])
    prefix_len     = progress_on_required_subseq(kinds_so_far, req_seq)
    next_required  = req_seq[prefix_len] if prefix_len < len(req_seq) else None
    later_required_set = set(req_seq[prefix_len+1:]) if prefix_len < len(req_seq) else set()

    # (3) max_cards와 req_seq 연계
    max_cards = constraints.get("max_cards", None)
    remaining_picks = int(max_cards) - len(kinds_so_far) if max_cards is not None else None
    remaining_required = len(req_seq) - prefix_len

    filtered = []
    for a in legal:
        k = state.hand[a].kind

        # 금지 종류
        if k in forbid:
            continue
        # 종류별 상한
        if k in limits and cnt.get(k, 0) + 1 > int(limits[k]):
            continue
        # 첫 ATK 전에 BLOCK_BREAK 필수
        if must_break_before_first_atk and (not first_atk_seen) and (not break_seen) and k == "ATK":
            continue
        # A before B: A 나오기 전 B 금지
        violated = False
        for A, B in require_before:
            if k == B and (A not in kinds_so_far):
                violated = True
                break
        if violated:
            continue
        # require_sequence: 다음 필요요소 이전에 뒤 요소 금지
        if next_required is not None:
            if (k in later_required_set) and (k != next_required):
                continue
        # 남은 픽 == 남은 필수요소 → 필수요소만 선택
        if remaining_picks is not None and remaining_required > 0 and remaining_picks == remaining_required:
            if next_required is not None and k != next_required:
                continue

        filtered.append(a)

    return filtered or legal  # 막다른 길 방지

# ---------- MCTS ----------
class Node:
    def __init__(self, state: PuzzleState, parent: Optional["Node"]=None, action_taken: Optional[int]=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[int, "Node"] = {}
        self.N = 0
        self.W = 0.0
    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N
    def is_fully_expanded(self, constraints: Optional[Dict]) -> bool:
        return set(self.children.keys()) == set(filtered_legal_actions(self.state, constraints))

def ucb(parent_N: int, child: Node, c: float = 1.2) -> float:
    if child.N == 0:
        return float("inf")
    return child.Q() + c * math.sqrt(math.log(parent_N + 1) / child.N)

def select(node: Node, constraints: Optional[Dict]) -> Node:
    while not node.state.is_terminal():
        if not node.is_fully_expanded(constraints):
            return expand(node, constraints)
        node = max(node.children.values(), key=lambda ch: ucb(node.N, ch))
    return node

def expand(node: Node, constraints: Optional[Dict]) -> Node:
    legal = filtered_legal_actions(node.state, constraints)
    untried = [a for a in legal if a not in node.children]
    a = random.choice(untried)
    child_state = step(node.state, a)
    child = Node(child_state, parent=node, action_taken=a)
    # 트랜스포지션 캐시 부트스트랩
    k = state_key(child_state)
    if k in TT:
        N, W = TT[k]
        child.N, child.W = N, W
    node.children[a] = child
    return child

# 롤아웃 정책(간단 휴리스틱)
def improved_rollout_policy(state: PuzzleState, legal: List[int]) -> int:
    hand = state.hand
    # 1수 킬 우선
    for a in legal:
        if terminal_reward(step(state, a)) == 1.0:
            return a
    # 휴리스틱: BLOCK_BREAK > DOUBLE/BUFF > 강 ATK > 약 ATK > HEAL > DRAW
    def score(i: int):
        c: Card = hand[i]
        if c.kind == "BLOCK_BREAK":     return 5
        if c.kind == "DOUBLE_NEXT_ATK": return 4.5
        if c.kind == "BUFF":            return 4 + 0.001*c.value
        if c.kind == "ATK":             return 3 + 0.001*(c.value + state.my_buff)
        if c.kind == "HEAL":            return 2 + 0.001*c.value
        if c.kind == "DRAW":            return 1.5
        return 0
    return max(legal, key=score)

def rollout(state: PuzzleState, constraints: Optional[Dict], max_depth: int = 20) -> float:
    cur = state; depth = 0
    while (not cur.is_terminal()) and depth < max_depth:
        legal = filtered_legal_actions(cur, constraints)
        a = improved_rollout_policy(cur, legal)
        cur = step(cur, a); depth += 1

    val = terminal_reward(cur)
    if constraints and val > 0.0:
        # must_end_with 위반 시 0
        if "must_end_with" in constraints:
            seqk = seq_kinds_from_state(cur)
            if not seqk or seqk[-1] != constraints["must_end_with"]:
                val = 0.0
        # require_sequence 전체 충족 확인
        if "require_sequence" in constraints:
            req = constraints["require_sequence"]
            if progress_on_required_subseq(seq_kinds_from_state(cur), req) < len(req):
                val = 0.0
    return val

def backpropagate(node: Node, value: float):
    while node is not None:
        node.N += 1; node.W += value
        # 상태 캐시에 누적
        k = state_key(node.state)
        N, W = TT.get(k, (0, 0.0))
        TT[k] = (N + 1, W + value)
        node = node.parent

def mcts_plan(root_state: PuzzleState, simulations: int = 1000, constraints: Optional[Dict] = None) -> Tuple[int, float]:
    root = Node(root_state)
    for _ in range(simulations):
        leaf = select(root, constraints)
        v = rollout(leaf.state, constraints)
        backpropagate(leaf, v)
    if not root.children:
        return -1, 0.0
    best = max(root.children.values(), key=lambda ch: ch.N)
    return best.action_taken, best.Q()

