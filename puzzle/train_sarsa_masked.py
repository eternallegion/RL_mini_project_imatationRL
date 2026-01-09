# train_sarsa_masked.py
import os
import json
import numpy as np
from puzzle_rl_env import CardPuzzleEnv, _encode_card, extract_card_feats

# from puzzle_rl_env import CardPuzzleEnv, _encode_card
from mcts_solver import solve_puzzle
from puzzle_rl_env import extract_card_feats
from typing import List
from typing import List, Dict, Any, Tuple


CHAIN_PUZZLES = {
    "chain_resolve_then_kill",
    "chain_negate_then_kill",
}
# ---- 상태-행동 특징 SARSA ----


OJAMA_DEMO_TOKENS = (
    "ACTIVATE:OJAMA_TRIO",
    "ACTIVATE:ZERO_GRAVITY",
    "ACTIVATE:BIG_EVOLUTION_PILL",
    "ATK",
    "STOP",
)


class LinearSARSASA:  # SA: State-Action
    def __init__(
        self,
        base_obs_dim: int,
        act_dim: int,
        card_feat_dim: int,
        alpha=3e-3,
        gamma=0.99,
        eps=0.35,
    ):
        self.obs_dim = base_obs_dim
        self.act_dim = act_dim
        self.card_feat_dim = card_feat_dim
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        # 공유 가중치
        self.w = np.zeros((self.obs_dim + self.card_feat_dim + 1,), dtype=np.float32)

    def _phi(self, obs: np.ndarray, card_feat: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [obs, card_feat, np.array([1.0], dtype=np.float32)], axis=0
        )

    def q_values(
        self, obs: np.ndarray, card_feats: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        A = card_feats.shape[0]
        qs = np.full((A,), -1e9, dtype=np.float32)
        for a in range(A):
            if mask[a] < 0.5:
                continue
            phi = self._phi(obs, card_feats[a])
            qs[a] = float(self.w @ phi)
        return qs

    def select_action(
        self, obs: np.ndarray, mask: np.ndarray, card_feats: np.ndarray
    ) -> int:
        valid = np.where(mask > 0.5)[0]
        if len(valid) == 0:
            return 0
        if np.random.rand() < self.eps:
            return int(np.random.choice(valid))
        q = self.q_values(obs, card_feats, mask)
        return int(np.argmax(q))

    def update(self, obs, a, r, obs2, a2, done, card_feat_a, card_feat_a2):
        phi = self._phi(obs, card_feat_a)
        phi2 = self._phi(obs2, card_feat_a2)
        q_sa = float(self.w @ phi)
        target = r + (0.0 if done else self.gamma * float(self.w @ phi2))
        td = target - q_sa
        self.w += self.alpha * td * phi

    def save(self, path: str):
        import numpy as np
        import os
        import json

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # np.savez(path, W=self.W, eps=self.eps, alpha=self.alpha, gamma=self.gamma)
        np.savez(path, W=self.w, eps=self.eps, alpha=self.alpha, gamma=self.gamma)

    def load(self, path: str):
        import numpy as np

        data = np.load(path, allow_pickle=True)
        self.w = data["W"]
        self.eps = float(data["eps"])
        self.alpha = float(data["alpha"])
        self.gamma = float(data["gamma"])


# ---- 카드 임베딩 추출 (슬롯별) ----


"""def extract_card_feats(env: CardPuzzleEnv) -> np.ndarray:
    kind_vocab = ["ATK","BUFF","BLOCK_BREAK","HEAL","DRAW","SCRIPT"]
    feats = []
    s = env.state
    # 카드 슬롯
    for i, c in enumerate(s.hand):
        if i in s.used:
            # used/pad: stop_flag=0
            feats.append(np.zeros(len(kind_vocab)+2, dtype=np.float32))
        else:
            base = _encode_card(c, kind_vocab)                # (6 + 1,)
            stop_flag = np.array([0.0], dtype=np.float32)     # <-- STOP 아님
            feats.append(np.concatenate([base, stop_flag], axis=0))
    # 패딩 슬롯 (STOP 아님)
    while len(feats) < env.max_hand:
        feats.append(np.zeros(len(kind_vocab)+2, dtype=np.float32))
    # STOP 슬롯: base=0, stop_flag=1.0
    stop_vec = np.zeros(len(kind_vocab)+2, dtype=np.float32)
    stop_vec[-1] = 1.0                                       # <-- STOP 표시
    feats.append(stop_vec)
    return np.stack(feats, axis=0)  # (A, feat_dim)"""


def extract_card_feats(env):
    kind_vocab = env.kind_vocab
    encode = getattr(env, "_encode_card", None)

    feats = []
    s = env.state
    for i, c in enumerate(s.hand):
        if i in s.used:
            feats.append(np.zeros(len(kind_vocab) + 2, dtype=np.float32))
        else:
            base = (
                encode(c)
                if callable(encode)
                else np.zeros(len(kind_vocab) + 1, dtype=np.float32)
            )
            feats.append(
                np.concatenate([base, np.array([0.0], dtype=np.float32)], axis=0)
            )

    while len(feats) < env.max_hand:
        feats.append(np.zeros(len(kind_vocab) + 2, dtype=np.float32))

    stop_vec = np.zeros(len(kind_vocab) + 2, dtype=np.float32)
    stop_vec[-1] = 1.0
    feats.append(stop_vec)

    return np.stack(feats, axis=0)


def _encode_card_local(c, kind_vocab):
    vec = np.zeros((len(kind_vocab) + 1,), dtype=np.float32)
    k = c.kind
    if k in kind_vocab:
        vec[kind_vocab[k]] = 1.0
    # value가 없으면 0, 있으면 0~1 스케일 (환경과 동일 스케일 유지: 보통 /2000 또는 /4000 사용)
    val = float(getattr(c, "value", 0.0) or 0.0)
    vec[-1] = np.clip(val / 2000.0, 0.0, 1.0)
    return vec


# --- helpers: place ABOVE warmstart_on_chain() ---


"""def _remap_demo_action(env, tok):
    s = env.state
    if tok is None:
        return env.max_hand  # STOP fallback

    # STOP
    if isinstance(tok, str) and tok.strip().upper() == "STOP":
        return env.max_hand

    # 현재 합법 액션
    mask = env.action_mask()
    legal = [i for i in range(env.max_hand) if mask[i] > 0.5]

    # 정수로 주어지면 그대로(합법 검사)
    if isinstance(tok, int):
        return tok if tok in legal else env.max_hand

    t = tok.strip()

    # "SCRIPT:OPCODE" 형식
    if ":" in t:
        head, tail = t.split(":", 1)
        head = head.strip().upper()
        op = tail.strip().upper()
        if head == "SCRIPT":
            # 스크립트 OP 매칭
            for i in legal:
                c = s.hand[i]
                if getattr(c, "kind", "") == "SCRIPT":
                    sc = getattr(c, "script", None)
                    sc_op = (sc or "")
                    sc_op = sc_op.upper() if isinstance(sc_op, str) else str(sc_op).upper()
                    if sc_op == op:
                        return i
            # 없으면 아무 SCRIPT
            for i in legal:
                if getattr(s.hand[i], "kind", "") == "SCRIPT":
                    return i
            return env.max_hand

    # 카드 이름으로 매칭 (예: "Finisher ATK")
    for i in legal:
        nm = getattr(s.hand[i], "name", "")
        if nm and nm.strip().lower() == t.strip().lower():
            return i

    # 종류 기반 (ATK/BUFF 등)
    tt = t.upper()
    want_kind = None
    if tt in ("ATK", "BUFF", "BLOCK_BREAK", "HEAL", "DRAW", "SCRIPT"):
        want_kind = tt
    if want_kind is not None:
        for i in legal:
            if getattr(s.hand[i], "kind", "") == want_kind:
                return i

    # 마지막 fallback
    return legal[0] if legal else env.max_hand"""
def _remap_demo_action(env, tok):
    tok = str(tok).strip()

    # 2-1) STOP 토큰 매핑 (mask 필요 없음)
    if tok.upper() in ("STOP", "END", "DONE"):
        return env.stop_index

    s = env.state
    if s is None:
        # 호출자가 reset을 안 했다면 방어적으로 한 번 호출
        env.reset(seed=42)
        s = env.state

    # 2-2) 인덱스 직접 지정 형태: "PLAY:3"
    if tok.upper().startswith("PLAY:"):
        try:
            idx = int(tok.split(":", 1)[1].strip())
            return idx
        except Exception:
            return env.stop_index  # 방어 리턴

    # 2-3) 이름/종류 기반
    # 예: "ATK" -> 아직 사용 안 한 ATK 첫 카드를 선택
    upper = tok.upper()
    if upper in ("ATK", "BUFF", "BLOCK_BREAK", "HEAL", "DRAW", "SCRIPT"):
        kind = upper
        for i, c in enumerate(s.hand):
            if i in s.used:
                continue
            if c.kind.upper() == kind:
                return i
        return env.stop_index

    # 2-4) 카드 이름으로 선택 (hand의 name 매칭)
    for i, c in enumerate(s.hand):
        if i in s.used:
            continue
        if c.name and c.name.strip().upper() == tok.upper():
            return i

    # 2-5) 못 찾으면 STOP
    return env.stop_index



def _remap_demo_seq(env, tokens):
    # state 보장
    if env.state is None:
        env.reset(seed=42)

    seq_idx = []
    for tok in tokens:
        a = _remap_demo_action(env, tok)
        seq_idx.append(a)
        # 시뮬레이션은 여기서 하지 않음 (매핑만 담당)
    return seq_idx


def _safe_index_card_feats(card_feats: np.ndarray, idx: int) -> np.ndarray:
    """idx 가 범위를 벗어나면 마지막(=STOP) 특징으로 안전 인덱싱"""
    if idx < 0 or idx >= card_feats.shape[0]:
        return card_feats[-1]
    return card_feats[idx]


# ---- 학습 루프 ----


def success_rate(env, policy, rollouts=40, max_steps=12) -> float:
    ok = 0
    for _ in range(rollouts):
        obs, info = env.reset()
        mask = info["action_mask"]
        card_feats = extract_card_feats(env)
        for _ in range(max_steps):
            a = policy.select_action(obs, mask, card_feats)
            obs2, r, done, info = env.step(a)
            card_feats2 = extract_card_feats(env)
            if done:
                # 성공 정의: LP 0 이하 && (끝내기 제약 없거나 마지막 사용이 ATK)
                end_ok = (env.state.opp_lp <= 0) and (
                    not env.constraints.get("must_end_with")
                    or env._last_used_kind() == "ATK"
                )
                ok += float(end_ok)
                break
            obs, mask, card_feats = obs2, info["action_mask"], card_feats2
    return ok / rollouts

def heuristic_pick(env):
    """스택이 열려있으면 RESOLVE/NEGATE류 SCRIPT 우선,
       스택이 닫혀있으면 ACTIVATE류 SCRIPT 우선, 그 외엔 ATK/BUFF 순."""
    s = env.state
    mask = env.action_mask()
    STOP = env.max_hand
    legal_idx = [i for i in range(env.max_hand) if mask[i] > 0.5]

    # 카드 참조
    def kind(i): return s.hand[i].kind
    def script_op(i):
        sc = getattr(s.hand[i], "script", None)
        if isinstance(sc, str):
            return sc.upper()
        return str(sc).upper() if sc else ""

    # 스택 열려있으면: RESOLVE/NEGATE 스크립트 먼저
    if len(s.stack) > 0:
        cand = [i for i in legal_idx if kind(i) == "SCRIPT" and script_op(i) in ("RESOLVE","NEGATE","CHAIN_RESOLVE","CHAIN_NEGATE")]
        if cand: return cand[0]
        # 그래도 없으면 SCRIPT 아무거나
        any_sc = [i for i in legal_idx if kind(i) == "SCRIPT"]
        if any_sc: return any_sc[0]
        # 안 되면 STOP 금지, 아무거나
        return legal_idx[0] if legal_idx else STOP

    # 스택 닫힘: ACTIVATE류 먼저
    cand = [i for i in legal_idx if kind(i) == "SCRIPT" and script_op(i) in ("ACTIVATE","SUMMON","OPEN")]
    if cand: return cand[0]

    # 그 다음 BUFF -> ATK
    buff = [i for i in legal_idx if kind(i) == "BUFF"]
    if buff: return buff[0]
    atk = [i for i in legal_idx if kind(i) == "ATK"]
    if atk: return atk[0]

    # 남는 게 없으면(드물지만) 아무거나
    return legal_idx[0] if legal_idx else STOP



def train(envs: List, agent, *, episodes=1500, max_steps=12, seed=1234, eval_every=250, stage=1):
    print("[train] envs in:", [e.puzzle_dict["name"] for e in envs])
    rng = np.random.RandomState(seed)

    for ep in range(1, episodes + 1):        # env 샘플링
        if len(envs) > 1:
            names = [e.puzzle_dict["name"] for e in envs]
            weights = np.array([3.0 if n in CHAIN_PUZZLES else 1.0 for n in names], dtype=np.float32)
            weights /= weights.sum()
            env = envs[int(rng.choice(len(envs), p=weights))]
        else:
            env = envs[0]

        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        mask = info["action_mask"]
        card_feats = extract_card_feats(env)
        use_heur = (stage == 5) and (ep <= 1000) and (rng.rand() < 0.5)
        if use_heur:
            a = heuristic_pick(env)
        else:
            a = agent.select_action(obs, mask, card_feats)
        for _ in range(max_steps):
            obs2, r, done, info = env.step(a)
            card_feats2 = extract_card_feats(env)
            mask2 = info["action_mask"]
            a2 = agent.select_action(obs2, mask2, card_feats2)

            cf_a  = card_feats[a]  if a  < len(card_feats)  else card_feats[-1]
            cf_a2 = card_feats2[a2] if a2 < len(card_feats2) else card_feats2[-1]

            agent.update(obs, a, r, obs2, a2, done, cf_a, cf_a2)
            obs, a, mask, card_feats = obs2, a2, mask2, card_feats2
            if done:
                break

        if ep % eval_every == 0:
            if stage == 5:
                agent.eps = max(0.10, agent.eps * 0.998)  # 탐색 조금 더 유지
            else:
                agent.eps = max(0.05, agent.eps * 0.995)
            rates = {e.puzzle_dict["name"]: success_rate(e, agent, rollouts=40, max_steps=max_steps) for e in envs}
            print(f"[Ep {ep}] eps={agent.eps:.3f} success={rates}")


def _index_for_action(env, act_name: str) -> int:
    if act_name == "STOP":
        return env.max_hand
    for i, c in enumerate(env.state.hand):
        if i in env.state.used:
            continue
        if c.name == act_name:
            return i
    raise RuntimeError(f"Action '{act_name}' not available in current hand.")


"""def warmstart_on_no_heal(agent, make_env_for, sims=400, demos=90, seed=2025, extract_card_feats=None):
    from mcts_solver import solve_puzzle
    # extract_card_feats가 None이면 내부에서 import
    if extract_card_feats is None:
        from puzzle_rl_env import extract_card_feats

    env = make_env_for("no_heal_cap_atk2")
    pzl = env.puzzle_dict

    ok, seq = solve_puzzle(pzl, sims=sims, seed=seed)
    if not ok or not seq:
        print("[warmstart] MCTS failed on 'no_heal_cap_atk2' -> skip")
        return

    import numpy as np
    rng = np.random.RandomState(seed)

    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        card_feats = extract_card_feats(env)
        for i, a in enumerate(seq):
            obs2, r, done, info = env.step(a)
            card_feats2 = extract_card_feats(env)
            a2 = seq[i+1] if i+1 < len(seq) else env.stop_index
            agent.update(obs, a, r, obs2, a2, done, card_feats[a], card_feats2[a2])
            obs, card_feats = obs2, card_feats2
            if done:
                break
    print(f"[warmstart] bootstrapped on 'no_heal_cap_atk2' with {demos} demos.")"""


def warmstart_on_no_heal(agent, make_env_for, sims=400, demos=90, seed=2025):
    """
    'no_heal_cap_atk2' 퍼즐을 MCTS로 풀어 얻은 데모 시퀀스로 SARSA 가중치를 워름스타트.
    - MCTS 결과 seq가 문자열/리스트 어떤 형식이든 처리
    - 데모 시퀀스를 env의 액션 인덱스 시퀀스로 리매핑한 뒤 TD 업데이트
    - 카드 특성 인덱싱은 안전 가드 적용
    """
    from mcts_solver import solve_puzzle
    from puzzle_rl_env import extract_card_feats
    import numpy as np

    # 1) 퍼즐/환경 준비
    env = make_env_for("no_heal_cap_atk2")
    pzl = env.puzzle_dict

    # 2) MCTS로 데모 시퀀스 확보
    ok, seq = solve_puzzle(pzl, sims=sims, seed=seed)
    if not ok or not seq:
        print("[warmstart] MCTS failed on 'no_heal_cap_atk2' -> skip")
        return

    # 3) seq를 토큰 리스트로 표준화
    if isinstance(seq, str):
        seq_tokens = seq.strip().split()
    elif isinstance(seq, (list, tuple)):
        seq_tokens = list(seq)
    else:
        print(f"[warmstart] unexpected seq type: {type(seq)} -> skip")
        return

    # 4) 토큰 → 액션 인덱스 시퀀스로 리매핑
    seq_idx = _remap_demo_seq(env, seq_tokens)
    if not seq_idx:
        print("[warmstart] remap failed -> skip")
        return

    # 5) 데모 여러 번 재생하며 SARSA 업데이트
    rng = np.random.RandomState(seed)
    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        card_feats = extract_card_feats(env)

        done = False
        for i, a in enumerate(seq_idx):
            # 다음 액션 a2 (마지막이면 STOP)
            a2 = env.stop_index if i == len(seq_idx) - 1 else seq_idx[i + 1]

            obs2, r, done, info = env.step(a)
            card_feats2 = extract_card_feats(env)

            # 안전 인덱싱
            cf_a = card_feats[a] if 0 <= a < len(card_feats) else card_feats[-1]
            cf_a2 = card_feats2[a2] if 0 <= a2 < len(card_feats2) else card_feats2[-1]

            agent.update(obs, a, r, obs2, a2, done, cf_a, cf_a2)

            obs, card_feats = obs2, card_feats2
            if done:
                break

    print(f"[warmstart] bootstrapped on 'no_heal_cap_atk2' with {demos} demos.")


"""def warmstart_on_chain(puzzles: list,agent,make_env_for,extract_card_feats,demos: int = 40,sims: int = 800,seed: int = 7,):

    #chain_* 퍼즐들을 MCTS 정답 시퀀스로 워름스타트.
    #- puzzles: puzzles.json 로드한 딕셔너리 리스트
    #- agent: LinearSARSASA 에이전트
    #- make_env_for(name) -> CardPuzzleEnv
    #- extract_card_feats(env) -> (A, feat_dim) 카드별 특징


    # 어떤 퍼즐을 체인 계열로 볼지 정의 (없으면 이름 접두사로 필터)
    try:
        CHAIN_PUZZLES = {"chain_resolve_then_kill", "chain_negate_then_kill"}
    except Exception:
        CHAIN_PUZZLES = set()

    try:
        from mcts_solver import solve_puzzle
    except Exception as e:
        print(f"[warmstart] skip chain warmstart (mcts_solver import fail): {e}")
        return

    # 대상 퍼즐 추출
    chain_list = [
        p for p in puzzles
        if (p.get("name") in CHAIN_PUZZLES) or str(p.get("name", "")).startswith("chain_")
    ]
    if not chain_list:
        print("[warmstart] no chain puzzles to warmstart.")
        return

    print("[warmstart] chain puzzles:", [p["name"] for p in chain_list])

    rng = np.random.RandomState(seed)"""


def warmstart_on_chain(agent, make_env_for, target_name, sims=800, demos=40, seed=7):
    """
    체인 퍼즐 워밍업:
    - MCTS로 얻은 시퀀스(seq)를 토큰으로 표준화
    - STOP 토큰 보장
    - env.reset() 후 인덱스 시퀀스로 리매핑
    - 안전 인덱싱으로 에이전트 업데이트
    """
    # 1) 의존 모듈
    from mcts_solver import solve_puzzle
    from puzzle_rl_env import extract_card_feats
    import numpy as np

    # 2) 환경 준비
    env = make_env_for(target_name)
    pzl = env.puzzle_dict
    print(f"[warmstart] bootstrapping '{target_name}' with MCTS demos...")

    # 3) MCTS로 시퀀스 얻기
    ok, seq = solve_puzzle(pzl, sims=sims, seed=seed)
    if not ok or not seq:
        print(f"[warmstart] MCTS failed on {target_name} -> skip")
        return

    # 4) 시퀀스 표준화 (str | list | tuple 모두 처리)
    if isinstance(seq, str):
        seq_tokens = seq.strip().split()
    elif isinstance(seq, (list, tuple)):
        seq_tokens = list(seq)
    else:
        print(f"[warmstart] unexpected seq type: {type(seq)} -> skip")
        return

    # 5) STOP 보장
    last = (seq_tokens[-1].upper() if seq_tokens else "")
    if last not in ("STOP", "END", "DONE"):
        seq_tokens.append("STOP")

    # 6) 리매핑 전 반드시 reset (state=None 보호)
    env.reset(seed=seed)

    # 7) 토큰 → 인덱스 리매핑
    seq_idx = _remap_demo_seq(env, seq_tokens)
    if not seq_idx:
        print(f"[warmstart] remap failed on {target_name} -> skip")
        return

    # 8) 시연 업데이트 루프 (시드 다양화)
    rng = np.random.RandomState(seed)
    STOP = getattr(env, "stop_index", env.max_hand)

    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        card_feats = extract_card_feats(env)

        for t, a in enumerate(seq_idx):
            # 환경 진행
            obs2, r, done, info = env.step(a)
            card_feats2 = extract_card_feats(env)

            # 다음 액션(마지막이면 STOP)
            a2 = seq_idx[t + 1] if (not done and t + 1 < len(seq_idx)) else STOP

            # 안전 인덱싱 (경계 보호)
            cf_a  = card_feats[a]  if 0 <= a  < len(card_feats)  else card_feats[-1]
            cf_a2 = card_feats2[a2] if 0 <= a2 < len(card_feats2) else card_feats2[-1]

            # 에이전트 업데이트
            agent.update(obs, a, r, obs2, a2, done, cf_a, cf_a2)

            # 다음 스텝 준비
            obs, card_feats = obs2, card_feats2
            if done:
                break

    print(f"[warmstart] bootstrapped '{target_name}' with {demos} demos (MCTS sims={sims}).")


    """def warmstart_on_ojama(agent, make_env_for, demos=60, seed=2025):
    """
    #[GX_Spirit_Caller]B03_Oh_Jama를 정답 시퀀스로 워름스타트.
    #정답: OJAMA_TRIO -> ZERO_GRAVITY -> BIG_EVOLUTION_PILL -> ATK(9900)
    """
    rng = np.random.RandomState(seed)
    env = make_env_for("[GX_Spirit_Caller]B03_Oh_Jama")

    # 손패 인덱스로 정답 시퀀스(0,1,2,3) 고정
    demo_seq = [0, 1, 2, 3]

    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        mask = info["action_mask"]
        card_feats = extract_card_feats(env)

        done = False
        for i, a in enumerate(demo_seq):
            # 다음 a' (없으면 STOP)
            a2 = demo_seq[i + 1] if i + 1 < len(demo_seq) else env.stop_index

            obs2, r, done, info = env.step(a)
            card_feats2 = extract_card_feats(env)

            # 안전 체크
            if a < card_feats.shape[0] and a2 < card_feats2.shape[0]:
                agent.update(obs, a, r, obs2, a2, done, card_feats[a], card_feats2[a2])

            obs, card_feats = obs2, card_feats2
            if done:
                break
    print(f"[warmstart] bootstrapped '{env.puzzle_dict['name']}' with {demos} demos.")"""

def warmstart_on_ojama(agent, make_env_for, sims=1200, demos=200, seed=2025):
    """Ojama 퍼즐을 MCTS(성공 시) 또는 Fallback 데모로 워밍업"""
    import numpy as np
    from mcts_solver import solve_puzzle
    from puzzle_rl_env import extract_card_feats

    name = "[GX_Spirit_Caller]B03_Oh_Jama"
    env = make_env_for(name)
    pzl = env.puzzle_dict
    rng = np.random.RandomState(seed)

    # ★ 리매핑/마스크 등에서 state가 필요할 수 있으므로 선제 reset
    env.reset(seed=int(rng.randint(0, 10_000_000)))

    print(f"[warmstart] bootstrapping '{pzl['name']}' with MCTS demos...")
    ok, seq = solve_puzzle(pzl, sims=sims, seed=seed)

    # --- 토큰 표준화 ---
    if ok and seq:
        if isinstance(seq, str):
            seq_tokens = seq.strip().split()
        elif isinstance(seq, (list, tuple)):
            seq_tokens = list(seq)
        else:
            print(f"[warmstart] unexpected seq type: {type(seq)} -> use FALLBACK demo")
            seq_tokens = list(OJAMA_DEMO_TOKENS)
    else:
        print(f"[warmstart] MCTS failed on '{pzl['name']}' -> use FALLBACK demo")
        # OJAMA_DEMO_TOKENS 가 위쪽에 정의되어 있어야 함
        # 예: OJAMA_DEMO_TOKENS = ("ACTIVATE:OJAMA_TRIO","ACTIVATE:ZERO_GRAVITY","ACTIVATE:BIG_EVOLUTION_PILL","ATK","STOP")
        seq_tokens = list(OJAMA_DEMO_TOKENS)

    # ★ 끝에 STOP 보장(없으면 추가)
    if not seq_tokens or seq_tokens[-1].upper() not in ("STOP", "END", "DONE"):
        seq_tokens.append("STOP")

    # --- 텍스트 토큰 → 액션 인덱스 리매핑 (여기서는 실행 X, 매핑만) ---
    seq_idx = _remap_demo_seq(env, seq_tokens)
    if not seq_idx:
        print("[warmstart] remap failed -> skip this demo")
        return

    # --- 데모를 agent에 반영 ---
    for _ in range(demos):
        obs, info = env.reset(seed=int(rng.randint(0, 10_000_000)))
        card_feats = extract_card_feats(env)

        for i, a in enumerate(seq_idx):
            obs2, r, done, info = env.step(a)
            card_feats2 = extract_card_feats(env)

            # 다음 액션 준비(없거나 종료면 STOP)
            stop_idx = getattr(env, "stop_index", env.max_hand)
            a2 = seq_idx[i + 1] if (not done and i + 1 < len(seq_idx)) else stop_idx

            # 안전 인덱싱
            cf_a  = card_feats[a]  if 0 <= a  < len(card_feats)  else card_feats[-1]
            cf_a2 = card_feats2[a2] if 0 <= a2 < len(card_feats2) else card_feats2[-1]

            agent.update(obs, a, r, obs2, a2, done, cf_a, cf_a2)
            obs, card_feats = obs2, card_feats2
            if done:
                break

    print(f"[warmstart] bootstrapped '{pzl['name']}' with {demos} demos (MCTS sims={sims}).")

def pick_action_by_name(env, name: str, mask: np.ndarray) -> int:
    """현재 환경 상태에서 '이름'에 해당하는 합법 액션 인덱스를 찾아준다."""
    # STOP 처리
    if name.upper() == "STOP":
        return env.stop_index

    # 손패에서 아직 안 쓴 동일 이름 카드 슬롯을 찾되, 마스크가 1인 것만 허용
    s = env.state
    for i, c in enumerate(s.hand):
        if i in s.used:
            continue
        if getattr(c, "name", None) == name and i < len(mask) and mask[i] > 0.0:
            return i

    # 이름으로 못 찾으면(혹은 마스크로 금지되었으면) 마지막 수단: 합법한 아무 액션
    legal = np.where(mask > 0.0)[0]
    if len(legal) == 0:
        # 전혀 없으면 그냥 STOP 시도
        return env.stop_index
    return int(legal[0])

    for pzl in chain_list:
        ok, seq_names = solve_puzzle(pzl, sims=sims, seed=int(rng.randint(10**9)))
        if not ok or not seq_names:
            print(f"[warmstart] MCTS failed on {pzl['name']} -> skip")
            continue

        env = make_env_for(pzl["name"])

        for d in range(demos):
            obs, info = env.reset(seed=int(rng.randint(10**9)))
            mask = info["action_mask"]
            card_feats = extract_card_feats(env)

            done = False
            # 시퀀스를 '이름' 기준으로 순차 실행
            for t, name in enumerate(seq_names):
                a = pick_action_by_name(env, name, mask)
                obs2, r, done, info = env.step(a)
                mask2 = info["action_mask"]
                card_feats2 = extract_card_feats(env)

                # 다음 액션 결정 (시퀀스 따라가되 끝나면 STOP 시도)
                if not done and (t + 1) < len(seq_names):
                    next_name = seq_names[t + 1]
                    a2 = pick_action_by_name(env, next_name, mask2)
                else:
                    # STOP이 금지된 상태면(예: 스택 열림) 합법 액션 중 하나로 대체
                    a2 = (
                        env.stop_index
                        if (env.stop_index < len(mask2) and mask2[env.stop_index] > 0.0)
                        else int(np.argmax(mask2))
                    )

                # SARSA 업데이트 (카드 피처는 a/a2 인덱스로 안전 접근)
                a_safe = int(np.clip(a, 0, card_feats.shape[0] - 1))
                a2_safe = int(np.clip(a2, 0, card_feats2.shape[0] - 1))
                agent.update(
                    obs, a, r, obs2, a2, done, card_feats[a_safe], card_feats2[a2_safe]
                )

                obs, mask, card_feats = obs2, mask2, card_feats2
                if done:
                    break

        print(f"[warmstart] bootstrapped on '{pzl['name']}' with {demos} demos.")


# ---- 엔트리 ----

if __name__ == "__main__":
    import os
    import json
    from puzzle_rl_env import CardPuzzleEnv, extract_card_feats

    os.makedirs("checkpoints", exist_ok=True)

    with open("puzzles.json", "r", encoding="utf-8") as f:
        puzzles = json.load(f)

    curriculum = [
        ["seq_block_buff_kill"],
        ["seq_block_buff_kill", "no_heal_cap_atk2"],
        ["seq_block_buff_kill", "no_heal_cap_atk2", "must_break_before_first_atk"],
        [
            "seq_block_buff_kill",
            "no_heal_cap_atk2",
            "must_break_before_first_atk",
            "chain_resolve_then_kill",
            "chain_negate_then_kill",
        ],
        ["[GX_Spirit_Caller]B03_Oh_Jama"],
    ]

    def make_env_for(name):
        p = next(p for p in puzzles if p["name"] == name)
        return CardPuzzleEnv(p, max_hand=8, length_penalty=0.004, progress_coef=1.0)

    # 에이전트 생성
    probe = make_env_for("seq_block_buff_kill")
    obs0, _ = probe.reset(seed=42)
    card_feat_dim = 8  # |kinds|(6) + value(1) + stop_flag(1)
    agent = LinearSARSASA(
        base_obs_dim=probe.observation_size,
        act_dim=probe.action_size,
        card_feat_dim=card_feat_dim,
        alpha=3e-3,
        gamma=0.99,
        eps=0.25,
    )

    for stage, names in enumerate(curriculum, start=1):
        envs = [make_env_for(n) for n in names]
        print(f"\n[Stage {stage}] puzzles = {names}")
        print("[train] envs in:", [e.puzzle_dict["name"] for e in envs])

        # Stage 2에 no_heal_cap_atk2가 있으면 MCTS 워름스타트(옵션)
        if ("no_heal_cap_atk2" in names) and (stage == 2):
            print("[warmstart] bootstrapping on 'no_heal_cap_atk2' with MCTS demos...")
            warmstart_on_no_heal(
                agent=agent, make_env_for=make_env_for, sims=400, demos=90, seed=2025
            )

        # Stage 4에 chain 퍼즐이 포함되면 체인 워름스타트
        if stage == 4 and any(n.startswith("chain_") for n in names):
            # warmstart_on_chain(
            #    puzzles=puzzles,
            #    agent=agent,
            #    make_env_for=make_env_for,
            #    extract_card_feats=extract_card_feats,
            #    demos=80, sims=600, seed=2025
            # )
            warmstart_on_chain(
                agent,
                make_env_for,
                "chain_resolve_then_kill",
                sims=600,
                demos=80,
                seed=2025,
            )
            warmstart_on_chain(
                agent,
                make_env_for,
                "chain_negate_then_kill",
                sims=600,
                demos=80,
                seed=2025,
            )




        envs = [ make_env_for("[GX_Spirit_Caller]B03_Oh_Jama") ]
        warmstart_on_ojama(agent, make_env_for, demos=200, sims=1200, seed=2025)
        if stage == 5:  # GX 오자마
            for i, e in enumerate(envs):
                e.length_penalty = 0.001  # or 0.0
                agent.eps = 0.15
            warmstart_on_ojama(agent, make_env_for, demos=200, sims=1200, seed=2025)
            train(
                envs,
                agent,
                episodes=3000,
                max_steps=18,
                seed=1234 + stage,
                eval_every=300,
            )

        # 학습 (에피소드 늘리고, stage=5 넘겨서 heuristic 활성화)
        train(
            envs, agent,
            episodes=1800,      # 1500 -> 1800 정도
            max_steps=14,       # 12 -> 14 (체인 여유)
            seed=5555,
            eval_every=300,
            stage=5             # ★ 중요: 위 heuristic_pick 스위치용
        )
        #save_agent(agent, "checkpoints/sarsa_stage5.npz")


        # if "[GX_Spirit_Caller]B03_Oh_Jama" in names:
        #    warmstart_on_ojama(agent, make_env_for, demos=60, seed=2025)

        # 학습 시작
        train(
            envs, agent, episodes=1500, max_steps=12, seed=1234 + stage, eval_every=250
        )

        # 체크포인트 저장
        agent.save(f"checkpoints/sarsa_stage{stage}.npz")
        print(f"[saved] checkpoints/sarsa_stage{stage}.npz")
