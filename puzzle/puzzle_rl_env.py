# puzzle_rl_env.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Set, Optional
import numpy as np
# ---------- 카드/상태 ----------

from dataclasses import dataclass

# @dataclass(frozen=True)for script in scripts:x
# class Card:
#    name: str
#    kind: str              # "ATK","BUFF","BLOCK_BREAK","HEAL","DRAW","SCRIPT"
#    value: int = 0         # ATK/BUFF/HEAL 등 숫자 효과가 있는 경우
#    text: str = ""         # 사람이 읽는 설명(있으면 유지)
#    script: Optional[Dict[str, Any]] = None  # <-- 새로 추가

# @dataclass
# class PuzzleState:
#    opp_lp: int
#    opp_block: bool
#    my_buff: int
#    hand: Tuple[Card, ...]
#    used: Tuple[int, ...] = field(default_factory=tuple)  # 이미 사용한 hand 인덱스
#    stack: Tuple[str, ...] = field(default_factory=tuple) # (간단 표시용)

#    def legal_actions(self) -> List[int]:
       # 사용하지 않은 카드만 선택 가능
#        return [i for i in range(len(self.hand)) if i not in self.used]

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional, Set




#@dataclass(frozen=True)
#class State:
#    opp_lp: int
#    my_buff: int
#    opp_block: bool
#    hand: Tuple[Card, ...]
#    used: Tuple[int, ...]
#    stack: Tuple[str, ...]          # 간단히 문자열 스택으로 둠
#    constraints: Dict[str, Any]
#    turn: int
#    my_lp: int
#    def legal_actions(self) -> List[int]:
#        return [i for i in range(len(self.hand)) if i not in self.used]
#PuzzleState = State



# ---------- 퍼즐 로딩 ----------
# ----오리지널----
# def make_state_from_dict(pzl: Dict[str, Any]) -> PuzzleState:
#    hand = tuple(Card(**c) for c in pzl.get("hand", []))
#    return PuzzleState(
#        opp_lp = int(pzl.get("opp_lp", 2000)),
#        opp_block = bool(pzl.get("opp_block", False)),
#        my_buff = 0,
#        hand = hand,
#        used = tuple(),
#        stack = tuple()
#    )


"""def make_state_from_dict(pzl: Dict[str, Any]) -> State:
    opp_lp = int(pzl.get("opp_lp", 4000))
    my_buff = int(pzl.get("my_buff", 0))
    opp_block = bool(pzl.get("opp_block", False))
    my_lp = int(pzl.get("my_lp", 4000))

    hand_list = []
    for c in pzl.get("hand", []):
        hand_list.append(Card(
            name=c.get("name", ""),
            kind=c.get("kind", "ATK"),
            value=int(c.get("value", 0)),
            script=c.get("script", None)
        ))
    hand = tuple(hand_list)

    stack = tuple()
    used = set()
    constraints = pzl.get("constraints", {}) or {}

    return State(
        opp_lp=opp_lp,
        my_buff=my_buff,
        opp_block=opp_block,
        hand=hand,
        stack=stack,
        used=used,
        constraints=constraints,
        turn=0,
        my_lp=my_lp
    )"""
# puzzle_rl_env.py
# === helpers =================================================
def _get_op_from_card_obj_or_str(c) -> str:
    """
    카드/문자열에서 스크립트 op를 안전하게 추출해 대문자로 반환.
    대응:
      - dataclass Card with .script (dict or str)
      - str 토큰 ("ACTIVATE:OJAMA_TRIO" 또는 "OJAMA_TRIO")
      - name 기반 휴리스틱 ("Zero Gravity" 등)
    """
    # 1) Card.script
    s = getattr(c, "script", None)
    if isinstance(s, dict):
        return str(s.get("op", "")).strip().upper()
    if isinstance(s, str):
        tok = s.strip()
        if ":" in tok:
            return tok.split(":", 1)[0].strip().upper()
        return tok.upper()

    # 2) 토큰 문자열
    if isinstance(c, str):
        tok = c.strip()
        if ":" in tok:
            return tok.split(":", 1)[0].strip().upper()
        return tok.upper()

    # 3) name 휴리스틱
    nm = str(getattr(c, "name", "")).strip().upper().replace(" ", "_")
    if "OJAMA" in nm and "TRIO" in nm:
        return "OJAMA_TRIO"
    if "ZERO" in nm and "GRAVITY" in nm:
        return "ZERO_GRAVITY"
    if "BIG" in nm and "EVOLUTION" in nm and "PILL" in nm:
        return "BIG_EVOLUTION_PILL"

    return ""

@dataclass(frozen=True)
class Card:
    name: str
    kind: str   # "ATK","BUFF","HEAL","BLOCK_BREAK","DRAW","SCRIPT" 등
    value: int = 0
    script: str | None = None

@dataclass(frozen=True)
class State:
    opp_lp: int
    my_lp: int
    my_buff: int
    opp_block: bool
    hand: Tuple[Card, ...]
    used: Tuple[int, ...]
    stack: Tuple[str, ...]          # 간단한 데모면 빈 튜플 유지
    constraints: Dict[str, Any]
    turn: int = 0
    
    def legal_actions(self) -> Tuple[int, ...]:
       # STOP은 환경(step)에서 따로 처리하므로 여기서는 손패 인덱스만
       if not self.hand:
           return tuple()
       used_set = set(self.used)
       return tuple(i for i in range(len(self.hand)) if i not in used_set)

def make_state_from_dict(pzl: Dict[str, Any]) -> State:
    hand = tuple(Card(**c) for c in pzl.get("hand", []))
    return State(
        opp_lp=int(pzl.get("opp_lp", 4000)),
        my_lp=int(pzl.get("my_lp", 4000)),
        my_buff=int(pzl.get("my_buff", 0)),
        opp_block=bool(pzl.get("opp_block", False)),
        hand=hand,
        used=tuple(pzl.get("used", ())),
        stack=tuple(pzl.get("stack", ())),
        constraints=pzl.get("constraints", {}),
        turn=int(pzl.get("turn", 0)),
    )



def extract_card_feats(env) -> np.ndarray:
    """
    액션 슬롯별(card 0..max_hand-1 + STOP) 피처를 (A, feat_dim)로 반환.
    feat_dim = |kinds| + 1(value) + 1(stop_flag)
    STOP 슬롯은 마지막 인덱스(env.max_hand)이며, stop_flag=1.0으로 표시.
    """
    kind_vocab = env.kind_vocab   # 환경과 동일 매핑 사용

    feats = []
    s = env.state

    # 손패 슬롯
    for i, c in enumerate(s.hand):
        if i in s.used:
            # 사용된/빈 슬롯: STOP 아님(마스크로 막히므로 zero 벡터)
            feats.append(np.zeros(len(kind_vocab) + 2, dtype=np.float32))
        else:
            # 환경과 동일 인코딩 사용 (value 스케일도 동일)
            # shape = (|kinds| + 1,)
            base = _encode_card(c, kind_vocab)
            # + stop_flag=0
            feats.append(np.concatenate([base, [0.0]], axis=0))

    # 패딩 슬롯(손패 길이가 max_hand보다 짧을 때)
    while len(feats) < env.max_hand:
        feats.append(np.zeros(len(kind_vocab) + 2, dtype=np.float32))

    # STOP 슬롯 (마지막): base=0, stop_flag=1
    stop_vec = np.zeros(len(kind_vocab) + 2, dtype=np.float32)
    stop_vec[-1] = 1.0
    feats.append(stop_vec)

    return np.stack(feats, axis=0)


# ---------- 내부 유틸 ----------

def _encode_kind(kind: str, kind_vocab: List[str]) -> np.ndarray:
    v = np.zeros((len(kind_vocab),), dtype=np.float32)
    if kind in kind_vocab:
        v[kind_vocab.index(kind)] = 1.0
    return v


def _encode_card(card: Card, kind_vocab: List[str]) -> np.ndarray:
    # one-hot(kind) + scaled value
    one = _encode_kind(card.kind, kind_vocab)
    val = np.array([np.clip(card.value, -4000, 4000) / 4000.0],
                   dtype=np.float32)
    return np.concatenate([one, val], axis=0)


# 외부에서 import 할 수 있게 노출
__all__ = [
    "Card",
    "PuzzleState",
    "make_state_from_dict",
    "_encode_card",
     "CardPuzzleEnv"]

# ---------- 환경 본체 ----------


class CardPuzzleEnv:
    """
    STOP 액션 지원 환경.
    - 액션: 0..max_hand-1 => 카드 선택,  max_hand => STOP(종료)
    - 제약 인지 마스킹:
        * forbid_kinds
        * limit_kind_counts
        * must_break_before_first_atk
        * must_end_with
    - 보상:
        * 성공(+1.0), 진행 shaping(-ΔLP*coef), 길이 패널티(-λ)
    관측:
        [opp_lp_norm, my_buff_norm, opp_block,
         forbid_HEAL, must_break_first, atk_limit2,
         stack_depth onehot(<=4),
         hand(각 카드 onehot(kind)+value_norm), PAD=0,
         bias(1.0)]
    """

    def __init__(
        self,
        puzzle_dict: Dict[str, Any],
        max_hand: int = 8,
        length_penalty=0.003, 
        progress_coef=1.3,
    ):
        self.puzzle_dict = puzzle_dict
        self.max_hand = max_hand
        self.length_penalty = length_penalty
        self.progress_coef = progress_coef
        self.kind_vocab: List[str] = ["ATK", "BUFF",
            "BLOCK_BREAK", "HEAL", "DRAW", "SCRIPT"]

        # 제약 보관
        self.constraints: Dict[str, Any] = self.puzzle_dict.get(
            "constraints", {}) or {}

        # 초기화
        self.state: Optional[PuzzleState] = None
        self._init_lp: int = 2000
        self._last_flags = {}     # 보상 셰이핑용 일시 플래그
        self._script_flags = {}
        self._t: int = 0

    # ---- 크기 정보 ----
    @property
    def action_size(self) -> int:
        # 카드 max_hand개 + STOP 1개
        return self.max_hand + 1

    @property
    def observation_size(self) -> int:
        # 기존: 6 + 5 + self.max_hand*(len(kind_vocab)+1) + 1
        return 8 + 5 + self.max_hand * (len(self.kind_vocab) + 1) + 1

    @property
    def stop_index(self) -> int:
        return self.max_hand

    # ---- 헬퍼 ----
    def _used_kind_counts(self) -> Dict[str, int]:
        s = self.state
        d: Dict[str, int] = {}
        for idx in s.used:
            k = getattr(s.hand[idx], "kind", None)
            if k:
                d[k] = d.get(k, 0) + 1
        return d

    def _breaker_used(self) -> bool:
        s = self.state
        for idx in s.used:
            if getattr(s.hand[idx], "kind", "") == "BLOCK_BREAK":
                return True
        return False

    def _last_used_kind(self) -> Optional[str]:
        s = self.state
        if not s.used:
            return None
        last_idx = s.used[-1]
        return getattr(s.hand[last_idx], "kind", None)

    def _can_stop_now(self) -> bool:
        """
        must_end_with == 'ATK'인 퍼즐에서,
        마지막 행동이 ATK고, 더 이상 ATK를 쓰기 어렵거나(한도 도달/없음) 지금 끝내도 되면 True
        """
        must_end = self.constraints.get("must_end_with", None)
        last_k = self._last_used_kind()

        if must_end == "ATK":
            if last_k != "ATK":
                return False
            limit = self.constraints.get("limit_kind_counts", {}) or {}
            used_counts = self._used_kind_counts()
            atk_limit = int(limit.get("ATK", 9999))
            no_more_atk_by_limit = used_counts.get("ATK", 0) >= atk_limit
            # 남은 손패에 ATK가 남아있는지
            no_more_atk_in_hand = True
            for i, c in enumerate(self.state.hand):
                if i in self.state.used:
                    continue
                if getattr(c, "kind", "") == "ATK":
                    no_more_atk_in_hand = False
                    break
            if no_more_atk_by_limit or no_more_atk_in_hand:
                return True
            return False
        # must_end 제약 없을 땐 기본적으로 STOP 안 보이게(보수적)
        return False

    def _apply_script_effect(self, s: State, script_name: str) -> State:
        """script_effects 사전(퍼즐 JSON) 기반으로 SCRIPT 효과를 적용."""
        eff_map = (self.puzzle_dict or {}).get("script_effects", {})
        eff = eff_map.get(script_name, {})
        opp_lp = int(s.opp_lp)
        my_buff = int(s.my_buff)
        opp_block = bool(s.opp_block)

        # 버프 적용
        add_buff = int(eff.get("apply_buff", 0))
        if add_buff:
            my_buff += add_buff

        # 블록 해제
        if bool(eff.get("clear_block", False)):
            opp_block = False

        # 즉시 데미지(필요 시)
        dmg = int(eff.get("damage", 0))
        if dmg > 0:
            # 블록 중이면 막히는 규칙을 따르려면 여기서 체크 가능
            if opp_block:
                dmg = 0
            opp_lp = max(0, opp_lp - dmg)

        # 스택 로깅(간단하게 script 이름만 남김)
        new_stack = tuple(list(s.stack) + [f"SCRIPT:{script_name}"])

        return State(
            opp_lp=opp_lp,
            my_buff=my_buff,
            opp_block=opp_block,
            hand=s.hand,
            used=s.used,
            stack=new_stack,
            constraints=s.constraints,
            turn=s.turn,
            my_lp=s.my_lp
        )



    # ---- 관측/마스크 ----
    def _pad_stack_depth(self, depth: int, max_depth: int = 4) -> np.ndarray:
        d = min(max(depth, 0), max_depth)
        one = np.zeros((max_depth + 1,), dtype=np.float32)
        one[d] = 1.0
        return one

    def _obs(self) -> np.ndarray:
        s = self.state
        opp_lp = np.clip(float(s.opp_lp) / max(1.0, self._init_lp), 0.0, 1.0)
        my_buff = np.clip(float(s.my_buff) / 4000.0, 0.0, 1.0)
        opp_block = 1.0 if s.opp_block else 0.0

        forbid_heal = 1.0 if "HEAL" in (
    self.constraints.get("forbid_kinds") or []) else 0.0
        must_break_first = 1.0 if self.constraints.get(
            "must_break_before_first_atk", False) else 0.0
        atk_limit2 = 1.0 if (
    self.constraints.get(
        "limit_kind_counts",
        {}) or {}).get(
            "ATK",
             None) == 2 else 0.0

        # ✅ 추가: 마지막 사용 카드가 ATK인지, 남은 ATK 개수(정규화)
        last_is_atk = 1.0 if self._last_used_kind() == "ATK" else 0.0
        limit = self.constraints.get("limit_kind_counts", {}) or {}
        used_counts = self._used_kind_counts()
        atk_limit = int(limit.get("ATK", 9999))
        atk_used = int(used_counts.get("ATK", 0))
        # 최대 3까지 클리핑해 0~1로 정규화(퍼즐 난이도에 맞춰 2~3 정도면 충분)
        atk_left_norm = float(min(max(atk_limit - atk_used, 0), 3)) / 3.0

        scalars = np.array(
            [opp_lp, my_buff, opp_block,
             forbid_heal, must_break_first, atk_limit2,
             last_is_atk, atk_left_norm],               # ⬅️ 두 항목 추가
            dtype=np.float32
        )

        stack_feat = self._pad_stack_depth(len(s.stack), max_depth=4)

        top_has = 1.0 if len(s.stack) > 0 else 0.0
        top_speed = float(s.stack[-1].get("speed", 0)) if top_has else 0.0
        top_neg = 1.0 if (
            len(s.stack) > 0 and s.stack[-1].get("negated", False)) else 0.0
        top_kind = s.stack[-1].get("kind") if top_has else None
        top_kind_one = np.zeros(len(self.kind_vocab), dtype=np.float32)
        if top_kind in self.kind_vocab:
            top_kind_one[self.kind_vocab.index(top_kind)] = 1.0
        stack_ctx = np.concatenate([np.array([top_has, top_speed, top_neg], np.float32),
                                    top_kind_one], axis=0)

        # 손패 인코딩 (사용된 카드=0)
        hand_feats: List[np.ndarray] = []
        for i, c in enumerate(s.hand):
            if i in s.used:
                hand_feats.append(
                    np.zeros((len(self.kind_vocab) + 1,), dtype=np.float32))
            else:
                hand_feats.append(_encode_card(c, self.kind_vocab))
        while len(hand_feats) < self.max_hand:
            hand_feats.append(
    np.zeros(
        (len(
            self.kind_vocab) +
            1,
            ),
             dtype=np.float32))
        hand_feat = np.concatenate(hand_feats, axis=0)

        bias = np.array([1.0], dtype=np.float32)

        obs = np.concatenate([scalars, stack_feat, hand_feat, bias], axis=0)
        return obs

    def action_mask(self) -> np.ndarray:
        """
        사용 가능 액션 마스크 생성.
        - 손패의 '아직 쓰지 않은 카드'는 1, 나머지는 0
        - STOP(=max_hand)은 기본 0에서 규칙/상태에 따라 1로 전환
        - forbid_kinds / must_break_before_first_atk / limit_kind_counts / must_end_with 적용
        - script_sequence 기반의 체인/힐 제한 등 보조 제약 적용
        """
        A = self.max_hand + 1
        STOP_IDX = self.max_hand
        mask = np.zeros(A, dtype=np.float32)

        # 안전장치
        s = self.state
        assert s is not None, "call reset() before action_mask()"

        cons = self.constraints or {}
        forbid = set(cons.get("forbid_kinds") or [])
        limit = cons.get("limit_kind_counts") or {}
        atk_limit = int(limit.get("ATK", 10))
        atk_used = int(self._used_kind_counts().get("ATK", 0))

        # 1) 기본: 아직 안 쓴 카드만 ON
        for i, _c in enumerate(s.hand):
            if i not in s.used:
                mask[i] = 1.0

        # 2) 패딩 구간은 0 유지
        if len(s.hand) < self.max_hand:
            mask[len(s.hand):self.max_hand] = 0.0

        # 3) STOP은 기본 OFF (아래에서 필요 시 켬)
        mask[STOP_IDX] = 0.0

        # ---- 전역 규칙 제약 ----
        # a) 금지 종류 제거
        if forbid:
            for i, c in enumerate(s.hand):
                if i in s.used:
                    continue
                if c.kind in forbid:
                    mask[i] = 0.0

        # b) 차단 해제 전 ATK 금지
        if cons.get("must_break_before_first_atk", False) and s.opp_block:
            for i, c in enumerate(s.hand):
                if i in s.used:
                    continue
                if c.kind == "ATK":
                    mask[i] = 0.0

        # ----- Ojama 전용 순서 강제 -----
        if (self.puzzle_dict.get("name") == "[GX_Spirit_Caller]B03_Oh_Jama"):
            need = self._ojama_required_next()
            if need is not None:
                # 아직 스크립트 단계라면: 필요한 op만 허용, STOP/ATK 금지
                for i, c in enumerate(s.hand):
                    if i in s.used or mask[i] == 0.0:
                        continue
                    # 스크립트 op 추출
                    op = _get_op_from_card_obj_or_str(c)
                    if op != need:
                        mask[i] = 0.0
                    # 스크립트가 아니면(ATK/BUFF/HEAL 등) 막기
                    if getattr(c, "kind", "") != "SCRIPT":
                       mask[i] = 0.0
                mask[STOP_IDX] = 0.0
            else:
                # 세 스크립트 완료 후: ATK만 허용(+ STOP 정책은 아래 일반 규칙 적용)
                for i, c in enumerate(s.hand):
                    if i in s.used or mask[i] == 0.0:
                        continue
                    if getattr(c, "kind", "") != "ATK":
                        mask[i] = 0.0




        # ---- ATK 사용 횟수 제한 ----
        if atk_used >= atk_limit:
            stack_open = (len(s.stack) > 0)
            # 모든 카드 비활성
            mask[:] = 0.0
            # 스택이 닫혀 있을 때만 STOP 허용
            if not stack_open:
                if cons.get("must_end_with") == "ATK":
                    mask[STOP_IDX] = 1.0 if (self._last_used_kind() == "ATK") else 0.0
                else:
                    mask[STOP_IDX] = 1.0
            # 조기 반환 (ATK 제한에 걸렸다면 다른 규칙 볼 필요 없음)
            return mask

        # ---- STOP 정책(일반) ----
        # - must_end_with=ATK이면 마지막 카드가 ATK이고 스택이 비어야 STOP 허용
        # - 아니면 스택이 닫혀 있을 때 STOP 허용
        if cons.get("must_end_with") == "ATK":
            mask[STOP_IDX] = 1.0 if (self._last_used_kind() == "ATK" and len(s.stack) == 0) else 0.0
        else:
            mask[STOP_IDX] = 0.0 if len(s.stack) > 0 else 1.0

        # ---- 스택(체인) 규칙 ----
        stack_open = (len(s.stack) > 0)

        def _get_op_from_card(card) -> str:
            # card.script가 dict 또는 str일 수 있음
            script = getattr(card, "script", None)
            if isinstance(script, dict):
                return str(script.get("op", "")).strip().upper()
            if isinstance(script, str):
                tok = script.strip()
                return tok.split(":", 1)[0].strip().upper() if ":" in tok else tok.upper()
            return ""

        if stack_open:
            # 스택 열림: ATK/STOP 금지, SCRIPT만 조건에 따라 허용
            mask[STOP_IDX] = 0.0
            for i, c in enumerate(s.hand):
                if i in s.used or mask[i] == 0.0:
                    continue
                if c.kind == "ATK":
                    mask[i] = 0.0
                    continue
                if c.kind == "SCRIPT":
                    op = _get_op_from_card(c)
                    if op == "RESOLVE":
                        mask[i] = 1.0 if len(s.stack) > 0 else 0.0
                    elif op == "NEGATE":
                        # (예시) 속도 비교 규칙: 내 속도가 더 커야 Negate 허용
                        # 스택 top의 speed와 비교 (없으면 1)
                        my_spd = 1
                        top_spd = int(s.stack[-1].get("speed", 1)) if len(s.stack) > 0 else 1
                        # card.script가 dict면 speed 키 사용
                        if isinstance(getattr(c, "script", None), dict):
                            my_spd = int(c.script.get("speed", 1))
                        mask[i] = 1.0 if (my_spd > top_spd) else 0.0
                    elif op == "ACTIVATE":
                        mask[i] = 1.0  # 체인 위에 더 쌓는 건 허용
                    else:
                        mask[i] = 0.0  # 정의되지 않은 스크립트는 보수적으로 막음
                else:
                    # BUFF/HEAL/DRAW 등은 스택 열림 중에는 금지(체인 정리 우선)
                    mask[i] = 0.0
        else:
            # 스택 닫힘: RESOLVE/NEGATE 같은 스택 전용 스크립트는 비활성
            for i, c in enumerate(s.hand):
                if i in s.used or mask[i] == 0.0:
                    continue
                if c.kind == "SCRIPT":
                    op = _get_op_from_card(c)
                    if op in ("RESOLVE", "NEGATE"):
                        mask[i] = 0.0

            # STOP은 위의 정책에서 이미 설정됨
 
        # ---- 퍼즐의 script_sequence 기반 보조 제약 ----
        scripts = self.puzzle_dict.get("script_sequence") or []
        if scripts:
            # 예: 힐 금지 같은 전역 제약 키워드
            forbid_heal = False
            forbid_stop_when_chain = False

            for item in scripts:
                if isinstance(item, dict):
                    op = str(item.get("op", "")).strip().upper()
                elif isinstance(item, str):
                    tok = item.strip()
                    op = tok.split(":", 1)[0].strip().upper() if ":" in tok else tok.upper()
                else:
                    continue

                if op in ("OH_JAMA_HEAL_LIMIT", "CHAIN_HEAL_BLOCK"):
                    forbid_heal = True
                if op in ("CHAIN_NEGATE", "CHAIN_RESOLVE"):
                    forbid_stop_when_chain = True

            if forbid_heal:
                for i, c in enumerate(s.hand):
                    if i in s.used:
                        continue
                    if c.kind == "HEAL":
                        mask[i] = 0.0

            if forbid_stop_when_chain and len(s.stack) > 0:
                mask[STOP_IDX] = 0.0

        return mask

    """def action_mask(self) -> np.ndarray:
        A = self.max_hand + 1
        STOP_IDX = self.max_hand
        mask = np.zeros(A, dtype=np.float32)
        s = self.state
        cons = self.constraints or {}
        forbid = set((cons.get("forbid_kinds") or []))
        limit = cons.get("limit_kind_counts", {}) or {}
        atk_limit = int(limit.get("ATK", 10))
        atk_used = int(self._used_kind_counts().get("ATK", 0))

        # 기본: 아직 안 쓴 카드만 ON
        for i in range(len(s.hand)):
            if i not in s.used:
                mask[i] = 1.0

        # 패딩 구간은 0
        if len(s.hand) < self.max_hand:
            mask[len(s.hand):self.max_hand] = 0.0

        # STOP은 기본 OFF, 조건에 따라 나중에 켬
        mask[STOP_IDX] = 0.0

        # ---- 전역 규칙 마스크 ----
        # 1) 금지 종류 제거
        if forbid:
            for i, c in enumerate(s.hand):
                if i in s.used:
                    continue
                if c.kind in forbid:
                    mask[i] = 0.0

        # 2) 차단 해제 전 ATK 금지
        if cons.get("must_break_before_first_atk", False) and s.opp_block:
            for i, c in enumerate(s.hand):
                if i in s.used:
                    continue
                if c.kind == "ATK":
                    mask[i] = 0.0

        # 3) ATK 사용 횟수 제한
        if atk_used >= atk_limit:
            # 스택이 열려 있으면 STOP 불가(먼저 스택 정리 필요)
            stack_open = (len(s.stack) > 0)
            mask[:] = 0.0
            if not stack_open:
                # must_end_with=ATK면 마지막이 ATK일 때만 STOP 허용
                if cons.get("must_end_with") == "ATK":
                    mask[STOP_IDX] = 1.0 if (
    self._last_used_kind() == "ATK") else 0.0
                else:
                    mask[STOP_IDX] = 1.0
            return mask


        scripts = self.puzzle_dict.get("script_sequence") or []
        stack_open = len(s.stack) > 0
        for item in scripts:
            if isinstance(item, dict):
               op = str(item.get("op", "")).strip().upper()
            elif isinstance(item, str):
                tok = item.strip()
                op = tok.split(":", 1)[0].strip(
                ).upper() if ":" in tok else tok.upper()
            else:
               continue
        사용 가능 액션 마스크 생성.
        - 손패의 '아직 쓰지 않은 카드'는 1, 나머지는 0
        - STOP(=max_hand)은 기본 0에서 규칙/상태에 따라 1로 전환
        - forbid_kinds / must_break_before_first_atk / limit_kind_counts / must_end_with 적용
        - script_sequence 기반의 체인/힐 제한 등 보조 제약 적용
        return mask"""
    """def action_mask(self) -> np.ndarray:
        A = self.max_hand + 1
        STOP_IDX = self.max_hand
        mask = np.zeros(A, dtype=np.float32)

        # 안전장치
        s = self.state
        assert s is not None, "call reset() before action_mask()"

        cons = self.constraints or {}
        forbid = set(cons.get("forbid_kinds") or [])
        limit = cons.get("limit_kind_counts") or {}
        atk_limit = int(limit.get("ATK", 10))
        atk_used = int(self._used_kind_counts().get("ATK", 0))

        # 1) 기본: 아직 안 쓴 카드만 ON
        for i, c in enumerate(s.hand):
            if i not in s.used:
                mask[i] = 1.0

        # 2) 패딩 구간은 0 유지
        if len(s.hand) < self.max_hand:
            mask[len(s.hand):self.max_hand] = 0.0

        # 3) STOP은 기본 OFF (아래 규칙에서 켬/끔 결정)
        mask[STOP_IDX] = 0.0

        # ---- 전역 규칙 제약 ----
        # a) 금지 종류 제거
        if forbid:
            for i, c in enumerate(s.hand):
                if i in s.used:
                    continue
                if c.kind in forbid:
                    mask[i] = 0.0

        # b) 차단 해제 전 ATK 금지
        if cons.get("must_break_before_first_atk", False) and s.opp_block:
            for i, c in enumerate(s.hand):
                if i in s.used:
                    continue
                if c.kind == "ATK":
                    mask[i] = 0.0

        # c) ATK 사용 횟수 제한 (limit_kind_counts)
        if atk_used >= atk_limit:
            # 스택이 열려 있으면 STOP 불가(먼저 스택 정리 필요)
            stack_open = (len(s.stack) > 0)
            mask[:] = 0.0
            if not stack_open:
                if cons.get("must_end_with") == "ATK":
                    mask[STOP_IDX] = 1.0 if (
    self._last_used_kind() == "ATK") else 0.0
                else:
                    mask[STOP_IDX] = 1.0
              return mask  # 조기 반환

        # d) 그 외 STOP 정책
        # - must_end_with=ATK: 마지막 사용 카드가 ATK이고 스택이 비어야 STOP 허용
        # - 일반 퍼즐: 스택이 열려 있으면 STOP 금지, 아니면 허용
        if cons.get("must_end_with") == "ATK":
            mask[STOP_IDX] = 1.0 if (
    self._last_used_kind() == "ATK" and len(
        s.stack) == 0) else 0.0
        else:
            mask[STOP_IDX] = 0.0 if len(s.stack) > 0 else 1.0

        # ---- 스크립트 기반 보조 제약 (퍼즐별 특수 규칙) ----
        scripts = self.puzzle_dict.get("script_sequence") or []
        stack_open = len(s.stack) > 0

        for item in scripts:
            if isinstance(item, dict):
                op = str(item.get("op", "")).strip().upper()
            elif isinstance(item, str):
                tok = item.strip()
                op = tok.split(":", 1)[0].strip(
                ).upper() if ":" in tok else tok.upper()
            else:
                continue

            # 체인 처리 중이면 STOP 금지
            if op in ("CHAIN_NEGATE", "CHAIN_RESOLVE"):
                if stack_open:
                    mask[STOP_IDX] = 0.0

            # 퍼즐 스크립트에서 힐 금지 같은 제약을 명시했다면 적용
            if op in ("OH_JAMA_HEAL_LIMIT", "CHAIN_HEAL_BLOCK"):
                for i, c in enumerate(s.hand):
                    if i in s.used:
                        continue
                    if c.kind == "HEAL":
                        mask[i] = 0.0

        

        return mask

        # ---- 체인(스택) 규칙 마스크 ----
        stack_open = (len(s.stack) > 0)
        if stack_open:
            # 스택 열렸으면 ATK/STOP 금지, SCRIPT만 조건부 허용
            for i, c in enumerate(s.hand):
                if i in s.used or mask[i] == 0.0:
                    continue
                if c.kind == "ATK":
                    mask[i] = 0.0
                elif c.kind == "SCRIPT":
                    script = getattr(c, "script", None) or {}
                    op = script.get("op", "").upper()
                    if op == "RESOLVE":
                        # 스택 있을 때만 허용
                        mask[i] = 1.0
                    elif op == "NEGATE":
                        # 내 속도가 탑보다 커야 Negate 가능
                        my_spd = int(script.get("speed", 1))
                        top_spd = int(s.stack[-1].get("speed", 1))
                        mask[i] = 1.0 if (my_spd > top_spd) else 0.0
                    elif op == "ACTIVATE":
                        # 체인 위에 더 쌓는 건 허용
                        mask[i] = 1.0
                    else:
                        # 정의되지 않은 스크립트는 보수적으로 막음
                        mask[i] = 0.0
                else:
                    # BUFF/HEAL/DRAW 등은 스택 열림 중에는 금지(체인 정리 우선)
                    mask[i] = 0.0

            # 스택 열림이면 STOP 금지
            mask[STOP_IDX] = 0.0

        else:
            # 스택이 닫혀 있으면: RESOLVE/NEGATE 같은 스택 전용 카드는 비활성
            for i, c in enumerate(s.hand):
                if i in s.used or mask[i] == 0.0:
                    continue
                if c.kind == "SCRIPT":
                    script = getattr(c, "script", None) or {}
                    op = script.get("op", "").upper()
                    if op in ("RESOLVE", "NEGATE"):
                        mask[i] = 0.0

            # STOP 허용 여부
            if cons.get("must_end_with") == "ATK":
                mask[STOP_IDX] = 1.0 if (
    self._last_used_kind() == "ATK") else 0.0
            else:
                mask[STOP_IDX] = 1.0

        return mask"""

    # ---- 리셋/보상/종료 ----

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.state = make_state_from_dict(self.puzzle_dict)
        self._t = 0
        
        assert self.state is not None, "make_state_from_dict() returned None"
        self._init_lp = int(self.state.opp_lp)
        self._last_flags = {}
        if self.state is None:
            raise RuntimeError("reset() failed: state is None")

        self._last_flags = {}     # 에피소드 시작 시 초기화
        self._script_flags = {}   # 스크립트 상태도 초기화
        return self._obs(), {"action_mask": self.action_mask()}
    #        return self._obs(), {"action_mask": self.action_mask()}
    # def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    #    if seed is not None:
    #        np.random.seed(seed)
    #    self.state = make_state_from_dict(self.puzzle_dict)
    #    self._init_lp = int(self.state.opp_lp)
    #    self._t = 0
    #    return self._obs(), {"action_mask": self.action_mask()}


    """def _done(self) -> bool:
        s = self.state
        if s.opp_lp <= 0:
            return True
        if len(s.used) >= min(self.max_hand, len(s.hand)):
            return True
        # 안전장치로 최대 스텝 20
        if self._t >= 20:
        
            return True
        return False"""
    def _done(self) -> bool:
        s = self.state
        if s is None:                  # <-- 가드 추가
            return True
        if s.opp_lp <= 0:
            return True
        # (기존 종료조건 유지: 예) if self._t >= self.max_steps: return True
        return False


    def _terminal_success(self) -> bool:
        s = self.state
        # must_end_with 제약 확인
        must_end = self.constraints.get("must_end_with", None)
        if s.opp_lp > 0:
            return False
        if must_end == "ATK":
            return self._last_used_kind() == "ATK"
        return True

    def _reward(self, done: bool) -> float:
        s = self.state
        progress = (self._init_lp - s.opp_lp) / float(max(1, self._init_lp))
        r = - self.length_penalty + self.progress_coef * progress

        cons = self.constraints or {}
        # 마지막이 ATK인데 아직 안 끝냈으면 STOP 유도 보너스
        if cons.get("must_end_with") == "ATK" and self._last_used_kind() == "ATK" and not done:
            r += 0.15
        # ATK 제한 소진 이후 계속 질질 끌면 페널티
        limit = cons.get("limit_kind_counts", {}) or {}
        atk_limit = int(limit.get("ATK", 9999))
        atk_used = int(self._used_kind_counts().get("ATK", 0))
        if atk_used >= atk_limit and not done:
            r -= 0.20

        if done and self._terminal_success():
            r += 1.0
        return float(r)

    # ---- 환경 step ----
    """def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.state is not None, "call reset() first"
        STOP_IDX = self.max_hand

        # --- 전상태 스냅샷(셰이핑용) ---
        prev_stack_len = len(self.state.stack)
        prev_buff = int(self.state.my_buff)

        # STOP 처리
        if action == STOP_IDX:
            self._t += 1
            done = True
            reward = self._reward(done=True)  # STOP은 기본 보상만
            obs = self._obs()
            return obs, reward, done, {"stopped": True, "action_mask": self.action_mask()}

        # 잘못된 액션 처리
        legal = self.state.legal_actions()
        if action not in legal or action >= self.max_hand:
            obs = self._obs()
            return obs, -1.0, True, {"invalid_action": True, "action_mask": self.action_mask()}

        # 카드 효과 적용(여기서 self._last_flags를 갱신했다고 가정)
        self.state = self._env_step(self.state, action)
        self._t += 1

        # --- 보상 계산 ---
        done = self._done()

        # 1) 기본 보상
        #reward = float(self._reward(done))
        base_reward = self._reward(done)
        shaping = 0.0
        # ... 셰이핑 계산 ...
        reward = base_reward + shaping
        # 2) 플래그 보상 (self._env_step에서 셋업했다고 가정)
        flags = getattr(self, "_last_flags", {}) or {}
        if flags.get("played_script_activate"):
            reward += 0.10
        if flags.get("did_resolve") and flags.get("applied_buff"):
            reward += 0.15
        if flags.get("illegal_resolve_or_negate"):
            reward -= 0.15
        if flags.get("stack_before", 0) > 0 and flags.get("stack_after", 0) == 0:
            reward += 0.20

        # 3) 안전 셰이핑(관찰 기반)
        cur_stack_len = len(self.state.stack)
        cur_buff = int(self.state.my_buff)

        # 스택 길이 증가(체인 개시) → 약간의 보상
        if cur_stack_len > prev_stack_len:
            reward += 0.03

        # 스택 감소 + 버프 증가(해결이 버프로 이어짐) → 약간의 보상
        if cur_stack_len < prev_stack_len and cur_buff > prev_buff:
            reward += 0.06

        obs = self._obs()
        info = {"action_mask": self.action_mask()}
        return obs, reward, done, info"""
    def step(self, action: int):
        assert self.state is not None, "call reset() first"
        STOP_IDX = self.max_hand

        # --- 플래그 기본값 세팅 (없으면 생성) ---
        self._last_flags = {
            "played_script_activate": False,
            "did_resolve": False,
            "applied_buff": False,
            "illegal_resolve_or_negate": False,
            "stack_before": len(self.state.stack) if self.state is not None else 0,
            "stack_after": None,
        }

        prev_stack_len = len(self.state.stack)
        prev_buff = self.state.my_buff

        if action == STOP_IDX:
            self._t += 1
            done = True
            reward = self._reward(done=True)
            obs = self._obs()
            # 스톱 시점의 after 기록
            self._last_flags["stack_after"] = len(self.state.stack)
            return obs, reward, done, {"stopped": True, "action_mask": self.action_mask()}

        # 합법성 체크(네가 쓰는 가드 유지)
        s = self.state
        legal = s.legal_actions() if (s is not None and hasattr(s, "legal_actions")) else tuple()
        if action not in legal or action >= self.max_hand:
            obs = self._obs()
            # after 기록
            self._last_flags["stack_after"] = len(self.state.stack)
            return obs, -1.0, True, {"invalid_action": True, "action_mask": self.action_mask()}

        # --- 전이 ---
        new_state = self._env_step(self.state, action)
        if not isinstance(new_state, State):
            obs = self._obs()
            self._last_flags["stack_after"] = len(self.state.stack)
            return obs, -1.0, True, {"invalid_transition": True, "action_mask": self.action_mask()}
        self.state = new_state
        self._t += 1

        done = self._done()
        base_reward = self._reward(done)

        # 셰이핑
        shaping = 0.0
        cur_stack_len = len(self.state.stack)
        cur_buff = self.state.my_buff
        if cur_stack_len > prev_stack_len:
            shaping += 0.03
        if cur_stack_len < prev_stack_len and cur_buff > prev_buff:
            shaping += 0.06

        reward = base_reward + shaping

        # after 기록
        self._last_flags["stack_after"] = len(self.state.stack)

        obs = self._obs()
        info = {"action_mask": self.action_mask()}
        return obs, reward, done, info


    # ---- 실제 효과 적용 로직 ----
    # def _env_step(self, s: PuzzleState, a: int) -> PuzzleState:

    """def _env_step(self, s: State, action: int) -> State:
        used = tuple(list(s.used) + [action])

        c = s.hand[action]
        opp_lp = int(s.opp_lp)
        my_buff = int(s.my_buff)
        opp_block = bool(s.opp_block)

        kind = c.kind
        val = int(c.value)

        # --- 플래그 기본값 ---
        flags = {
            "played_script_activate": False,
            "did_resolve": False,          # ex) BLOCK_BREAK처럼 체인 해제/해결에 해당
            "applied_buff": False,
            "illegal_resolve_or_negate": False,
            "stack_before": len(s.stack)
        }

        if kind == "BUFF":
            my_buff += max(0, val)
            flags["applied_buff"] = True

        elif kind == "BLOCK_BREAK":
            opp_block = False
            flags["did_resolve"] = True

        elif kind == "ATK":
            dmg = max(0, val + my_buff)
            if opp_block:
                dmg = 0
                # 필요하면 잘못된 타이밍의 공격에 패널티
                # flags["illegal_resolve_or_negate"] = True
            opp_lp = max(0, opp_lp - dmg)

        elif kind == "HEAL":
            opp_lp = min(self._init_lp, opp_lp + max(0, val))

        elif kind == "DRAW":
            pass

        elif kind == "SCRIPT":
            # 카드에 script 문자열이 있어야 함
            scr = (c.script or "").strip().upper()
            if scr:
                # 스크립트 효과 적용
                new_state = self._apply_script_effect(s, scr)
                # 사용 리스트 업데이트
                return State(
                    opp_lp=new_state.opp_lp,
                    my_buff=new_state.my_buff,
                    opp_block=new_state.opp_block,
                    hand=new_state.hand,
                    used=tuple(list(s.used) + [action]),
                    stack=new_state.stack,
                    constraints=new_state.constraints,
                    turn=s.turn + 1,
                    my_lp=new_state.my_lp
                )
            # 스크립트 이름이 없다면 변화 없음(사용만 처리)
            return State(
                opp_lp=opp_lp,
                my_buff=my_buff,
                opp_block=opp_block,
                hand=s.hand,
                used=used,
                stack=s.stack,
                constraints=s.constraints,
                turn=s.turn + 1,
                my_lp=s.my_lp
            )"""
    def _ojama_required_next(self):
        """Ojama 전용: 다음에 반드시 내야 할 스크립트의 op를 반환. 없으면 None."""
        f = getattr(self, "_script_flags", {})
        # 아직 Trio 안 냈으면 Trio
        if not f.get("TRIO", False):
            return "OJAMA_TRIO"
        # Trio 냈고 Zero Gravity 안 냈으면 Zero Gravity
        if not f.get("ZERO", False):
            return "ZERO_GRAVITY"
        # Zero 냈고 Pill 안 냈으면 Pill
        if not f.get("PILL", False):
            return "BIG_EVOLUTION_PILL"
        # 셋 다 끝났으면 제한 없음(ATK 가능)
        return None



    def _env_step(self, s: State, action: int) -> State:
        used = tuple(list(s.used) + [action])
        c = s.hand[action]
        opp_lp    = int(s.opp_lp)
        my_lp     = int(s.my_lp)
        my_buff   = int(s.my_buff)
        opp_block = bool(s.opp_block)
        stack     = tuple(s.stack)
        new_stack = s.stack

        kind = c.kind
        val = int(getattr(c, "value", 0))

        if kind == "BUFF":
            my_buff += max(0, val)

        elif kind == "BLOCK_BREAK":
            opp_block = False

        elif kind == "ATK":
            dmg = max(0, val + my_buff)
            if opp_block:
                dmg = 0
            opp_lp = max(0, opp_lp - dmg)

        elif kind == "HEAL":
            opp_lp = min(self._init_lp, opp_lp + max(0, val))

        elif kind == "DRAW":
            pass

        elif kind == "SCRIPT":
            op = _get_op_from_card_obj_or_str(c)
            # ✅ 여기가 핵심: op를 안전하게 구한 뒤 사용
            if not hasattr(self, "_script_flags") or self._script_flags is None:
                self._script_flags = {}
            if not hasattr(self, "_last_flags") or self._last_flags is None:
                self._last_flags = {}

            op = _get_op_from_card_obj_or_str(c)
            # 필요시 스택 push(체인 모델링), 셰이핑 플래그 세팅
            # 기본적으로 ACTIVATE처럼 취급해서 스택 위에 쌓는 예시
            stack_list = list(s.stack)

            if op == "OJAMA_TRIO":
                self._script_flags["TRIO"] = True
                stack_list.append({"op": "ACTIVATE", "tag": "OJAMA_TRIO", "speed": 1})
                self._last_flags["played_script_activate"] = True
                new_stack = tuple(stack_list)

            elif op == "ZERO_GRAVITY":
                self._script_flags["ZERO"] = True
                stack_list.append({"op": "ACTIVATE", "tag": "ZERO_GRAVITY", "speed": 2})
                self._last_flags["played_script_activate"] = True
                new_stack = tuple(stack_list)

            elif op == "BIG_EVOLUTION_PILL":
                self._script_flags["PILL"] = True
                # 간단히 버프를 크게 주는 형태로 모델링 (원본 규칙에 맞게 조정 가능)
                my_buff += 6900
                stack_list.append({"op": "ACTIVATE", "tag": "BIG_EVOLUTION_PILL", "speed": 2})
                self._last_flags["played_script_activate"] = True
                new_stack = tuple(stack_list)

            else:
                # 알 수 없는 스크립트는 무효로 처리하거나 no-op
                # 여기서는 no-op (필요시 invalid 처리)
                pass

            lf = getattr(self, "_last_flags", {})
            lf["played_script_activate"] = True

        # 최종 상태 구성 (new_stack, 갱신된 변수 사용)
        return State(
            opp_lp=opp_lp,
            my_buff=my_buff,
            opp_block=opp_block,
            hand=s.hand,
            used=used,
            stack=new_stack,
            constraints=s.constraints,
            turn=s.turn + 1,
            my_lp=s.my_lp,
        )


        """elif kind == "SCRIPT":
            # 최소 VM: 오자마 3종만 처리
            sname = (getattr(c, "script", None) or "").upper()
            self._last_flags["played_script_activate"] = True  # shaping용

            if sname == "OJAMA_TRIO":
                # 상대 필드가 막혔다고 가정 → 이후 ATK가 막힐 수 있으니
                # 다음 스텝에 ZERO_GRAVITY로 차단 해제해야 함
                opp_block = True
                self._script_flags["TRIO"] = True

            elif sname == "ZERO_GRAVITY":
                # 차단 해제
                opp_block = False
                self._script_flags["GRAV"] = True
                # 해제 후 버프가 적용될 수 있는 상황이라고 표시(보상 셰이핑)
                self._last_flags["did_resolve"] = True

            elif sname == "BIG_EVOLUTION_PILL":
                # 마무리용 버프 크게 주기(4000 LP 컷을 넘기도록 충분히 크게)
                # Finisher ATK가 3000이므로 버프 7000이면 총 10000 데미지
                add = 7000
                my_buff += add
                self._script_flags["PILL"] = True
                self._last_flags["applied_buff"] = True

            else:
                # 기타 스크립트는 일단 무효(향후 확장 지점)
                pass"""




"""def _env_step(self, s: State, action: int) -> State:
        # 현재 상태 복사/초기화
        # hand = list(s.hand)  # (손패 변경이 없다면 굳이 안 써도 됨)
        used = tuple(list(s.used) + [action])

        c = s.hand[action]
        opp_lp    = int(s.opp_lp)
        my_buff   = int(s.my_buff)
        opp_block = bool(s.opp_block)

        kind = c.kind
        val  = int(c.value)

        if kind == "BUFF":
            my_buff += max(0, val)

        elif kind == "BLOCK_BREAK":
            opp_block = False

        elif kind == "ATK":
            dmg = max(0, val + my_buff)
            if opp_block:
                # 차단 중이면 데미지 0으로 막힘
                dmg = 0
            opp_lp = max(0, opp_lp - dmg)
            # 필요하면 버프를 사용 후 0으로
            # my_buff = 0

        elif kind == "HEAL":
            # 상대 회복(퍼즐 실패 유도용)
            opp_lp = min(self._init_lp, opp_lp + max(0, val))

        elif kind == "DRAW":
            # 데모에선 무효(실전은 hand 확장 로직 추가)
            pass

        elif kind == "SCRIPT":
            # 데모에선 무효(실전은 VM/스크립트 처리)
            pass

        # 새 상태 반환: new_* 같은 미정의 변수 쓰지 말고 위 로컬 변수 사용
        return State(
            opp_lp=opp_lp,
            my_buff=my_buff,
            opp_block=opp_block,
            hand=s.hand,          # 손패 변경 없다면 그대로 유지
            used=used,
            stack=s.stack,        # 스택 로직 없다면 그대로
            constraints=s.constraints,
            turn=s.turn + 1,
            my_lp=s.my_lp
        )"""
