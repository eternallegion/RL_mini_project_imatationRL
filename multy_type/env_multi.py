import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Optional, Union, List

from loader import find_puzzle_by_name


class MultiPuzzleEnv(gym.Env):
    """
    여러 퍼즐을 공용으로 돌리는 카드 퍼즐 환경.

    - 관측(observation): 고정 길이 10 벡터
        [ norm_lp, step_ratio, used_ratio, one-hot(last_kind 6종) ]
    - 행동(action):
        0 .. max_hand-1 : 해당 슬롯의 카드 사용
        max_hand        : STOP (체인 종료 및 데미지 계산)

    - 데미지 계산은 기본적으로 STOP 시점에만 수행.
    - 퍼즐마다 damage_rules / constraints 로 세부 규칙 설정.
    """

    metadata = {"render_modes": []}

    def __init__(self, puzzle_or_name: Union[Dict[str, Any], str], *, seed: Optional[int] = None):
        super().__init__()

        # 퍼즐 로딩
        if isinstance(puzzle_or_name, str):
            self.puzzle: Dict[str, Any] = find_puzzle_by_name(puzzle_or_name)
        else:
            self.puzzle = dict(puzzle_or_name)

        # 고정 파라미터
        self.max_obs_len: int = 10
        self.max_steps: int = int(self.puzzle.get("max_steps", 12))
        self.raw_hand: List[Dict[str, Any]] = list(self.puzzle.get("hand", []))

        # max_hand: 액션 슬롯 수 (STOP 전까지 카드 슬롯 개수)
        self.max_hand: int = int(self.puzzle.get("max_hand", max(5, min(10, len(self.raw_hand)))))
        self.STOP_IDX: int = self.max_hand  # 마지막 인덱스가 STOP

        # 관측/행동 공간 정의
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_obs_len,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_hand + 1)  # +1은 STOP

        # RNG 및 기타 상태
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        self._seed = seed

        # hand_len 은 reset 이전에 action_mask가 불릴 수 있으므로 기본값 설정
        self.hand_len: int = 0



        # 나머지 상태는 reset에서 설정
        self.reset()

    # ------------------------------------------------------------------
    # Gymnasium 스타일 reset
    # ------------------------------------------------------------------


    def _card_kind(self, idx: int) -> str:
        """
        주어진 action index에 해당하는 카드의 kind 문자열을 반환.
        - STOP 인덱스면 "STOP"
        - 범위 밖이면 "PAD"
        - 그 외에는 hand[idx]["kind"] (기본 "PAD")
        """
        if idx == self.STOP_IDX:
            return "STOP"
        if idx < 0 or idx >= self.max_hand:
            return "PAD"
        return self.hand[idx].get("kind", "PAD")



    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self._seed = seed

        # 에피소드 상태
        self.steps: int = 0
        self.done: bool = False
        self.truncated: bool = False

        # LP 및 룰
        self.opp_lp: int = int(self.puzzle.get("opp_lp", 4000))
        self.start_opp_lp: int = self.opp_lp

        self.damage_rules: Dict[str, Any] = dict(self.puzzle.get("damage_rules", {}))
        self.constraints: Dict[str, Any] = dict(self.puzzle.get("constraints", {}))

        # 핸드 세팅 (PAD 로 패딩 혹은 잘라내기)
        self.hand: List[Dict[str, Any]] = list(self.raw_hand)
        if len(self.hand) < self.max_hand:
            pad_n = self.max_hand - len(self.hand)
            self.hand.extend([{"kind": "PAD"} for _ in range(pad_n)])
        elif len(self.hand) > self.max_hand:
            self.hand = self.hand[: self.max_hand]

        self.hand_len = len(self.hand)

        # 사용 이력
        self.used_cards: set[int] = set()      # 사용된 슬롯 인덱스
        self.used_seq_kinds: List[str] = []    # 사용 순서대로 kind 기록

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    # ------------------------------------------------------------------
    # 내부: 관측 벡터 구성
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        # 0~1 로 정규화된 LP, 스텝 정보
        norm_lp = 0.0
        if self.start_opp_lp > 0:
            norm_lp = np.clip(self.opp_lp / float(self.start_opp_lp), 0.0, 1.0)

        step_ratio = np.clip(self.steps / float(max(self.max_steps, 1)), 0.0, 1.0)
        used_ratio = np.clip(len(self.used_seq_kinds) / float(max(self.max_hand, 1)), 0.0, 1.0)

        # 마지막 사용 카드 kind one-hot
        kinds = ["ZERO", "PILL", "TRIO", "BUFF", "BLOCK", "ATK"]
        one_hot = np.zeros(len(kinds), dtype=np.float32)
        if self.used_seq_kinds:
            last = self.used_seq_kinds[-1]
            if last in kinds:
                one_hot[kinds.index(last)] = 1.0

        vec = np.concatenate(
            [
                np.array([norm_lp, step_ratio, used_ratio], dtype=np.float32),
                one_hot,
            ]
        )

        # max_obs_len 에 맞춰 패딩/자르기
        if vec.shape[0] < self.max_obs_len:
            vec = np.pad(vec, (0, self.max_obs_len - vec.shape[0]), mode="constant")
        elif vec.shape[0] > self.max_obs_len:
            vec = vec[: self.max_obs_len]

        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # 내부: 체인에 따른 데미지 계산 (STOP 에서 호출)
    # ------------------------------------------------------------------
    def _apply_damage_from_sequence(self):
        """
        used_seq_kinds 에 기록된 카드 시퀀스로 데미지를 계산하여 self.opp_lp 감소.
        기본 규칙:
          - damage_rules.atk_base: ATK 한 번의 기본 데미지
          - buff_multiplier: BUFF 가 하나라도 있으면 ATK에 곱해짐
          - trio_multiplier: TRIO 가 하나라도 있으면 ATK에 곱해짐
          - pill_bonus: PILL 이 하나라도 있으면 고정 추가 데미지
        """
        rules = self.damage_rules
        base = float(rules.get("atk_base", 0.0))
        buff_mult = float(rules.get("buff_multiplier", 1.0))
        trio_mult = float(rules.get("trio_multiplier", 1.0))
        pill_bonus = float(rules.get("pill_bonus", 0.0))

        seq = list(self.used_seq_kinds)

        # 버프/트리거 여부
        has_buff = any(k == "BUFF" for k in seq)
        has_trio = any(k == "TRIO" for k in seq)
        has_pill = any(k == "PILL" for k in seq)

        atk_cnt = sum(1 for k in seq if k == "ATK")

        total_damage = 0.0
        if base > 0.0 and atk_cnt > 0:
            mult = 1.0
            if has_buff:
                mult *= buff_mult
            if has_trio:
                mult *= trio_mult
            total_damage += base * atk_cnt * mult

        if has_pill and pill_bonus != 0.0:
            total_damage += pill_bonus

        if total_damage > 0.0:
            self.opp_lp = max(0, int(self.opp_lp - total_damage))

        # 특수 제약: zero_only_win
        #   - constraints.zero_only_win == True 이고
        #   - NON-PAD 카드가 오직 ZERO 하나뿐일 때만 무조건 승리로 간주
        if self.constraints.get("zero_only_win", False):
            non_pad = [k for k in seq if k != "PAD"]
            if len(non_pad) == 1 and non_pad[0] == "ZERO":
                self.opp_lp = 0  # 강제 킬

    # ------------------------------------------------------------------
    # 행동 마스크 (sb3-contrib ActionMasker 에서 사용)
    # ------------------------------------------------------------------
    def action_mask(self) -> np.ndarray:
        """
        shape = (max_hand + 1, )

        - 아직 안쓴 카드 + PAD가 아닌 슬롯만 1
        - constraints.forbid_kinds 에 있는 kind는 0
        - constraints.must_break_before_first_atk:
            ZERO를 한 번도 안 썼으면 ATK는 모두 0
        - constraints.limit_kind_counts: {"ATK":2} 같이 kind별 최대 사용 횟수
        - STOP(=self.STOP_IDX):
            * 카드를 한 장이라도 사용했거나
            * 플레이 가능한 카드가 없을 때 1
        - 마스크가 전부 0이면 STOP만 1로 강제
        """
        mask = np.zeros(self.max_hand + 1, dtype=np.float32)

        # 1) 기본: 사용 안 했고, PAD 아닌 슬롯만 1
        for i in range(self.hand_len):
            if i in self.used_cards:
                continue
            kind = self.hand[i].get("kind", "PAD")
            if kind == "PAD":
                continue
            mask[i] = 1.0

        # 2) forbid_kinds
        forbid = set(self.constraints.get("forbid_kinds") or [])
        if forbid:
            for i in range(self.hand_len):
                if mask[i] == 1.0 and self.hand[i].get("kind") in forbid:
                    mask[i] = 0.0

        # 3) must_break_before_first_atk: ZERO 나오기 전에는 ATK 금지
        if self.constraints.get("must_break_before_first_atk", False):
            zero_used = any(k == "ZERO" for k in self.used_seq_kinds)
            if not zero_used:
                for i in range(self.hand_len):
                    if mask[i] == 1.0 and self.hand[i].get("kind") == "ATK":
                        mask[i] = 0.0

        # 4) limit_kind_counts: kind별 사용 횟수 상한
        limit = self.constraints.get("limit_kind_counts") or {}
        for k_kind, max_cnt in limit.items():
            used_cnt = sum(1 for kk in self.used_seq_kinds if kk == k_kind)
            if used_cnt >= int(max_cnt):
                for i in range(self.hand_len):
                    if mask[i] == 1.0 and self.hand[i].get("kind") == k_kind:
                        mask[i] = 0.0

        # 5) STOP 허용 여부
        any_card_played = len(self.used_seq_kinds) > 0
        any_playable = bool(mask[: self.max_hand].any())

        if any_card_played or not any_playable:
            mask[self.STOP_IDX] = 1.0
        else:
            mask[self.STOP_IDX] = 0.0

        # 6) 전부 0 이면 STOP 강제 허용
        if not mask.any():
            mask[self.STOP_IDX] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Gymnasium 스타일 step
    # ------------------------------------------------------------------
    '''def step(self, action: int):
        """
        returns: obs, reward, terminated, truncated, info
        """

        if self.done or self.truncated:
            # 이미 끝난 에피소드면 변화 없이 0 보상
            return self._get_obs(), 0.0, self.done, self.truncated, {}

        self.steps += 1
        reward = 0.0
        info: Dict[str, Any] = {}

        # 1) STOP 액션: 체인을 종료하고 데미지 계산
        if action == self.STOP_IDX:
            # 체인 기반 데미지 적용
            self._apply_damage_from_sequence()

            if self.opp_lp <= 0:
                reward = 1.0
            else:
                reward = -1.0

            self.done = True
            obs = self._get_obs()
            return obs, reward, self.done, self.truncated, info

        # 2) 범위 밖 / 잘못된 액션 -> 패널티 후 종료
        if action < 0 or action >= self.max_hand:
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, self.done, self.truncated, info

        if action in self.used_cards:
            # 이미 사용한 슬롯 다시 사용 -> 실패
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, self.done, self.truncated, info

        card = self.hand[action]
        kind = card.get("kind", "PAD")

        if kind == "PAD":
            # PAD 를 고르면 의미 없는 행동이므로 실패 처리
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, self.done, self.truncated, info

        # 3) 정상 카드 사용
        self.used_cards.add(action)
        self.used_seq_kinds.append(kind)

        # 스텝 제한 초과 시 자동 실패
        if self.steps >= self.max_steps:
            self.truncated = False
            self.done = True
            reward = -1.0
            return self._get_obs(), reward, self.done, self.truncated, info

        # 중간 스텝에서는 보상은 0 (에피소드 유지만 함)
        obs = self._get_obs()
        return obs, reward, self.done, self.truncated, info'''
    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action"
        if self.done:
            # 이미 끝난 에피소드면 아무 변화 없음
            return self._get_obs(), 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        truncated = False

        # 1) STOP 액션 처리
        if action == self.STOP_IDX:
            self.done = True

            # 퍼즐 JSON에 win_seq_exact 가 있으면,
            # LP 계산 대신 "정확한 시퀀스 퍼즐"로 처리
            win_seq = self.puzzle.get("win_seq_exact")

            if win_seq is not None:
                # 정확히 일치해야만 성공
                if self.used_seq_kinds == list(win_seq):
                    # 보기 좋게 LP도 0으로 만들자
                    self.opp_lp = 0
                    reward = 1.0
                else:
                    # 정답 시퀀스와 다르면 무조건 실패
                    reward = -1.0

            else:
                # 기존 방식: 사용한 시퀀스를 기반으로 데미지 계산
                self._apply_damage_from_sequence()
                if self.opp_lp <= 0:
                    reward = 1.0
                else:
                    reward = -1.0

            obs = self._get_obs()
            return obs, reward, self.done, truncated, {}

        # 2) STOP 이 아닌 카드 선택
        if action in self.used_cards:
            # 이미 사용한 카드를 또 고르면 즉시 실패
            self.done = True
            reward = -1.0
            obs = self._get_obs()
            return obs, reward, self.done, truncated, {}

        if action < self.max_hand:
            card = self.hand[action]
            kind = card.get("kind", "PAD")
            self.used_cards.add(action)           # set 이니까 add 사용
            # 시퀀스 기록
            self.used_seq_kinds.append(kind)

        else:
            # 이 경우는 이론상 오면 안 되지만 방어용
            self.done = True
            reward = -1.0
            obs = self._get_obs()
            return obs, reward, self.done, truncated, {}

        # 3) 스텝 수 한도 체크
        if self.steps >= self.max_steps:
            self.done = True
            # 아직 STOP 안 눌렀는데 max_steps 도달 → 실패
            reward = -1.0

        obs = self._get_obs()
        return obs, reward, self.done, truncated, {}

    # ------------------------------------------------------------------
    # 렌더/클로즈 (필요시 확장)
    # ------------------------------------------------------------------
    def render(self):
        # 간단한 디버그용 출력만 필요하면 여기에 구현
        pass

    def close(self):
        pass
