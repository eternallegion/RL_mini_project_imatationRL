"""
Oh Jama 퍼즐 전용 환경

퍼즐 정보:
- Player LP: 100
- Opponent LP: 9900
- 목표: 1턴킬

정답:
1. Ojama Trio → Token 3개 수비표시 소환
2. Zero Gravity → 모든 몬스터 포지션 변경
3. Big Evolution Pill → Ultimate Tyranno 소환
4. Battle Phase → Token 공격으로 승리
"""

from typing import Tuple, Dict
import numpy as np

from env_base import YuGiOhPuzzleEnvBase
from state import create_oh_jama_puzzle, GameState
from actions import ActionType
from card import Position, ReflectBounder, OjamaToken


class OhJamaEnv(YuGiOhPuzzleEnvBase):
    """
    Oh Jama 퍼즐 환경
    
    핵심 전략:
    - Ojama Trio → Zero Gravity → Big Evolution Pill 순서가 중요!
    - Zero Gravity로 Token을 공격표시로, Reflect Bounder를 수비표시로
    - 공격표시 Token 공격 시 3000 + 300 = 3300 데미지
    """
    
    def __init__(self, max_steps: int = 30, reward_shaping: bool = True):
        super().__init__(max_steps, reward_shaping)
        
        # Oh Jama 전용 상태 추적
        self.ojama_trio_used = False
        self.zero_gravity_used = False
        self.big_evo_pill_used = False
        self.correct_sequence_bonus_given = False
        self.prev_token_in_attack = 0
        self.prev_bounder_safe = False
        self.perfect_state_bonus_given = False
    
    @property
    def puzzle_name(self) -> str:
        return "Oh Jama"
    
    @property
    def initial_opponent_lp(self) -> int:
        return 9900
    
    def create_puzzle(self) -> GameState:
        return create_oh_jama_puzzle()
    
    def _reset_puzzle_state(self):
        """Oh Jama 전용 상태 초기화"""
        self.ojama_trio_used = False
        self.zero_gravity_used = False
        self.big_evo_pill_used = False
        self.correct_sequence_bonus_given = False
        self.prev_token_in_attack = 0
        self.prev_bounder_safe = False
        self.perfect_state_bonus_given = False
    
    def calculate_shaped_reward(self, action_obj, base_reward: float) -> float:
        """
        Oh Jama 퍼즐 전용 Reward Shaping
        
        핵심:
        1. Ojama Trio를 먼저 사용하면 보너스
        2. Zero Gravity를 Ojama Trio 후, Big Evolution Pill 전에 사용하면 큰 보너스
        3. 잘못된 순서는 페널티
        4. Token 공격표시, Bounder 수비표시 상태에 보너스
        """
        # 기본 shaped reward 계산
        shaped_reward = self.calculate_base_shaped_reward(base_reward)
        
        # ============================================================
        # 시퀀스 기반 보상
        # ============================================================
        
        if action_obj.action_type == ActionType.ACTIVATE_TRAP:
            zone_idx = action_obj.zone_index
            
            # Zone 2 = Ojama Trio
            if zone_idx == 2 and not self.ojama_trio_used:
                self.ojama_trio_used = True
                # Ojama Trio를 먼저 사용하면 보너스
                if not self.zero_gravity_used and not self.big_evo_pill_used:
                    shaped_reward += 20.0
                    if self.verbose:
                        print("  -> BONUS: Ojama Trio used FIRST! (+20)")
            
            # Zone 1 = Zero Gravity
            elif zone_idx == 1 and not self.zero_gravity_used:
                self.zero_gravity_used = True
                
                # Zero Gravity를 Ojama Trio 후, Big Evolution Pill 전에 사용
                if self.ojama_trio_used and not self.big_evo_pill_used:
                    if not self.correct_sequence_bonus_given:
                        shaped_reward += 100.0
                        self.correct_sequence_bonus_given = True
                        if self.verbose:
                            print("  -> BIG BONUS: Correct sequence! Ojama Trio → Zero Gravity (before Pill) (+100)")
                
                # Big Evolution Pill 후에 Zero Gravity 사용 → 큰 페널티!
                elif self.big_evo_pill_used:
                    shaped_reward -= 80.0
                    if self.verbose:
                        print("  -> BIG PENALTY: Zero Gravity used AFTER Big Evolution Pill! (-80)")
                
                # Zero Gravity를 먼저 사용 → 페널티
                elif not self.ojama_trio_used:
                    shaped_reward -= 30.0
                    if self.verbose:
                        print("  -> PENALTY: Zero Gravity used FIRST! (-30)")
        
        elif action_obj.action_type == ActionType.ACTIVATE_SPELL:
            # Big Evolution Pill (hand index 1)
            if action_obj.card_index == 1 and not self.big_evo_pill_used:
                self.big_evo_pill_used = True
                
                # 완벽한 순서: Ojama Trio → Zero Gravity → Big Evolution Pill
                if self.ojama_trio_used and self.zero_gravity_used:
                    shaped_reward += 50.0
                    if self.verbose:
                        print("  -> BONUS: Big Evolution Pill used AFTER Zero Gravity! (+50)")
                
                # Zero Gravity 후지만 Ojama Trio 없음
                elif self.zero_gravity_used:
                    shaped_reward += 20.0
                    if self.verbose:
                        print("  -> BONUS: Big Evolution Pill used after Zero Gravity (+20)")
                
                # Zero Gravity 전에 사용 → 페널티!
                else:
                    shaped_reward -= 50.0
                    if self.verbose:
                        print("  -> PENALTY: Big Evolution Pill used BEFORE Zero Gravity! (-50)")
        
        # ============================================================
        # 필드 상태 기반 보상
        # ============================================================
        
        # Reflect Bounder 수비표시 확인
        reflect_bounder_safe = False
        for monster in self.simulator.game_state.opponent.monster_zones:
            if monster and isinstance(monster, ReflectBounder):
                if monster.position == Position.FACEUP_DEFENSE:
                    reflect_bounder_safe = True
                    break
        
        # Ojama Token 공격표시 개수 확인
        token_in_attack = 0
        for monster in self.simulator.game_state.opponent.monster_zones:
            if monster and isinstance(monster, OjamaToken):
                if monster.position == Position.FACEUP_ATTACK:
                    token_in_attack += 1
        
        # 새로운 Token이 공격표시가 됨
        if token_in_attack > self.prev_token_in_attack:
            new_tokens = token_in_attack - self.prev_token_in_attack
            shaped_reward += 50.0 * new_tokens
            if self.verbose:
                print(f"  -> BONUS: {new_tokens} Token(s) now in ATTACK! (+{50*new_tokens})")
        
        # Reflect Bounder가 수비로 전환됨
        if reflect_bounder_safe and not self.prev_bounder_safe:
            shaped_reward += 30.0
            if self.verbose:
                print("  -> BONUS: Reflect Bounder now in DEFENSE! (+30)")
        
        # 완벽한 상태 달성 (한 번만)
        if reflect_bounder_safe and token_in_attack >= 3:
            if not self.perfect_state_bonus_given:
                shaped_reward += 50.0
                self.perfect_state_bonus_given = True
                if self.verbose:
                    print("  -> PERFECT SETUP BONUS! (+50)")
        
        # 상태 저장
        self.prev_token_in_attack = token_in_attack
        self.prev_bounder_safe = reflect_bounder_safe
        
        return shaped_reward


# 하위 호환성을 위한 별칭
OhJamaPuzzleEnv = OhJamaEnv
