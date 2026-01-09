"""
Ancient Kings 퍼즐 전용 환경

퍼즐 정보:
- Player LP: 100
- Opponent LP: 4200
- 목표: 1턴킬

정답:
1. Big Evolution Pill - Mammoth 제물 → Megazowler 소환
2. Mystik Wok - Megazowler 제물 → DEF(2000) LP 회복 (100→2100)
3. Confiscation - 1000 LP 지불 → Kuriboh 버림 (2100→1100)
4. Ultimate Offering - 500 LP → Mad Sword Beast 소환 (1100→600)
5. Ultimate Offering - 500 LP → Ultimate Tyranno 소환 (600→100)
6. Battle Phase
7. Attack: Mad Sword Beast(1400) + Ultimate Tyranno(3000) = 4400 > 4200 → 승리!
"""

from typing import Tuple, Dict
import numpy as np
from env_base import YuGiOhPuzzleEnvBase
from state import create_ancient_kings_puzzle, GameState
from card import Position, UltimateTyranno, Megazowler, MadSwordBeast
from actions import ActionType, ActionSpace

class AncientKingsEnv(YuGiOhPuzzleEnvBase):
    """
    Ancient Kings 퍼즐 환경
    
    핵심 전략:
    - 정확한 5단계 시퀀스 필수!
    - LP 관리가 핵심: 100 → 2100 → 1100 → 600 → 100
    - Kuriboh 제거 필수 (직접 공격 무효화 방지)
    """
    
    def __init__(self, max_steps: int = 30, reward_shaping: bool = True):
        super().__init__(max_steps, reward_shaping)
        
        # 시퀀스 추적 (5단계)
        self.step1_big_evo_pill = False      # Big Evolution Pill
        self.step2_mystik_wok = False         # Mystik Wok
        self.step3_confiscation = False       # Confiscation
        self.step4_ultimate_offering_1 = False # Ultimate Offering 1st
        self.step5_ultimate_offering_2 = False # Ultimate Offering 2nd
        
        # 상태 추적
        self.prev_player_lp = 100
        self.kuriboh_removed = False
        self.megazowler_summoned = False
        self.mad_sword_summoned = False
        self.tyranno_summoned = False
        self.ultimate_offering_count = 0
        self.sequence_bonus_given = False

        self.victory_ready = False
        self.victory_ready_bonus_given = False
    
    @property
    def puzzle_name(self) -> str:
        return "Ancient Kings"
    
    @property
    def initial_opponent_lp(self) -> int:
        return 4200
    
    def create_puzzle(self) -> GameState:
        return create_ancient_kings_puzzle()
    
    def _reset_puzzle_state(self):
        """Ancient Kings 전용 상태 초기화"""
        self.step1_big_evo_pill = False
        self.step2_mystik_wok = False
        self.step3_confiscation = False
        self.step4_ultimate_offering_1 = False
        self.step5_ultimate_offering_2 = False
        
        self.prev_player_lp = 100
        self.kuriboh_removed = False
        self.megazowler_summoned = False
        self.mad_sword_summoned = False
        self.tyranno_summoned = False
        self.ultimate_offering_count = 0
        self.sequence_bonus_given = False


        self.victory_ready = False
        self.victory_ready_bonus_given = False

    
    def get_valid_actions(self) -> np.ndarray:
        """
        Ancient Kings 퍼즐 전용 액션 마스크
        - 기본 룰(듀얼 규칙)은 그대로 두고
        - 퍼즐 정답 시퀀스를 벗어나는 액션만 추가로 막는다.
        """
        # 1) 기본 마스크 (듀얼 규칙 기반)
        base_mask = super().get_valid_actions()
        mask = base_mask.copy()

        game = self.simulator.game_state

        # 헬퍼 함수들
        def is_confiscation(card) -> bool:
            return card is not None and "Confiscation" in card.name

        def is_big_evo_pill(card) -> bool:
            return card is not None and "Big Evolution Pill" in card.name

        # 현재 필드 총 공격력
        total_atk = sum(m.atk for m in game.player.monster_zones if m)

        # "이기기 직전" 상태 판단
        victory_ready = (
            total_atk >= 4200
            and self.kuriboh_removed
            and self.mad_sword_summoned
            and self.tyranno_summoned
        )

        # 2) 각 액션에 대해 퍼즐 전용 마스킹 적용
        for idx, valid in enumerate(mask):
            if not valid:
                continue

            action = ActionSpace.index_to_action(idx)

            # --- [규칙 1] Mystik Wok은 Big Evolution Pill 이후에만 허용 ---
            #   정답: Step 1이 Big Evolution Pill → Megazowler 소환
            #   → 그 전에 Mystik Wok으로 Mammoth를 제물로 쓰면 퍼즐이 사실상 터짐
            if (
                action.action_type == ActionType.ACTIVATE_TRAP
                and action.zone_index == 2   # S/T Zone 2 → Mystik Wok
                and not self.step1_big_evo_pill
            ):
                mask[idx] = False
                continue

            # --- [규칙 2] Confiscation은 Mystik Wok 이후 + LP 충분할 때만 ---
            #   정답: LP 2100 만든 다음 1000 LP 지불 → 1100으로 Kuriboh 제거
            if action.action_type == ActionType.ACTIVATE_SPELL and action.card_index is not None:
                if action.card_index < len(game.player.hand):
                    card = game.player.hand[action.card_index]
                    if is_confiscation(card):
                        # Mystik Wok을 안 썼거나, LP가 1000 미만이면 금지
                        if (not self.step2_mystik_wok) or (game.player.lp < 1000):
                            mask[idx] = False
                            continue

            # --- [규칙 3] Ultimate Offering은 Kuriboh 제거 후에만 + 최대 2번 ---
            #   정답: Step 4,5에서 딱 2번 사용
            if (
                action.action_type == ActionType.ACTIVATE_TRAP
                and action.zone_index == 3   # S/T Zone 3 → Ultimate Offering
            ):
                # Kuriboh 제거 전에 쓰면, 공격이 막혀서 퍼즐 실패 경로
                if not self.step3_confiscation:
                    mask[idx] = False
                    continue

                # 정답 시퀀스에서는 2번만 사용
                if self.ultimate_offering_count >= 2:
                    mask[idx] = False
                    continue

            # --- [규칙 4] 승리 조건이 갖춰진 뒤에는 공격/페이즈 체인지만 허용 ---
            #   Mad Sword Beast + Ultimate Tyranno + Kuriboh 제거 + ATK ≥ 4200
            #   → 이 상태에서 스펠/트랩/소환/엔드턴은 모두 쓸모없는 액션
            if victory_ready:
                if action.action_type not in (ActionType.ATTACK, ActionType.CHANGE_PHASE):
                    mask[idx] = False
                    continue

        if not mask.any():
            if self.verbose:
                print("[WARN] AncientKingsEnv] all actions masked out -> fallback to base_mask")
            return base_mask

        return mask


    def calculate_shaped_reward(self, action_obj, base_reward: float) -> float:
        """
        Ancient Kings 퍼즐 - 시퀀스 기반 Reward Shaping
        
        정답 시퀀스:
        1. Big Evolution Pill → Megazowler
        2. Mystik Wok → LP 회복
        3. Confiscation → Kuriboh 제거
        4. Ultimate Offering → Mad Sword Beast
        5. Ultimate Offering → Ultimate Tyranno
        """
        # 기본 shaped reward
        shaped_reward = self.calculate_base_shaped_reward(base_reward)
        
        game = self.simulator.game_state
        current_player_lp = game.player.lp
        
        # ============================================================
        # Step 1: Big Evolution Pill (Megazowler 소환)
        # ============================================================
        if action_obj.action_type == ActionType.ACTIVATE_SPELL:
            card_idx = action_obj.card_index
            
            # Big Evolution Pill (초기 hand index 2)
            if not self.step1_big_evo_pill:
                # Megazowler가 필드에 있는지 확인
                megazowler_found = False
                tyranno_found = False
                for monster in game.player.monster_zones:
                    if monster and isinstance(monster, Megazowler):
                        megazowler_found = True
                    if monster and isinstance(monster, UltimateTyranno):
                        tyranno_found = True
                
                if megazowler_found:
                    self.step1_big_evo_pill = True
                    self.megazowler_summoned = True
                    shaped_reward += 50.0  # 30 → 50
                    if self.verbose:
                        print("  -> STEP 1 BONUS: Big Evolution Pill → Megazowler! (+50)")
                elif tyranno_found and not self.step1_big_evo_pill:
                    # 잘못된 소환! Tyranno를 먼저 소환하면 Mystik Wok 콤보 불가
                    shaped_reward -= 50.0
                    if self.verbose:
                        print("  -> PENALTY: Big Evolution Pill → Ultimate Tyranno (wrong order!) (-50)")
            
            # Confiscation
            if card_idx < len(game.player.hand):
                card = game.player.hand[card_idx] if card_idx < len(game.player.hand) else None
                if card and 'Confiscation' in card.name:
                    pass  # Confiscation은 아래에서 처리
        
        # ============================================================
        # Step 2: Mystik Wok (LP 회복) - 가장 중요!
        # ============================================================
        if action_obj.action_type == ActionType.ACTIVATE_TRAP:
            zone_idx = action_obj.zone_index
            
            # Mystik Wok (zone 2)
            if zone_idx == 2 and not self.step2_mystik_wok:
                lp_gained = current_player_lp - self.prev_player_lp
                if lp_gained > 0:
                    self.step2_mystik_wok = True
                    
                    # Step 1 후에 사용했으면 시퀀스 보너스
                    if self.step1_big_evo_pill:
                        shaped_reward += 80.0  # 50 → 80 (핵심 콤보!)
                        if self.verbose:
                            print("  -> STEP 2 BIG BONUS: Mystik Wok after Big Evolution Pill! (+80)")
                    else:
                        # 순서 틀림 - 큰 페널티 (Megazowler 없이 Mammoth 제물)
                        shaped_reward -= 50.0  # -20 → -50
                        if self.verbose:
                            print("  -> BIG PENALTY: Mystik Wok used before Big Evolution Pill! (-50)")
            
            # Ultimate Offering (zone 3)
            elif zone_idx == 3:
                self.ultimate_offering_count += 1
                
                # ⭐ Ultimate Offering 2번 초과 사용 → 페널티 (LP 낭비)
                if self.ultimate_offering_count > 2:
                    shaped_reward -= 30.0
                    if self.verbose:
                        print(f"  -> PENALTY: Ultimate Offering used {self.ultimate_offering_count} times (max 2)! (-30)")
                    # 더 이상 처리하지 않음
                    self.prev_player_lp = current_player_lp
                    return shaped_reward
                
                # LP가 부족하면 Ultimate Offering 사용 불가!
                if current_player_lp < 500:
                    shaped_reward -= 20.0
                    if self.verbose:
                        print("  -> PENALTY: Not enough LP for Ultimate Offering! (-20)")
                
                # ⭐ Confiscation 없이 Ultimate Offering 사용 → 큰 페널티!
                if not self.step3_confiscation and self.step2_mystik_wok:
                    shaped_reward -= 80.0  # 큰 페널티!
                    if self.verbose:
                        print("  -> BIG PENALTY: Ultimate Offering used before Confiscation! Kuriboh will block! (-80)")
                
                if self.ultimate_offering_count == 1 and not self.step4_ultimate_offering_1:
                    # Mad Sword Beast 소환 확인
                    for monster in game.player.monster_zones:
                        if monster and isinstance(monster, MadSwordBeast):
                            self.step4_ultimate_offering_1 = True
                            self.mad_sword_summoned = True
                            
                            # Step 3 (Confiscation) 후면 보너스
                            if self.step3_confiscation:
                                shaped_reward += 60.0  # 40 → 60
                                if self.verbose:
                                    print("  -> STEP 4 BONUS: Ultimate Offering → Mad Sword Beast! (+60)")
                            else:
                                # Kuriboh 제거 안 하고 소환 → 보너스 없음
                                if self.verbose:
                                    print("  -> Step 4: Mad Sword Beast summoned (but Kuriboh blocks attack!)")
                            break
                
                elif self.ultimate_offering_count == 2 and not self.step5_ultimate_offering_2:
                    # Ultimate Tyranno 소환 확인
                    for monster in game.player.monster_zones:
                        if monster and isinstance(monster, UltimateTyranno):
                            self.step5_ultimate_offering_2 = True
                            self.tyranno_summoned = True
                            
                            # Step 4 후면 보너스
                            if self.step4_ultimate_offering_1:
                                shaped_reward += 70.0  # 50 → 70
                                if self.verbose:
                                    print("  -> STEP 5 BONUS: Ultimate Offering → Ultimate Tyranno! (+70)")
                            break
        #=======================================================================================================이거 추가
        def check_victory(self):
            game = self.simulator.game_state
            total_atk = sum(m.atk for m in game.player.monster_zones if m)
            return (total_atk >= 4200 
                    and self.kuriboh_removed 
                    and self.mad_sword_summoned 
                    and self.tyranno_summoned)

        # 승리 조건 충족했는데 Change Phase가 아닌 다른 액션 → 페널티
        if self.victory_ready and action_obj.action_type != ActionType.CHANGE_PHASE:
            if action_obj.action_type == ActionType.END_TURN:
                shaped_reward -= 100.0
                if self.verbose:
                    print("  -> HUGE PENALTY: Victory ready but ended turn! (-100)")
            elif action_obj.action_type != ActionType.ATTACK:  # 공격은 OK
                shaped_reward -= 20.0
                if self.verbose:
                    print("  -> PENALTY: Victory ready but not entering Battle Phase! (-20)")

        #=========================여기까지 추가됨


        # ============================================================
        # 승리 조건 충족 후 Battle Phase 유도
        # ============================================================
        total_atk = sum(m.atk for m in game.player.monster_zones if m)
        victory_ready = (total_atk >= 4200 and self.kuriboh_removed and 
                        self.mad_sword_summoned and self.tyranno_summoned)
        
        # 승리 조건 충족했는데 Change Phase가 아닌 다른 액션 → 페널티
        if victory_ready and action_obj.action_type != ActionType.CHANGE_PHASE:
            if action_obj.action_type == ActionType.END_TURN:
                # END_TURN은 최악의 선택! 이길 수 있는데 턴을 넘김
                shaped_reward -= 100.0
                if self.verbose:
                    print("  -> HUGE PENALTY: Victory ready but ended turn! (-100)")
            elif action_obj.action_type != ActionType.ATTACK:  # 공격은 OK
                shaped_reward -= 20.0
                if self.verbose:
                    print("  -> PENALTY: Victory ready but not entering Battle Phase! (-20)")
        
        # ============================================================
        # Step 3: Confiscation (Kuriboh 제거)
        # ============================================================
        if len(game.opponent.hand) == 0 and not self.kuriboh_removed:
            self.kuriboh_removed = True
            self.step3_confiscation = True
            
            # Step 2 후에 사용했으면 시퀀스 보너스
            if self.step2_mystik_wok:
                shaped_reward += 40.0
                if self.verbose:
                    print("  -> STEP 3 BONUS: Confiscation removed Kuriboh! (+40)")
        
        # ============================================================
        # 완벽한 시퀀스 보너스
        # ============================================================
        if (self.step1_big_evo_pill and self.step2_mystik_wok and 
            self.step3_confiscation and self.step4_ultimate_offering_1 and 
            self.step5_ultimate_offering_2 and not self.sequence_bonus_given):
            
            self.sequence_bonus_given = True
            shaped_reward += 100.0
            if self.verbose:
                print("  -> PERFECT SEQUENCE BONUS! All 5 steps completed! (+100)")
        
        # ============================================================
        # 필드 상태 확인 (승리 가능 여부)
        # ============================================================
        total_atk = sum(m.atk for m in game.player.monster_zones if m)


        if (
            self.mad_sword_summoned
            and self.tyranno_summoned
            and total_atk >= 4200
            and self.kuriboh_removed
        ):
            # 🔹 승리 준비 상태로 처음 진입했을 때만 보너스 1회 지급
            if not self.victory_ready_bonus_given:
                shaped_reward += 30.0
                self.victory_ready_bonus_given = True
                if self.verbose:
                    print(
                        f"  -> VICTORY READY! Total ATK {total_atk} >= 4200, "
                        f"Kuriboh removed! (+30, first time only)"
                    )

        
        # 두 몬스터 모두 있고 ATK >= 4200
        if self.mad_sword_summoned and self.tyranno_summoned:
            if total_atk >= 4200 and self.kuriboh_removed:
                shaped_reward += 30.0
                if self.verbose:
                    print(f"  -> VICTORY READY! Total ATK {total_atk} >= 4200, Kuriboh removed! (+30)")
        
        # ============================================================
        # 직접 공격 보너스 (Kuriboh 제거 후)
        # ============================================================
        if action_obj.action_type == ActionType.ATTACK:
            if action_obj.target_index == 5:  # Direct attack
                if self.kuriboh_removed:
                    shaped_reward += 20.0
                    if self.verbose:
                        print("  -> BONUS: Safe direct attack! (+20)")
                else:
                    shaped_reward -= 30.0
                    if self.verbose:
                        print("  -> PENALTY: Direct attack but Kuriboh can block! (-30)")
        
        # 상태 저장
        self.prev_player_lp = current_player_lp
        
        return shaped_reward


# 하위 호환성을 위한 별칭
AncientKingsPuzzleEnv = AncientKingsEnv
