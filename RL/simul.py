"""
Yu-Gi-Oh Puzzle Duel Simulator - Game Logic

[변경사항]
1. _attack(): OjamaToken 파괴 시 상대(컨트롤러)에게 300 데미지
2. _attack(): UltimateTyranno가 몬스터 공격 시 attacked_monster_this_turn = True
3. _attack(): UltimateTyranno 직접공격 시 attacked_monster_this_turn 체크
4. _change_phase(): Battle Phase 진입 시 attacked_monster_this_turn 리셋
"""

from state import GameState, print_game_state
from actions import Action, ActionType
from card import (Position, OjamaTrio, ZeroGravity, BigEvolutionPill, 
                   BlackStego, UltimateTyranno, MammothGraveyard, Megazowler,
                   MystikWok, Confiscation, UltimateOffering, MadSwordBeast, 
                   Monster, OjamaToken, ReflectBounder)
from typing import Tuple, Optional

class YuGiOhSimulator:
    """Simulates Yu-Gi-Oh puzzle duels"""
    
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.verbose = False
        
    def reset(self, game_state: GameState):
        """Reset simulator with new game state"""
        self.game_state = game_state
        
    def step(self, action: Action) -> Tuple[GameState, float, bool, dict]:
        """Execute action and return (next_state, reward, done, info)"""
        reward = 0.0
        done = False
        info = {}
        
        valid_actions = self._get_valid_actions()
        action_valid = any(
            a.action_type == action.action_type and
            a.card_index == action.card_index and
            a.target_index == action.target_index and
            a.zone_index == action.zone_index
            for a in valid_actions
        )
        
        if not action_valid:
            return self.game_state.clone(), -5.0, False, {'error': 'Invalid action'}
        
        try:
            if action.action_type == ActionType.ACTIVATE_TRAP:
                reward = self._activate_trap(action)
            elif action.action_type == ActionType.ACTIVATE_SPELL:
                reward = self._activate_spell(action)
            elif action.action_type == ActionType.ATTACK:
                reward = self._attack(action)
            elif action.action_type == ActionType.CHANGE_PHASE:
                reward = self._change_phase()
            elif action.action_type == ActionType.END_TURN:
                reward = self._end_turn()
            else:
                reward = -5.0
                info['error'] = 'Unknown action type'
        except Exception as e:
            reward = -5.0
            info['error'] = str(e)
            if self.verbose:
                print(f"Error executing action: {e}")
        
        if self.game_state.is_terminal():
            done = True
            winner = self.game_state.get_winner()
            if winner == 0:
                reward += 100.0
                info['result'] = 'win'
            else:
                reward -= 100.0
                info['result'] = 'lose'
        
        return self.game_state.clone(), reward, done, info
    
    def _get_valid_actions(self):
        from actions import ActionSpace
        return ActionSpace.get_valid_actions(self.game_state)
    
    def _activate_trap(self, action: Action) -> float:
        zone_index = action.zone_index
        card = self.game_state.player.spell_trap_zones[zone_index]
        
        if card is None:
            return -5.0
        
        if self.verbose:
            print(f"Activating trap/spell: {card.name}")
        
        if isinstance(card, OjamaTrio):
            success = card.activate(self.game_state)
            if success:
                self.game_state.player.spell_trap_zones[zone_index] = None
                if self.verbose:
                    print(f"  -> Summoned 3 Ojama Tokens to opponent's field")
                return 10.0
        
        elif isinstance(card, ZeroGravity):
            card.activate(self.game_state)
            self.game_state.player.spell_trap_zones[zone_index] = None
            if self.verbose:
                print(f"  -> All monsters switched battle positions")
            return 10.0
        
        elif isinstance(card, MystikWok):
            tribute_monster = None
            for monster in self.game_state.player.monster_zones:
                if monster:
                    tribute_monster = monster
                    break
            
            if tribute_monster:
                lp_before = self.game_state.player.lp
                success = card.activate(self.game_state, tribute_monster, use_def=True)
                if success:
                    self.game_state.player.spell_trap_zones[zone_index] = None
                    if self.verbose:
                        print(f"  -> Tributed {tribute_monster.name}, gained {self.game_state.player.lp - lp_before} LP")
                        print(f"  -> LP: {lp_before} -> {self.game_state.player.lp}")
                    return 12.0
        
        elif isinstance(card, UltimateOffering):
            if self.game_state.player.lp > 500 and len(self.game_state.player.hand) > 0:
                # 패에서 소환 가능한 몬스터 찾기
                for i, hand_card in enumerate(self.game_state.player.hand):
                    if isinstance(hand_card, Monster):
                        # Big Evolution Pill 효과: 공룡족은 제물 없이 소환 가능
                        is_dinosaur = isinstance(hand_card, (UltimateTyranno, Megazowler, MadSwordBeast, BlackStego, MammothGraveyard))
                        
                        # Ultimate Tyranno는 보통 2제물 필요하지만 Big Evolution Pill 효과로 무시
                        if isinstance(hand_card, UltimateTyranno):
                            if not self.game_state.big_evolution_pill_active:
                                continue  # Pill 효과 없으면 Tyranno 소환 불가
                        
                        lp_before = self.game_state.player.lp
                        success = card.activate(self.game_state, i)
                        if success:
                            # Ultimate Offering은 영구 함정이라 필드에서 제거하지 않음
                            if self.verbose:
                                print(f"  -> Paid 500 LP, summoned {hand_card.name}")
                                print(f"  -> LP: {lp_before} -> {self.game_state.player.lp}")
                                if is_dinosaur and self.game_state.big_evolution_pill_active:
                                    print(f"  -> (Big Evolution Pill effect: No tribute needed)")
                            return 10.0
                        break
        
        return -2.0
    
    def _activate_spell(self, action: Action) -> float:
        card_index = action.card_index
        
        if card_index >= len(self.game_state.player.hand):
            return -5.0
        
        card = self.game_state.player.hand[card_index]
        
        if card is None:
            return -5.0
        
        if self.verbose:
            print(f"Activating spell: {card.name}")
        
        if isinstance(card, BigEvolutionPill):
            dinosaur_field = None
            is_ancient_kings = False
            
            for monster in self.game_state.player.monster_zones:
                if monster and isinstance(monster, (BlackStego, MammothGraveyard, Megazowler)):
                    dinosaur_field = monster
                    # Mammoth Graveyard가 있으면 Ancient Kings 퍼즐
                    if isinstance(monster, MammothGraveyard):
                        is_ancient_kings = True
                    break
            
            summon_monster = None
            
            if is_ancient_kings:
                # Ancient Kings: Megazowler 우선 소환 (Mystik Wok 콤보용)
                for hand_card in self.game_state.player.hand:
                    if hand_card and isinstance(hand_card, Megazowler):
                        summon_monster = hand_card
                        break
                # Megazowler 없으면 Ultimate Tyranno
                if summon_monster is None:
                    for hand_card in self.game_state.player.hand:
                        if hand_card and isinstance(hand_card, UltimateTyranno):
                            summon_monster = hand_card
                            break
            else:
                # Oh Jama: Ultimate Tyranno 소환 (공격용)
                for hand_card in self.game_state.player.hand:
                    if hand_card and isinstance(hand_card, UltimateTyranno):
                        summon_monster = hand_card
                        break
                # Ultimate Tyranno 없으면 Megazowler
                if summon_monster is None:
                    for hand_card in self.game_state.player.hand:
                        if hand_card and isinstance(hand_card, Megazowler):
                            summon_monster = hand_card
                            break
            
            if dinosaur_field and summon_monster:
                success = card.activate(self.game_state, dinosaur_field, summon_monster)
                if success:
                    self.game_state.player.hand.remove(card)
                    # Big Evolution Pill 지속 효과 활성화!
                    self.game_state.big_evolution_pill_active = True
                    if self.verbose:
                        print(f"  -> Tributed {dinosaur_field.name}, summoned {summon_monster.name}")
                        print(f"  -> Big Evolution Pill effect: Dinosaurs can be summoned without tribute this turn!")
                    return 15.0
        
        elif isinstance(card, Confiscation):
            if self.game_state.player.lp > 1000 and len(self.game_state.opponent.hand) > 0:
                discarded_card = self.game_state.opponent.hand[0] if self.game_state.opponent.hand else None
                lp_before = self.game_state.player.lp
                success = card.activate(self.game_state, 0)
                if success:
                    self.game_state.player.hand.remove(card)
                    if self.verbose:
                        print(f"  -> Paid 1000 LP, discarded opponent's {discarded_card.name if discarded_card else 'card'}")
                        print(f"  -> LP: {lp_before} -> {self.game_state.player.lp}")
                    return 12.0
        
        return -2.0
    
    def _attack(self, action: Action) -> float:
        """Execute an attack"""
        if self.game_state.phase != "battle":
            return -5.0
        
        attacker_index = action.card_index
        target_index = action.target_index
        
        attacker = self.game_state.player.monster_zones[attacker_index]
        
        if attacker is None:
            return -5.0
        
        if attacker.position != Position.FACEUP_ATTACK:
            return -5.0
        
        # Check if already attacked (Ultimate Tyranno can attack multiple monsters)
        if not isinstance(attacker, UltimateTyranno) and attacker.has_attacked:
            return -5.0
        
        # Direct attack (target_index == 5)
        if target_index == 5:
            # [수정] 상대 몬스터가 있으면 직접 공격 불가
            if self.game_state.count_opponent_monsters() > 0:
                return -5.0
            
            # [추가] UltimateTyranno는 몬스터를 공격했으면 직접공격 불가
            if isinstance(attacker, UltimateTyranno) and attacker.attacked_monster_this_turn:
                if self.verbose:
                    print(f"{attacker.name} cannot attack directly after attacking a monster!")
                return -5.0
            
            damage = attacker.atk
            self.game_state.opponent.lp -= damage
            attacker.has_attacked = True
            if self.verbose:
                print(f"{attacker.name} attacks directly for {damage} damage!")
                print(f"  -> Opponent LP: {self.game_state.opponent.lp + damage} -> {self.game_state.opponent.lp}")
            return 20.0
        
        # Attack opponent monster
        else:
            target = self.game_state.opponent.monster_zones[target_index]
            
            if target is None:
                return -5.0
            
            if self.verbose:
                print(f"{attacker.name} (ATK {attacker.atk}) attacks {target.name}!")
            
            # Mark as attacked
            if not isinstance(attacker, UltimateTyranno):
                attacker.has_attacked = True
            else:
                # [추가] UltimateTyranno가 몬스터를 공격하면 직접공격 불가
                attacker.attacked_monster_this_turn = True
            
            # Battle calculation
            if target.position == Position.FACEUP_ATTACK:
                # [추가] Reflect Bounder 효과: 공격받으면 공격 몬스터 ATK만큼 반사 데미지
                if isinstance(target, ReflectBounder):
                    reflect_damage = attacker.atk
                    self.game_state.player.lp -= reflect_damage
                    if self.verbose:
                        print(f"  -> Reflect Bounder effect! {reflect_damage} damage to player!")
                        print(f"  -> Player LP: {self.game_state.player.lp + reflect_damage} -> {self.game_state.player.lp}")
                    
                    # 반사 데미지로 플레이어 패배 체크
                    if self.game_state.player.lp <= 0:
                        if self.verbose:
                            print(f"  -> Player LP reduced to 0! Player loses!")
                        return -100.0  # 큰 패널티
                
                if attacker.atk >= target.atk:
                    damage = attacker.atk - target.atk
                    self.game_state.opponent.lp -= damage
                    self.game_state.opponent.monster_zones[target_index] = None
                    
                    # [추가] OjamaToken 파괴 시 컨트롤러(상대)에게 300 데미지
                    if isinstance(target, OjamaToken) and hasattr(target, 'destroy_damage'):
                        self.game_state.opponent.lp -= target.destroy_damage
                        if self.verbose:
                            print(f"  -> {target.name} destroyed! {damage} battle damage dealt.")
                            print(f"  -> {target.name} effect: {target.destroy_damage} damage to opponent!")
                            print(f"  -> Opponent LP: {self.game_state.opponent.lp + damage + target.destroy_damage} -> {self.game_state.opponent.lp}")
                    else:
                        if self.verbose:
                            print(f"  -> {target.name} destroyed! {damage} damage dealt.")
                            if damage > 0:
                                print(f"  -> Opponent LP: {self.game_state.opponent.lp + damage} -> {self.game_state.opponent.lp}")
                    return 8.0
                else:
                    damage = target.atk - attacker.atk
                    self.game_state.player.lp -= damage
                    self.game_state.player.monster_zones[attacker_index] = None
                    if self.verbose:
                        print(f"  -> {attacker.name} destroyed! {damage} damage taken.")
                    return -10.0
            
            elif target.position == Position.FACEUP_DEFENSE:
                if attacker.atk > target.defense:
                    self.game_state.opponent.monster_zones[target_index] = None
                    
                    # [추가] OjamaToken 파괴 시 컨트롤러(상대)에게 300 데미지
                    if isinstance(target, OjamaToken) and hasattr(target, 'destroy_damage'):
                        self.game_state.opponent.lp -= target.destroy_damage
                        if self.verbose:
                            print(f"  -> {target.name} (DEF {target.defense}) destroyed!")
                            print(f"  -> {target.name} effect: {target.destroy_damage} damage to opponent!")
                            print(f"  -> Opponent LP: {self.game_state.opponent.lp + target.destroy_damage} -> {self.game_state.opponent.lp}")
                    else:
                        if self.verbose:
                            print(f"  -> {target.name} (DEF {target.defense}) destroyed!")
                    return 5.0
                elif attacker.atk == target.defense:
                    if self.verbose:
                        print(f"  -> No effect (ATK = DEF).")
                    return 1.0
                else:
                    damage = target.defense - attacker.atk
                    self.game_state.player.lp -= damage
                    if self.verbose:
                        print(f"  -> Attack failed! {damage} damage taken.")
                    return -8.0
        
        return 0.0
    
    def _change_phase(self) -> float:
        if self.game_state.phase == "main1":
            self.game_state.phase = "battle"
            # Reset attack flags when entering battle phase
            for monster in self.game_state.player.monster_zones:
                if monster:
                    monster.has_attacked = False
                    # [추가] UltimateTyranno의 attacked_monster_this_turn 리셋
                    if isinstance(monster, UltimateTyranno):
                        monster.attacked_monster_this_turn = False
            if self.verbose:
                print("=== Entering Battle Phase ===")
            return 2.0
        elif self.game_state.phase == "battle":
            self.game_state.phase = "main2"
            if self.verbose:
                print("=== Entering Main Phase 2 ===")
            return -1.0
        elif self.game_state.phase == "main2":
            self.game_state.phase = "end"
            if self.verbose:
                print("=== Entering End Phase ===")
            return -2.0
        return -5.0
    
    def _end_turn(self) -> float:
        if self.verbose:
            print("=== Turn ended ===")
        self.game_state.turn += 1
        self.game_state.phase = "main1"
        return -50.0
    
    def get_state_vector(self) -> list:
        state = []
        state.append(self.game_state.player.lp / 10000.0)
        state.append(self.game_state.opponent.lp / 10000.0)
        
        for i in range(5):
            if i < len(self.game_state.player.hand) and self.game_state.player.hand[i]:
                card = self.game_state.player.hand[i]
                state.extend([1, 0, 0] if card.card_type.value == "monster" else
                           [0, 1, 0] if card.card_type.value == "spell" else
                           [0, 0, 1])
            else:
                state.extend([0, 0, 0])
        
        for monster in self.game_state.player.monster_zones:
            if monster:
                state.append(1)
                state.append(monster.atk / 5000.0)
                state.append(1 if monster.position == Position.FACEUP_ATTACK else 0)
            else:
                state.extend([0, 0, 0])
        
        for card in self.game_state.player.spell_trap_zones:
            state.append(1 if card else 0)
        
        for monster in self.game_state.opponent.monster_zones:
            if monster:
                state.append(1)
                state.append(monster.atk / 5000.0)
                state.append(1 if monster.position == Position.FACEUP_ATTACK else 0)
            else:
                state.extend([0, 0, 0])
        
        phase_encoding = {
            "main1": [1, 0, 0, 0],
            "battle": [0, 1, 0, 0],
            "main2": [0, 0, 1, 0],
            "end": [0, 0, 0, 1]
        }
        state.extend(phase_encoding.get(self.game_state.phase, [0, 0, 0, 0]))
        
        return state
    
    @staticmethod
    def get_state_size() -> int:
        return 56
