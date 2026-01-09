"""
Yu-Gi-Oh Puzzle Duel Simulator - Action Definitions
"""

from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass

class ActionType(Enum):
    ACTIVATE_TRAP = "activate_trap"
    ACTIVATE_SPELL = "activate_spell"
    SUMMON = "summon"
    SPECIAL_SUMMON = "special_summon"
    ATTACK = "attack"
    CHANGE_PHASE = "change_phase"
    END_TURN = "end_turn"

@dataclass
class Action:
    """Represents a game action"""
    action_type: ActionType
    card_index: Optional[int] = None  # Index in hand or field
    target_index: Optional[int] = None  # Target monster zone, etc.
    zone_index: Optional[int] = None  # Which zone to use
    
    def __repr__(self):
        return f"Action({self.action_type.value}, card={self.card_index}, target={self.target_index}, zone={self.zone_index})"

class ActionSpace:
    """Defines all possible actions in the game"""
    
    @staticmethod
    def get_all_possible_actions() -> list:
        """Get all theoretically possible actions (for neural network output size)"""
        actions = []
        
        # Activate trap from spell/trap zones (5 zones)
        for i in range(5):
            actions.append(Action(ActionType.ACTIVATE_TRAP, zone_index=i))
        
        # Activate spell from hand (max 5 cards in hand for this puzzle)
        for i in range(5):
            actions.append(Action(ActionType.ACTIVATE_SPELL, card_index=i))
        
        # Attack with monster (5 monster zones) to opponent monster (5 zones) or direct
        for i in range(5):
            for j in range(6):  # 5 opponent monsters + 1 direct attack
                actions.append(Action(ActionType.ATTACK, card_index=i, target_index=j))
        
        # Change phase
        actions.append(Action(ActionType.CHANGE_PHASE))
        
        # End turn
        actions.append(Action(ActionType.END_TURN))
        
        return actions
    
    @staticmethod
    def get_action_space_size() -> int:
        """Get total number of possible actions"""
        # 5 traps + 5 spells + (5 monsters * 6 targets) + 1 phase change + 1 end turn
        return 5 + 5 + 30 + 1 + 1  # = 42 actions
    
    @staticmethod
    def action_to_index(action: Action) -> int:
        """Convert action to index for neural network"""
        if action.action_type == ActionType.ACTIVATE_TRAP:
            return action.zone_index
        elif action.action_type == ActionType.ACTIVATE_SPELL:
            return 5 + action.card_index
        elif action.action_type == ActionType.ATTACK:
            return 10 + (action.card_index * 6) + action.target_index
        elif action.action_type == ActionType.CHANGE_PHASE:
            return 40
        elif action.action_type == ActionType.END_TURN:
            return 41
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
    
    @staticmethod
    def index_to_action(index: int) -> Action:
        """Convert index to action"""
        if index < 5:
            return Action(ActionType.ACTIVATE_TRAP, zone_index=index)
        elif index < 10:
            return Action(ActionType.ACTIVATE_SPELL, card_index=index - 5)
        elif index < 40:
            attack_index = index - 10
            card_index = attack_index // 6
            target_index = attack_index % 6
            return Action(ActionType.ATTACK, card_index=card_index, target_index=target_index)
        elif index == 40:
            return Action(ActionType.CHANGE_PHASE)
        elif index == 41:
            return Action(ActionType.END_TURN)
        else:
            raise ValueError(f"Invalid action index: {index}")
    
    @staticmethod
    def get_valid_actions(game_state) -> list:
        """Get list of valid actions given current game state"""
        from state import GameState
        valid_actions = []
        
        # Main Phase 1: Can activate spells/traps
        if game_state.phase == "main1":
            # Activate traps
            for i, card in enumerate(game_state.player.spell_trap_zones):
                if card is not None:
                    valid_actions.append(Action(ActionType.ACTIVATE_TRAP, zone_index=i))
            
            # Activate spells from hand
            for i, card in enumerate(game_state.player.hand):
                if card and card.card_type.value == "spell":
                    valid_actions.append(Action(ActionType.ACTIVATE_SPELL, card_index=i))
            
            # Change to battle phase
            valid_actions.append(Action(ActionType.CHANGE_PHASE))
        
        # Battle Phase: Can attack
        elif game_state.phase == "battle":
            for i, monster in enumerate(game_state.player.monster_zones):
                if monster is not None:
                    # Attack each opponent monster
                    for j, opp_monster in enumerate(game_state.opponent.monster_zones):
                        if opp_monster is not None:
                            valid_actions.append(Action(ActionType.ATTACK, card_index=i, target_index=j))
                    
                    # Direct attack if no opponent monsters
                    if game_state.count_opponent_monsters() == 0:
                        valid_actions.append(Action(ActionType.ATTACK, card_index=i, target_index=5))
            
            # Change to main phase 2 or end
            valid_actions.append(Action(ActionType.CHANGE_PHASE))
        
        # Main Phase 2 or End Phase
        elif game_state.phase in ["main2", "end"]:
            valid_actions.append(Action(ActionType.END_TURN))
        
        return valid_actions
    
    @staticmethod
    def get_valid_action_mask(game_state) -> list:
        """
        Get boolean mask for all 42 actions (True = valid, False = invalid)
        Used by neural network for action masking
        """
        valid_actions = ActionSpace.get_valid_actions(game_state)
        mask = [False] * ActionSpace.get_action_space_size()  # 42 actions
        
        for action in valid_actions:
            idx = ActionSpace.action_to_index(action)
            mask[idx] = True
        
        return mask
