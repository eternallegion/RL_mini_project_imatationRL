"""
Yu-Gi-Oh Puzzle Duel Simulator - Card Definitions
순환 import 문제 해결된 버전
"""

from enum import Enum
from typing import Optional


class CardType(Enum):
    MONSTER = "monster"
    SPELL = "spell"
    TRAP = "trap"


class Position(Enum):
    FACEUP_ATTACK = "faceup_attack"
    FACEUP_DEFENSE = "faceup_defense"
    FACEDOWN_ATTACK = "facedown_attack"
    FACEDOWN_DEFENSE = "facedown_defense"


class Card:
    def __init__(self, card_id: int, name: str, card_type: CardType):
        self.card_id = card_id
        self.name = name
        self.card_type = card_type
        self.position = Position.FACEDOWN_ATTACK
        
    def __repr__(self):
        return f"{self.name} (ID: {self.card_id})"


class Monster(Card):
    def __init__(self, card_id: int, name: str, atk: int, defense: int):
        super().__init__(card_id, name, CardType.MONSTER)
        self.atk = atk
        self.defense = defense
        self.has_attacked = False


class Spell(Card):
    def __init__(self, card_id: int, name: str):
        super().__init__(card_id, name, CardType.SPELL)
        

class Trap(Card):
    def __init__(self, card_id: int, name: str):
        super().__init__(card_id, name, CardType.TRAP)


# ============================================
# Oh Jama Puzzle Cards
# ============================================

class BlackStego(Monster):
    """Black Stego - 1200 ATK / 2000 DEF"""
    def __init__(self):
        super().__init__(79409334, "Black Stego", 1200, 2000)


class UltimateTyranno(Monster):
    """Ultimate Tyranno - 3000 ATK / 2200 DEF
    Effect: Must attack all opponent monsters once each.
            Cannot attack directly if it attacked a monster this turn.
    """
    def __init__(self):
        super().__init__(84808313, "Ultimate Tyranno", 3000, 2200)
        self.must_attack_all = True
        self.attacked_monster_this_turn = False


class OjamaToken(Monster):
    """Ojama Token - 0 ATK / 1000 DEF
    When destroyed: Inflict 300 damage to this card's controller.
    """
    def __init__(self):
        super().__init__(12482652, "Ojama Token", 0, 1000)
        self.destroy_damage = 300


class ReflectBounder(Monster):
    """Reflect Bounder - 1700 ATK / 1000 DEF"""
    def __init__(self):
        super().__init__(2851070, "Reflect Bounder", 1700, 1000)
    def on_attacked(self, attacker):
        # 공격자의 ATK만큼 상대에게 데미지
        attacker.controller.take_damage(attacker.atk)


class OjamaTrio(Trap):
    """Ojama Trio - Special Summon 3 Ojama Tokens to opponent's field"""
    def __init__(self):
        super().__init__(83133491, "Ojama Trio")
    
    def activate(self, game_state):
        tokens_summoned = 0
        for i in range(5):
            if game_state.opponent.monster_zones[i] is None:
                token = OjamaToken()
                token.position = Position.FACEUP_DEFENSE
                game_state.opponent.monster_zones[i] = token
                tokens_summoned += 1
                if tokens_summoned == 3:
                    break
        return tokens_summoned == 3


class ZeroGravity(Trap):
    """Zero Gravity - Change all face-up monsters' battle positions"""
    def __init__(self):
        super().__init__(29843091, "Zero Gravity")
    
    def activate(self, game_state):
        for monster in game_state.player.monster_zones:
            if monster and monster.position in [Position.FACEUP_ATTACK, Position.FACEUP_DEFENSE]:
                if monster.position == Position.FACEUP_ATTACK:
                    monster.position = Position.FACEUP_DEFENSE
                else:
                    monster.position = Position.FACEUP_ATTACK
        
        for monster in game_state.opponent.monster_zones:
            if monster and monster.position in [Position.FACEUP_ATTACK, Position.FACEUP_DEFENSE]:
                if monster.position == Position.FACEUP_ATTACK:
                    monster.position = Position.FACEUP_DEFENSE
                else:
                    monster.position = Position.FACEUP_ATTACK
        return True


class BigEvolutionPill(Spell):
    """Big Evolution Pill - Tribute 1 Dinosaur; Special Summon 1 Dinosaur from hand"""
    def __init__(self):
        super().__init__(511003009, "Big Evolution Pill")
    
    def activate(self, game_state, tribute_monster, summon_monster):
        for i, monster in enumerate(game_state.player.monster_zones):
            if monster and monster.card_id == tribute_monster.card_id:
                game_state.player.monster_zones[i] = None
                break
        
        for i, card in enumerate(game_state.player.hand):
            if card and card.card_id == summon_monster.card_id:
                game_state.player.hand.pop(i)
                break
        
        for i in range(5):
            if game_state.player.monster_zones[i] is None:
                summon_monster.position = Position.FACEUP_ATTACK
                game_state.player.monster_zones[i] = summon_monster
                return True
        return False


# ============================================
# Ancient Kings Puzzle Cards
# ============================================

class MammothGraveyard(Monster):
    """Mammoth Graveyard - 1200 ATK / 800 DEF"""
    def __init__(self):
        super().__init__(40374923, "Mammoth Graveyard", 1200, 800)


class Megazowler(Monster):
    """Megazowler - 1800 ATK / 2000 DEF"""
    def __init__(self):
        super().__init__(75390004, "Megazowler", 1800, 2000)


class MadSwordBeast(Monster):
    """Mad Sword Beast - 1400 ATK / 1200 DEF (Piercing)"""
    def __init__(self):
        super().__init__(79870141, "Mad Sword Beast", 1400, 1200)
        self.piercing = True


class Kuriboh(Monster):
    """Kuriboh - 300 ATK / 200 DEF"""
    def __init__(self):
        super().__init__(40640057, "Kuriboh", 300, 200)


class MystikWok(Spell):
    """Mystik Wok - Tribute 1 monster; gain LP equal to ATK or DEF"""
    def __init__(self):
        super().__init__(80161395, "Mystik Wok")
    
    def activate(self, game_state, tribute_monster, use_def=True):
        for i, monster in enumerate(game_state.player.monster_zones):
            if monster and monster.card_id == tribute_monster.card_id:
                lp_gain = tribute_monster.defense if use_def else tribute_monster.atk
                game_state.player.lp += lp_gain
                game_state.player.monster_zones[i] = None
                return True
        return False


class Confiscation(Spell):
    """Confiscation - Pay 1000 LP; discard 1 card from opponent's hand"""
    def __init__(self):
        super().__init__(17375316, "Confiscation")
    
    def activate(self, game_state, target_card_index=0):
        if game_state.player.lp <= 1000:
            return False
        game_state.player.lp -= 1000
        if target_card_index < len(game_state.opponent.hand):
            game_state.opponent.hand.pop(target_card_index)
            return True
        return False


class UltimateOffering(Trap):
    """Ultimate Offering - Pay 500 LP; Normal Summon 1 monster"""
    def __init__(self):
        super().__init__(511003023, "Ultimate Offering")
    
    def activate(self, game_state, summon_monster_index, use_big_pill=False):
        if game_state.player.lp <= 500:
            return False
        game_state.player.lp -= 500
        
        if summon_monster_index >= len(game_state.player.hand):
            return False
        
        monster = game_state.player.hand[summon_monster_index]
        if not isinstance(monster, Monster):
            return False
        
        for i in range(5):
            if game_state.player.monster_zones[i] is None:
                monster.position = Position.FACEUP_ATTACK
                game_state.player.monster_zones[i] = monster
                game_state.player.hand.pop(summon_monster_index)
                return True
        return False


def create_card(card_id: int) -> Card:
    """Create card instance from card ID"""
    card_map = {
        79409334: BlackStego,
        84808313: UltimateTyranno,
        12482652: OjamaToken,
        2851070: ReflectBounder,
        83133491: OjamaTrio,
        29843091: ZeroGravity,
        511003009: BigEvolutionPill,
        40374923: MammothGraveyard,
        75390004: Megazowler,
        79870141: MadSwordBeast,
        40640057: Kuriboh,
        80161395: MystikWok,
        17375316: Confiscation,
        511003023: UltimateOffering,
    }
    
    card_class = card_map.get(card_id)
    if card_class:
        return card_class()
    else:
        raise ValueError(f"Unknown card ID: {card_id}")
