"""
Yu-Gi-Oh Puzzle Duel Simulator - Game State Management
"""

from typing import List, Optional
from card import Card, Monster, Position, create_card
import copy

class Player:
    def __init__(self, lp: int):
        self.lp = lp
        self.hand: List[Optional[Card]] = []
        self.monster_zones: List[Optional[Monster]] = [None] * 5
        self.spell_trap_zones: List[Optional[Card]] = [None] * 5
        self.graveyard: List[Card] = []
        
    def __repr__(self):
        return f"Player(LP: {self.lp}, Hand: {len(self.hand)}, Monsters: {sum(1 for m in self.monster_zones if m)})"

class GameState:
    def __init__(self, player_lp: int, opponent_lp: int):
        self.player = Player(player_lp)
        self.opponent = Player(opponent_lp)
        self.phase = "main1"  # main1, battle, main2, end
        self.turn = 1
        
    def clone(self):
        """Deep copy of game state"""
        return copy.deepcopy(self)
    
    def is_terminal(self) -> bool:
        """Check if game is over"""
        return self.player.lp <= 0 or self.opponent.lp <= 0
    
    def get_winner(self) -> Optional[int]:
        """Return winner: 0 (player), 1 (opponent), None (game ongoing)"""
        if self.opponent.lp <= 0:
            return 0
        elif self.player.lp <= 0:
            return 1
        return None
    
    def count_opponent_monsters(self) -> int:
        """Count opponent's monsters"""
        return sum(1 for m in self.opponent.monster_zones if m is not None)
    
    def get_player_monsters(self) -> List[Monster]:
        """Get all player's monsters"""
        return [m for m in self.player.monster_zones if m is not None]
    
    def get_opponent_monsters(self) -> List[Monster]:
        """Get all opponent's monsters"""
        return [m for m in self.opponent.monster_zones if m is not None]
    
    def __repr__(self):
        return f"GameState(Turn: {self.turn}, Phase: {self.phase}, Player: {self.player}, Opponent: {self.opponent})"

def create_oh_jama_puzzle() -> GameState:
    """Create the Oh Jama puzzle initial state"""
    game = GameState(player_lp=100, opponent_lp=9900)
    
    # Player's hand
    game.player.hand.append(create_card(79409334))  # Black Stego
    game.player.hand.append(create_card(511003009))  # Big Evolution Pill
    game.player.hand.append(create_card(84808313))  # Ultimate Tyranno
    
    # Player's monster zone
    black_stego_field = create_card(79409334)
    black_stego_field.position = Position.FACEUP_ATTACK
    game.player.monster_zones[2] = black_stego_field
    
    # Player's spell/trap zones (face-down)
    game.player.spell_trap_zones[2] = create_card(83133491)  # Ojama Trio
    game.player.spell_trap_zones[1] = create_card(29843091)  # Zero Gravity
    
    # Opponent's monster zones
    token1 = create_card(2851070)
    token1.position = Position.FACEUP_ATTACK
    game.opponent.monster_zones[2] = token1
    
    token2 = create_card(2851070)
    token2.position = Position.FACEUP_ATTACK
    game.opponent.monster_zones[3] = token2
    
    return game

def print_game_state(game: GameState):
    """Print game state in readable format"""
    print("=" * 60)
    print(f"Turn {game.turn} - Phase: {game.phase}")
    print(f"Player LP: {game.player.lp} | Opponent LP: {game.opponent.lp}")
    print("-" * 60)
    
    print("OPPONENT FIELD:")
    print("  Hand:", [str(c) for c in game.opponent.hand])
    print("  Monsters:", [str(m) if m else "Empty" for m in game.opponent.monster_zones])
    
    print("\nPLAYER FIELD:")
    print("  Hand:", [str(c) for c in game.player.hand])
    print("  Monsters:", [f"{m} ({m.position.value})" if m else "Empty" for m in game.player.monster_zones])
    print("  S/T Zones:", [str(c) if c else "Empty" for c in game.player.spell_trap_zones])
    print("=" * 60)

def create_ancient_kings_puzzle() -> GameState:
    """Create the Ancient Kings puzzle initial state"""
    game = GameState(player_lp=100, opponent_lp=4200)
    
    # Player's hand (5 cards)
    game.player.hand.append(create_card(75390004))  # Megazowler
    game.player.hand.append(create_card(79870141))  # Mad Sword Beast
    game.player.hand.append(create_card(511003009))  # Big Evolution Pill
    game.player.hand.append(create_card(17375316))  # Confiscation
    game.player.hand.append(create_card(84808313))  # Ultimate Tyranno
    
    # Player's monster zone
    mammoth = create_card(40374923)  # Mammoth Graveyard
    mammoth.position = Position.FACEUP_ATTACK
    game.player.monster_zones[2] = mammoth
    
    # Player's spell/trap zones
    game.player.spell_trap_zones[2] = create_card(80161395)  # Mystik Wok (face-down)
    game.player.spell_trap_zones[3] = create_card(511003023)  # Ultimate Offering (face-up)
    
    # Opponent's hand
    game.opponent.hand.append(create_card(40640057))  # Kuriboh
    
    return game
