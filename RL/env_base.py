"""
Yu-Gi-Oh! 퍼즐 환경 기본 클래스
모든 퍼즐 환경의 공통 기능 정의
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
import numpy as np

from state import GameState
from simul import YuGiOhSimulator
from actions import ActionSpace, ActionType


class YuGiOhPuzzleEnvBase(ABC):
    """
    Yu-Gi-Oh! 퍼즐 환경의 추상 기본 클래스
    
    모든 퍼즐 환경은 이 클래스를 상속받아 구현합니다.
    """
    
    def __init__(self, max_steps: int = 30, reward_shaping: bool = True):
        """
        Args:
            max_steps: 최대 스텝 수
            reward_shaping: Dense reward 사용 여부
        """
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        
        # 공간 정의
        self.observation_size = YuGiOhSimulator.get_state_size()  # 56
        self.action_size = ActionSpace.get_action_space_size()    # 42
        
        # 환경 상태
        self.simulator: Optional[YuGiOhSimulator] = None
        self.current_step = 0
        self.prev_opponent_lp = 0
        self.verbose = False
        
    @property
    @abstractmethod
    def puzzle_name(self) -> str:
        """퍼즐 이름 반환"""
        pass
    
    @property
    @abstractmethod
    def initial_opponent_lp(self) -> int:
        """초기 상대 LP 반환"""
        pass
    
    @abstractmethod
    def create_puzzle(self) -> GameState:
        """퍼즐 초기 상태 생성"""
        pass
    
    @abstractmethod
    def calculate_shaped_reward(self, action_obj, base_reward: float) -> float:
        """퍼즐별 맞춤 보상 계산"""
        pass
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        환경 리셋
        
        Returns:
            observation: 초기 상태 벡터
            info: 추가 정보
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 퍼즐 생성
        game_state = self.create_puzzle()
        self.simulator = YuGiOhSimulator(game_state)
        self.simulator.verbose = self.verbose
        self.current_step = 0
        self.prev_opponent_lp = self.simulator.game_state.opponent.lp
        
        # 퍼즐별 상태 초기화
        self._reset_puzzle_state()
        
        observation = np.array(self.simulator.get_state_vector(), dtype=np.float32)
        info = {
            'puzzle_name': self.puzzle_name,
            'player_lp': self.simulator.game_state.player.lp,
            'opponent_lp': self.simulator.game_state.opponent.lp
        }
        
        return observation, info
    
    def _reset_puzzle_state(self):
        """퍼즐별 상태 초기화 (서브클래스에서 오버라이드)"""
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        액션 실행
        
        Args:
            action: 액션 인덱스 (0-41)
            
        Returns:
            observation: 다음 상태
            reward: 보상
            terminated: 게임 종료 여부
            truncated: 최대 스텝 초과 여부
            info: 추가 정보
        """
        self.current_step += 1
        
        # 액션 변환
        action_obj = ActionSpace.index_to_action(action)
        
        # 시뮬레이터 실행
        _, base_reward, done, info = self.simulator.step(action_obj)
        
        # Reward Shaping
        if self.reward_shaping:
            reward = self.calculate_shaped_reward(action_obj, base_reward)
        else:
            reward = base_reward
        
        # 이전 상대 LP 업데이트
        self.prev_opponent_lp = self.simulator.game_state.opponent.lp
        
        # 종료 조건 확인
        terminated = done
        truncated = self.current_step >= self.max_steps and not done
        
        if truncated:
            reward = -20.0  # 시간 초과 페널티
            info['result'] = 'timeout'
        
        # 다음 상태
        observation = np.array(self.simulator.get_state_vector(), dtype=np.float32)
        
        return observation, reward, terminated, truncated, info
    
    def get_valid_actions(self) -> np.ndarray:
        """
        현재 상태에서 유효한 액션 마스크 반환
        
        Returns:
            valid_mask: 유효한 액션은 True, 무효한 액션은 False
        """
        valid_actions = ActionSpace.get_valid_actions(self.simulator.game_state)
        valid_mask = np.zeros(self.action_size, dtype=bool)
        
        for action in valid_actions:
            idx = ActionSpace.action_to_index(action)
            valid_mask[idx] = True
        
        return valid_mask
    
    def calculate_base_shaped_reward(self, base_reward: float) -> float:
        """
        공통 기본 보상 계산 (LP 데미지 기반)
        
        Args:
            base_reward: 시뮬레이터에서 받은 기본 보상
            
        Returns:
            shaped_reward: 기본 shaped 보상
        """
        current_opponent_lp = self.simulator.game_state.opponent.lp
        lp_damage = self.prev_opponent_lp - current_opponent_lp
        
        if lp_damage > 0:
            # LP 데미지에 큰 보상
            shaped_reward = (lp_damage / self.initial_opponent_lp) * 100.0
        elif base_reward > 0:
            # 카드 효과 발동 등 진행 보상
            shaped_reward = base_reward * 0.5
        elif base_reward < 0:
            # 무효 액션 패널티
            shaped_reward = base_reward * 0.5
        else:
            # 중립 행동
            shaped_reward = -0.1
        
        return shaped_reward
    
    def render(self):
        """현재 게임 상태 출력"""
        if self.simulator:
            game = self.simulator.game_state
            print(f"\n{'='*50}")
            print(f"Step: {self.current_step}")
            print(f"Player LP: {game.player.lp}")
            print(f"Opponent LP: {game.opponent.lp}")
            print(f"Phase: {game.phase}")
            print(f"{'='*50}")
