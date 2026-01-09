"""
Yu-Gi-Oh Puzzle Duel - Environment Wrapper (Router)

퍼즐별로 분리된 환경을 라우팅합니다.
하위 호환성을 위해 기존 인터페이스 유지.
"""

from typing import Optional
from env_oh_jama import OhJamaEnv
from env_ancient_kings import AncientKingsEnv


def make_env(puzzle_id: int = 0, max_steps: int = 30, reward_shaping: bool = True):
    """
    퍼즐 ID에 따라 적절한 환경 반환
    
    Args:
        puzzle_id: 0 = Oh Jama, 1 = Ancient Kings
        max_steps: 최대 스텝 수
        reward_shaping: Dense reward 사용 여부
        
    Returns:
        퍼즐 환경 인스턴스
    """
    if puzzle_id == 0:
        return OhJamaEnv(max_steps=max_steps, reward_shaping=reward_shaping)
    elif puzzle_id == 1:
        return AncientKingsEnv(max_steps=max_steps, reward_shaping=reward_shaping)
    else:
        raise ValueError(f"Unknown puzzle_id: {puzzle_id}. Available: 0 (Oh Jama), 1 (Ancient Kings)")


# 하위 호환성을 위한 클래스 (make_env 사용 권장)
class YuGiOhPuzzleEnv:
    """
    하위 호환성을 위한 래퍼 클래스
    
    권장: make_env(puzzle_id) 또는 OhJamaEnv/AncientKingsEnv 직접 사용
    """
    
    def __new__(cls, puzzle_id: int = 0, max_steps: int = 30, reward_shaping: bool = True):
        """팩토리 패턴으로 적절한 환경 반환"""
        return make_env(puzzle_id, max_steps, reward_shaping)


# 퍼즐 정보 유틸리티
PUZZLE_INFO = {
    0: {
        'name': 'Oh Jama',
        'player_lp': 100,
        'opponent_lp': 9900,
        'description': 'Ojama Trio → Zero Gravity → Big Evolution Pill 순서로 1턴킬',
        'env_class': OhJamaEnv
    },
    1: {
        'name': 'Ancient Kings',
        'player_lp': 100,
        'opponent_lp': 4200,
        'description': 'LP 관리와 Kuriboh 제거로 1턴킬',
        'env_class': AncientKingsEnv
    }
}


def get_puzzle_names():
    """사용 가능한 퍼즐 이름 반환"""
    return {pid: info['name'] for pid, info in PUZZLE_INFO.items()}


def get_puzzle_info(puzzle_id: int):
    """퍼즐 정보 반환"""
    if puzzle_id not in PUZZLE_INFO:
        raise ValueError(f"Unknown puzzle_id: {puzzle_id}")
    return PUZZLE_INFO[puzzle_id]
