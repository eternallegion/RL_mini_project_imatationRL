import os
import random
from typing import List, Dict, Any, Callable

import numpy as np
import torch
import gym  # 경고만 뜨고, 실제로는 사용 안 해도 상관 없음

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from env_multi import MultiPuzzleEnv
from loader import load_all_puzzles


# ===== 공통 SEED 설정 =====
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
try:
    torch.manual_seed(SEED)
except Exception:
    pass


# ===== 액션 마스크 함수 =====
def mask_fn(env: MultiPuzzleEnv):
    """
    sb3-contrib의 ActionMasker에서 사용하는 마스크 함수.
    env.action_mask()가 shape (max_hand+1,) 의 np.ndarray를 반환해야 함.
    """
    return env.action_mask().astype(bool)


def make_env(puzzle_def: Dict[str, Any]) -> Callable:
    """
    DummyVecEnv용 환경 생성 함수(thunk)
    """
    def _thunk():
        env = MultiPuzzleEnv(puzzle_def)
        env = ActionMasker(env, mask_fn)
        return env

    return _thunk


def main():
    # 1) 퍼즐 로드
    all_puzzles: List[Dict[str, Any]] = load_all_puzzles()
    if not all_puzzles:
        raise RuntimeError("퍼즐 데이터를 하나도 로드하지 못했습니다.")

    # 2) 학습에 사용할 퍼즐만 선택
    #    → polish_q 결과 기준으로 '실제로 킬 가능한 퍼즐'만 사용
    KILLABLE_NAMES = {
        "seq_block_buff_kill",
        "[GX_Spirit_Caller]B03_Oh_Jama",
        "no_heal_cap_atk2",
        # zero_then_atk, damage_chain_min, zero_only_win 은
        # 현재 설정상 승리 불가능(per polish_q). 나중에 JSON을 손보면 추가 가능.
    }

    puzzles = [p for p in all_puzzles if p.get("name") in KILLABLE_NAMES]
    if not puzzles:
        raise RuntimeError("학습에 사용할 퍼즐이 없습니다. KILLABLE_NAMES 설정을 확인하세요.")

    print("[INFO] 학습에 사용할 퍼즐 목록:")
    for p in puzzles:
        print("  -", p["name"])

    # 3) 벡터라이즈드 환경 구성
    env_fns = [make_env(p) for p in puzzles]
    env = DummyVecEnv(env_fns)

    # 4) PPO 마스크 모델 생성
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device, "device")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log="logs/maskable_ppo",
        seed=SEED,
    )

    # 5) 학습
    total_timesteps = 200_000  # 필요에 따라 조정 가능
    print(f"[INFO] 학습 시작: total_timesteps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps)

    # 6) 체크포인트 저장
    os.makedirs("ckpt", exist_ok=True)
    ckpt_path = os.path.join("ckpt", "maskable_ppo")
    model.save(ckpt_path)
    print(f"[INFO] 모델 저장 완료: {ckpt_path}")


if __name__ == "__main__":
    main()
