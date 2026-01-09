#!/usr/bin/env python3
"""
저장된 모델로 학습된 에이전트의 플레이를 시청하는 스크립트
"""

import torch
import argparse
import sys
from pathlib import Path

from env_wrapper import make_env
from env_oh_jama import OhJamaEnv
from env_ancient_kings import AncientKingsEnv
from policy_gradient_agent import REINFORCEAgent, A2CAgent
from actions import ActionSpace, ActionType


def get_action_name(action_obj, env):
    """액션을 읽기 쉬운 이름으로 변환"""
    game = env.simulator.game_state
    
    if action_obj.action_type == ActionType.ACTIVATE_TRAP:
        zone = action_obj.zone_index
        card = game.player.spell_trap_zones[zone]
        return f'ACTIVATE_TRAP: {card.name if card else "Unknown"} (zone {zone})'
    
    elif action_obj.action_type == ActionType.ACTIVATE_SPELL:
        idx = action_obj.card_index
        card = game.player.hand[idx] if idx < len(game.player.hand) else None
        return f'ACTIVATE_SPELL: {card.name if card else "Unknown"} (hand {idx})'
    
    elif action_obj.action_type == ActionType.ATTACK:
        attacker_zone = action_obj.card_index
        target_zone = action_obj.target_index
        attacker = game.player.monster_zones[attacker_zone]
        if target_zone < 5:
            target = game.opponent.monster_zones[target_zone]
            target_name = target.name if target else 'Empty'
        else:
            target_name = 'Direct'
        return f'ATTACK: {attacker.name if attacker else "Unknown"} → {target_name}'
    
    elif action_obj.action_type == ActionType.CHANGE_PHASE:
        return f'CHANGE_PHASE: {game.phase} → next'
    
    elif action_obj.action_type == ActionType.END_TURN:
        return 'END_TURN'
    
    return str(action_obj)


def play_episode(env, agent, verbose=True):
    """에이전트가 한 에피소드를 플레이"""
    state, info = env.reset()
    
    if verbose:
        print()
        print(' Initial State:')
        print(f'   Player LP: {env.simulator.game_state.player.lp}')
        print(f'   Opponent LP: {env.simulator.game_state.opponent.lp}')
        print()
        print('   Player Hand:', [c.name for c in env.simulator.game_state.player.hand])
        print('   Player Monsters:', 
              [(m.name, m.atk) if m else None for m in env.simulator.game_state.player.monster_zones])
        print('   Player S/T:', 
              [c.name if c else None for c in env.simulator.game_state.player.spell_trap_zones])
        print()
        print('   Opponent Monsters:', 
              [(m.name, m.atk, m.position.value) if m else None 
               for m in env.simulator.game_state.opponent.monster_zones])
        print()
    
    total_reward = 0
    step = 1
    trajectory = []
    
    while step <= 30:
        # 에이전트가 액션 선택
        valid_mask = env.get_valid_actions()
        action = agent.select_action(state, valid_mask, training=False)
        
        # 액션 이름 가져오기
        action_obj = ActionSpace.index_to_action(action)
        action_name = get_action_name(action_obj, env)
        
        trajectory.append((step, action_name))
        
        if verbose:
            print(f'🔹 Step {step}: {action_name}')
            print('-' * 50)
        
        # 액션 실행
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if verbose:
            print(f'   Reward: {reward:+.2f} (Total: {total_reward:.2f})')
        
        if terminated or truncated:
            if verbose:
                print()
                print('=' * 70)
                if info.get('result') == 'win':
                    print(' VICTORY!')
                elif info.get('result') == 'lose':
                    print(' DEFEAT')
                else:
                    print(f' {info.get("result", "timeout").upper()}')
                print('=' * 70)
            break
        
        state = next_state
        step += 1
        if verbose:
            print()
    
    return {
        'result': info.get('result', 'timeout'),
        'total_reward': total_reward,
        'steps': step,
        'trajectory': trajectory,
        'player_lp': env.simulator.game_state.player.lp,
        'opponent_lp': env.simulator.game_state.opponent.lp
    }


def main():
    parser = argparse.ArgumentParser(description='Watch trained agent play Yu-Gi-Oh puzzle')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model (.pth)')
    parser.add_argument('--puzzle', type=int, default=0, help='Puzzle ID (0=Oh Jama, 1=Ancient Kings)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('--algorithm', type=str, default='reinforce_baseline',
                        choices=['reinforce_baseline', 'a2c'], help='Algorithm used for training')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    puzzle_names = {0: 'Oh Jama', 1: 'Ancient Kings'}
    
    print('=' * 70)
    print(f' TRAINED AGENT PLAYING: {puzzle_names.get(args.puzzle, "Unknown")}')
    print('=' * 70)
    
    # 환경 생성
    env = make_env(puzzle_id=args.puzzle)
    env.verbose = not args.quiet
    
    state_size = env.observation_size
    action_size = env.action_size
    
    # 에이전트 생성
    if args.algorithm == 'reinforce_baseline':
        agent = REINFORCEAgent(
            state_size, action_size,
            hidden_size=256,
            lr=0.001,
            gamma=0.99,
            use_baseline=True
        )
    else:  # a2c
        agent = A2CAgent(
            state_size, action_size,
            hidden_size=256,
            lr=0.001,
            gamma=0.99
        )
    
    # 모델 로드
    print(f' Loading model: {args.model}')
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    # 체크포인트 형식 확인 및 로드
    print(f'   Checkpoint keys: {list(checkpoint.keys())}')
    
    if 'policy_state_dict' in checkpoint:
        # 형식 1: {'policy_state_dict': ..., 'baseline_state_dict': ...}
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    elif 'policy_net' in checkpoint:
        # 형식 2: REINFORCE {'policy_net': ..., 'policy_optimizer': ...}
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        if hasattr(agent, 'value_net') and 'value_net' in checkpoint:
            agent.value_net.load_state_dict(checkpoint['value_net'])
    elif 'ac_net' in checkpoint:
        # 형식 3: A2C {'ac_net': ..., 'optimizer': ...}
        if hasattr(agent, 'ac_net'):
            agent.ac_net.load_state_dict(checkpoint['ac_net'])
        else:
            print(' Model is A2C format but agent is REINFORCE. Use --algorithm a2c')
            sys.exit(1)
    elif 'model_state_dict' in checkpoint:
        # 형식 4: {'model_state_dict': ...}
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 형식 5: state_dict 직접 저장
        first_key = list(checkpoint.keys())[0]
        if '.' in first_key:  # 레이어 이름 형식 (fc1.weight 등)
            agent.policy_net.load_state_dict(checkpoint)
        else:
            print(f' Unknown checkpoint format. Keys: {list(checkpoint.keys())}')
            sys.exit(1)
    print(' Model loaded successfully!')
    
    # 에피소드 실행
    wins = 0
    total_rewards = []
    
    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f'\n{"="*70}')
            print(f' Episode {ep + 1}/{args.episodes}')
            print('=' * 70)
        
        result = play_episode(env, agent, verbose=not args.quiet)
        
        if result['result'] == 'win':
            wins += 1
        total_rewards.append(result['total_reward'])
        
        if not args.quiet:
            print()
            print(' Episode Summary:')
            print(f'   Result: {result["result"]}')
            print(f'   Total Steps: {result["steps"]}')
            print(f'   Total Reward: {result["total_reward"]:.2f}')
            print(f'   Player LP: {result["player_lp"]}')
            print(f'   Opponent LP: {result["opponent_lp"]}')
    
    # 전체 통계
    if args.episodes > 1:
        print()
        print('=' * 70)
        print(' OVERALL STATISTICS')
        print('=' * 70)
        print(f'   Win Rate: {wins}/{args.episodes} ({wins/args.episodes*100:.1f}%)')
        print(f'   Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}')


if __name__ == '__main__':
    main()
