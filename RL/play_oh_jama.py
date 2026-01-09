#!/usr/bin/env python3
"""
Oh Jama 퍼즐 전용 - 학습된 에이전트 플레이 시각화

특수 기능:
- 3단계 시퀀스 진행 상황 표시 (Ojama Trio → Zero Gravity → Big Evolution Pill)
- Token 공격표시/수비표시 상태 표시
- Reflect Bounder 수비 전환 표시
- 전투 데미지 계산 표시
"""

import torch
import argparse
import sys
from pathlib import Path

from env_wrapper import make_env
from env_oh_jama import OhJamaEnv
from policy_gradient_agent import REINFORCEAgent, A2CAgent
from actions import ActionSpace, ActionType
from card import Position, ReflectBounder, OjamaToken


def get_action_name(action_obj, game_state):
    """액션을 읽기 쉬운 이름으로 변환"""
    
    if action_obj.action_type == ActionType.ACTIVATE_TRAP:
        zone = action_obj.zone_index
        card = game_state.player.spell_trap_zones[zone]
        card_name = card.name if card else "Unknown"
        
        # Oh Jama 트랩 이름 매핑
        trap_names = {1: "Zero Gravity", 2: "Ojama Trio"}
        display_name = trap_names.get(zone, card_name)
        return f'ACTIVATE_TRAP: {display_name}'
    
    elif action_obj.action_type == ActionType.ACTIVATE_SPELL:
        idx = action_obj.card_index
        card = game_state.player.hand[idx] if idx < len(game_state.player.hand) else None
        return f'ACTIVATE_SPELL: {card.name if card else "Unknown"}'
    
    elif action_obj.action_type == ActionType.ATTACK:
        attacker_zone = action_obj.card_index
        target_zone = action_obj.target_index
        attacker = game_state.player.monster_zones[attacker_zone]
        attacker_name = attacker.name if attacker else "Unknown"
        attacker_atk = attacker.atk if attacker else 0
        
        if target_zone < 5:
            target = game_state.opponent.monster_zones[target_zone]
            if target:
                pos = "ATK" if target.position == Position.FACEUP_ATTACK else "DEF"
                stat = target.atk if pos == "ATK" else target.defense
                target_name = f'{target.name} ({pos} {stat})'
            else:
                target_name = 'Empty'
        else:
            target_name = 'Direct Attack'
        return f'ATTACK: {attacker_name} (ATK {attacker_atk}) → {target_name}'
    
    elif action_obj.action_type == ActionType.CHANGE_PHASE:
        return 'CHANGE_PHASE → Battle Phase'
    
    elif action_obj.action_type == ActionType.END_TURN:
        return 'END_TURN'
    
    return str(action_obj)


def get_sequence_status(env):
    """3단계 시퀀스 진행 상황 반환"""
    status = []
    
    # Step 1: Ojama Trio
    if env.ojama_trio_used:
        status.append(" Step 1: Ojama Trio (Token 3개 소환)")
    else:
        status.append(" Step 1: Ojama Trio")
    
    # Step 2: Zero Gravity
    if env.zero_gravity_used:
        if env.correct_sequence_bonus_given:
            status.append(" Step 2: Zero Gravity (올바른 순서!)")
        else:
            status.append(" Step 2: Zero Gravity (순서 틀림)")
    else:
        status.append(" Step 2: Zero Gravity")
    
    # Step 3: Big Evolution Pill
    if env.big_evo_pill_used:
        status.append(" Step 3: Big Evolution Pill → Ultimate Tyranno")
    else:
        status.append(" Step 3: Big Evolution Pill")
    
    return status


def get_field_status(game_state):
    """필드 상태 반환 (Token 및 Reflect Bounder 상태)"""
    tokens_atk = 0
    tokens_def = 0
    bounders_atk = 0
    bounders_def = 0
    
    for m in game_state.opponent.monster_zones:
        if m:
            if isinstance(m, OjamaToken):
                if m.position == Position.FACEUP_ATTACK:
                    tokens_atk += 1
                else:
                    tokens_def += 1
            elif isinstance(m, ReflectBounder):
                if m.position == Position.FACEUP_ATTACK:
                    bounders_atk += 1
                else:
                    bounders_def += 1
    
    player_monsters = []
    for m in game_state.player.monster_zones:
        if m:
            pos = "ATK" if m.position == Position.FACEUP_ATTACK else "DEF"
            player_monsters.append(f"{m.name} ({pos} {m.atk})")
    
    return {
        'tokens_atk': tokens_atk,
        'tokens_def': tokens_def,
        'bounders_atk': bounders_atk,
        'bounders_def': bounders_def,
        'player_monsters': player_monsters
    }


def play_episode(env, agent, verbose=True):
    """에이전트가 한 에피소드를 플레이"""
    state, info = env.reset()
    game = env.simulator.game_state
    
    if verbose:
        print()
        print('┌' + '─' * 68 + '┐')
        print('│' + '  INITIAL STATE'.ljust(68) + '│')
        print('├' + '─' * 68 + '┤')
        print(f'│  Player LP: {game.player.lp} ( 즉사 위험!)'.ljust(69) + '│')
        print(f'│  Opponent LP: {game.opponent.lp}'.ljust(69) + '│')
        print('├' + '─' * 68 + '┤')
        print('│  Player Hand:'.ljust(69) + '│')
        for c in game.player.hand:
            print(f'│    • {c.name}'.ljust(69) + '│')
        print('├' + '─' * 68 + '┤')
        print('│  Player S/T Zones:'.ljust(69) + '│')
        for i, c in enumerate(game.player.spell_trap_zones):
            if c:
                print(f'│    [Zone {i}] {c.name}'.ljust(69) + '│')
        print('├' + '─' * 68 + '┤')
        print('│  Opponent Monsters:'.ljust(69) + '│')
        for m in game.opponent.monster_zones:
            if m:
                pos = "ATK" if m.position == Position.FACEUP_ATTACK else "DEF"
                print(f'│    • {m.name} ({pos} {m.atk})  반사 데미지!'.ljust(69) + '│')
        print('└' + '─' * 68 + '┘')
        print()
        
        print('┌' + '─' * 68 + '┐')
        print('│' + '  CORRECT SEQUENCE'.ljust(68) + '│')
        print('├' + '─' * 68 + '┤')
        print('│  1. Ojama Trio → Token 3개 (수비표시)'.ljust(69) + '│')
        print('│  2. Zero Gravity → Token 공격표시, Bounder 수비표시'.ljust(69) + '│')
        print('│  3. Big Evolution Pill → Ultimate Tyranno (ATK 3000)'.ljust(69) + '│')
        print('│  4. Battle Phase → Token 공격 (3000 + 300 = 3300 × 3)'.ljust(69) + '│')
        print('└' + '─' * 68 + '┘')
        print()
        
        print('  WARNING: Reflect Bounder 공격표시에서 공격하면 3000 반사 → 즉사!')
        print()
    
    total_reward = 0
    step = 1
    trajectory = []
    prev_opp_lp = game.opponent.lp
    
    while step <= 30:
        # 에이전트가 액션 선택
        valid_mask = env.get_valid_actions()
        action = agent.select_action(state, valid_mask, training=False)
        
        # 액션 이름 가져오기
        action_obj = ActionSpace.index_to_action(action)
        action_name = get_action_name(action_obj, env.simulator.game_state)
        
        trajectory.append((step, action_name))
        
        if verbose:
            print(f' Step {step}: {action_name}')
            print('─' * 70)
        
        # 액션 실행
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 데미지 계산
        current_opp_lp = env.simulator.game_state.opponent.lp
        damage = prev_opp_lp - current_opp_lp
        
        if verbose:
            print(f'   Reward: {reward:+.2f} (Total: {total_reward:.2f})')
            
            # 데미지 표시
            if damage > 0:
                print(f'    Damage: {damage} (Opponent LP: {prev_opp_lp} → {current_opp_lp})')
            
            # 필드 상태 표시
            field = get_field_status(env.simulator.game_state)
            if field['tokens_atk'] > 0 or field['tokens_def'] > 0:
                print(f'   Tokens: {field["tokens_atk"]} ATK, {field["tokens_def"]} DEF')
            if field['bounders_atk'] > 0 or field['bounders_def'] > 0:
                bounder_status = " 위험!" if field['bounders_atk'] > 0 else " 안전"
                print(f'   Bounders: {field["bounders_atk"]} ATK, {field["bounders_def"]} DEF ({bounder_status})')
            if field['player_monsters']:
                print(f'   Your Monsters: {", ".join(field["player_monsters"])}')
        
        prev_opp_lp = current_opp_lp
        
        if terminated or truncated:
            if verbose:
                print()
                print('═' * 70)
                if info.get('result') == 'win':
                    print(' VICTORY!')
                elif info.get('result') == 'lose':
                    print(' DEFEAT (Reflect Bounder 반사 데미지?)')
                else:
                    print(f' {info.get("result", "timeout").upper()}')
                print('═' * 70)
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
    parser = argparse.ArgumentParser(description='Watch trained agent play Oh Jama puzzle')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model (.pth)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('--algorithm', type=str, default='reinforce_baseline',
                        choices=['reinforce_baseline', 'a2c'], help='Algorithm used for training')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    print('═' * 70)
    print(' OH JAMA PUZZLE - TRAINED AGENT PLAYBACK')
    print('═' * 70)
    
    # 환경 생성 (Oh Jama = puzzle_id 0)
    env = OhJamaEnv()
    
    state_size = env.observation_size
    action_size = env.action_size
    
    # 모델 로드
    print(f' Loading model: {args.model}')
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    print(f'   Checkpoint keys: {list(checkpoint.keys())}')
    
    # 알고리즘 자동 감지
    if 'ac_net' in checkpoint:
        detected_algo = 'a2c'
    elif 'policy_net' in checkpoint:
        detected_algo = 'reinforce_baseline'
    else:
        detected_algo = args.algorithm
    
    if detected_algo != args.algorithm:
        print(f'   Auto-detected algorithm: {detected_algo} (ignoring --algorithm {args.algorithm})')
    
    # 에이전트 생성 (자동 감지된 알고리즘 사용)
    if detected_algo == 'a2c':
        agent = A2CAgent(
            state_size, action_size,
            hidden_size=256,
            lr=0.001,
            gamma=0.99
        )
        agent.ac_net.load_state_dict(checkpoint['ac_net'])
    else:  # reinforce_baseline
        agent = REINFORCEAgent(
            state_size, action_size,
            hidden_size=256,
            lr=0.001,
            gamma=0.99,
            use_baseline=True
        )
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        if hasattr(agent, 'value_net') and 'value_net' in checkpoint:
            agent.value_net.load_state_dict(checkpoint['value_net'])
    
    print(' Model loaded successfully!')
    
    # 에피소드 실행
    wins = 0
    total_rewards = []
    
    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f'\n{"═" * 70}')
            print(f' Episode {ep + 1}/{args.episodes}')
            print('═' * 70)
        
        result = play_episode(env, agent, verbose=not args.quiet)
        
        if result['result'] == 'win':
            wins += 1
        total_rewards.append(result['total_reward'])
        
        if not args.quiet:
            print()
            print('┌' + '─' * 68 + '┐')
            print('│' + '  EPISODE SUMMARY'.ljust(68) + '│')
            print('├' + '─' * 68 + '┤')
            print(f'│  Result: {result["result"]}'.ljust(69) + '│')
            print(f'│  Total Steps: {result["steps"]}'.ljust(69) + '│')
            print(f'│  Total Reward: {result["total_reward"]:.2f}'.ljust(69) + '│')
            print(f'│  Final Player LP: {result["player_lp"]}'.ljust(69) + '│')
            print(f'│  Final Opponent LP: {result["opponent_lp"]}'.ljust(69) + '│')
            print('└' + '─' * 68 + '┘')
            
            # 시퀀스 상태 출력
            print()
            print('┌' + '─' * 68 + '┐')
            print('│' + '  SEQUENCE PROGRESS'.ljust(68) + '│')
            print('├' + '─' * 68 + '┤')
            for status in get_sequence_status(env):
                print(f'│  {status}'.ljust(69) + '│')
            print('└' + '─' * 68 + '┘')
    
    # 전체 통계
    if args.episodes > 1:
        print()
        print('═' * 70)
        print(' OVERALL STATISTICS')
        print('═' * 70)
        print(f'   Win Rate: {wins}/{args.episodes} ({wins/args.episodes*100:.1f}%)')
        print(f'   Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}')


if __name__ == '__main__':
    main()
