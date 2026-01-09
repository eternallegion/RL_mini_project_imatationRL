#!/usr/bin/env python3
"""
Ancient Kings 퍼즐 전용 - 학습된 에이전트 플레이 시각화

특수 기능:
- LP 변화 추적 (100 → 2100 → 1100 → 600 → 100)
- 5단계 시퀀스 진행 상황 표시
- Kuriboh 제거 상태 표시
- 필드 몬스터 ATK 합계 표시
"""

import torch
import argparse
import sys
from pathlib import Path

from env_wrapper import make_env
from env_ancient_kings import AncientKingsEnv
from policy_gradient_agent import REINFORCEAgent, A2CAgent
from actions import ActionSpace, ActionType
from card import UltimateTyranno, Megazowler, MadSwordBeast


def get_action_name(action_obj, game_state):
    """액션을 읽기 쉬운 이름으로 변환"""
    
    if action_obj.action_type == ActionType.ACTIVATE_TRAP:
        zone = action_obj.zone_index
        card = game_state.player.spell_trap_zones[zone]
        card_name = card.name if card else "Unknown"
        
        # Ancient Kings 트랩 이름 매핑
        trap_names = {2: "Mystik Wok", 3: "Ultimate Offering"}
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
            target_name = target.name if target else 'Empty'
        else:
            target_name = 'Direct Attack'
        return f'ATTACK: {attacker_name} (ATK {attacker_atk}) → {target_name}'
    
    elif action_obj.action_type == ActionType.CHANGE_PHASE:
        return 'CHANGE_PHASE → Battle Phase'
    
    elif action_obj.action_type == ActionType.END_TURN:
        return 'END_TURN'
    
    return str(action_obj)


def get_sequence_status(env):
    """5단계 시퀀스 진행 상황 반환"""
    status = []
    
    # Step 1: Big Evolution Pill
    if env.step1_big_evo_pill:
        status.append(" Step 1: Big Evolution Pill")
    else:
        status.append(" Step 1: Big Evolution Pill")
    
    # Step 2: Mystik Wok
    if env.step2_mystik_wok:
        status.append(" Step 2: Mystik Wok")
    else:
        status.append(" Step 2: Mystik Wok")
    
    # Step 3: Confiscation
    if env.step3_confiscation:
        status.append(" Step 3: Confiscation")
    else:
        status.append(" Step 3: Confiscation")
    
    # Step 4: Ultimate Offering (Mad Sword Beast)
    if env.step4_ultimate_offering_1:
        status.append(" Step 4: Ultimate Offering → Mad Sword Beast")
    else:
        status.append(" Step 4: Ultimate Offering → Mad Sword Beast")
    
    # Step 5: Ultimate Offering (Ultimate Tyranno)
    if env.step5_ultimate_offering_2:
        status.append(" Step 5: Ultimate Offering → Ultimate Tyranno")
    else:
        status.append(" Step 5: Ultimate Offering → Ultimate Tyranno")
    
    return status


def get_field_status(game_state):
    """필드 상태 반환"""
    monsters = []
    total_atk = 0
    
    for m in game_state.player.monster_zones:
        if m:
            monsters.append(f"{m.name} (ATK {m.atk})")
            total_atk += m.atk
    
    kuriboh_status = " Kuriboh in hand" if game_state.opponent.hand else " Kuriboh removed"
    
    return monsters, total_atk, kuriboh_status


def play_episode(env, agent, verbose=True):
    """에이전트가 한 에피소드를 플레이"""
    state, info = env.reset()
    game = env.simulator.game_state
    
    if verbose:
        print()
        print('┌' + '─' * 68 + '┐')
        print('│' + '  INITIAL STATE'.ljust(68) + '│')
        print('├' + '─' * 68 + '┤')
        print(f'│  Player LP: {game.player.lp}'.ljust(69) + '│')
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
        print('│  Opponent Hand:'.ljust(69) + '│')
        for c in game.opponent.hand:
            print(f'│    • {c.name}  (blocks direct attack!)'.ljust(69) + '│')
        print('└' + '─' * 68 + '┘')
        print()
        
        print('┌' + '─' * 68 + '┐')
        print('│' + '  CORRECT SEQUENCE'.ljust(68) + '│')
        print('├' + '─' * 68 + '┤')
        print('│  1. Big Evolution Pill → Megazowler'.ljust(69) + '│')
        print('│  2. Mystik Wok → LP 100→2100'.ljust(69) + '│')
        print('│  3. Confiscation → Remove Kuriboh (LP 2100→1100)'.ljust(69) + '│')
        print('│  4. Ultimate Offering → Mad Sword Beast (LP 1100→600)'.ljust(69) + '│')
        print('│  5. Ultimate Offering → Ultimate Tyranno (LP 600→100)'.ljust(69) + '│')
        print('│  6. Battle Phase → Direct Attack (1400 + 3000 = 4400)'.ljust(69) + '│')
        print('└' + '─' * 68 + '┘')
        print()
    
    total_reward = 0
    step = 1
    trajectory = []
    prev_lp = game.player.lp
    
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
        
        # LP 변화 계산
        current_lp = env.simulator.game_state.player.lp
        lp_change = current_lp - prev_lp
        
        if verbose:
            print(f'   Reward: {reward:+.2f} (Total: {total_reward:.2f})')
            
            # LP 변화 표시
            if lp_change != 0:
                direction = "↑" if lp_change > 0 else "↓"
                print(f'   LP: {prev_lp} → {current_lp} ({direction}{abs(lp_change)})')
            
            # 필드 상태 표시
            monsters, total_atk, kuriboh_status = get_field_status(env.simulator.game_state)
            if monsters:
                print(f'   Field: {", ".join(monsters)}')
                print(f'   Total ATK: {total_atk} | {kuriboh_status}')
                
                # 승리 가능 여부
                if total_atk >= 4200 and not env.simulator.game_state.opponent.hand:
                    print(f'    VICTORY CONDITION MET! ({total_atk} >= 4200, Kuriboh removed)')
        
        prev_lp = current_lp
        
        if terminated or truncated:
            if verbose:
                print()
                print('═' * 70)
                if info.get('result') == 'win':
                    print(' VICTORY!')
                elif info.get('result') == 'lose':
                    print(' DEFEAT')
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
    parser = argparse.ArgumentParser(description='Watch trained agent play Ancient Kings puzzle')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model (.pth)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('--algorithm', type=str, default='reinforce_baseline',
                        choices=['reinforce_baseline', 'a2c'], help='Algorithm used for training')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    print('═' * 70)
    print(' ANCIENT KINGS PUZZLE - TRAINED AGENT PLAYBACK')
    print('═' * 70)
    
    # 환경 생성 (Ancient Kings = puzzle_id 1)
    env = AncientKingsEnv()
    
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
    
    if 'policy_net' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        if hasattr(agent, 'value_net') and 'value_net' in checkpoint:
            agent.value_net.load_state_dict(checkpoint['value_net'])
    elif 'ac_net' in checkpoint:
        if hasattr(agent, 'ac_net'):
            agent.ac_net.load_state_dict(checkpoint['ac_net'])
        else:
            print(' Model is A2C format but agent is REINFORCE. Use --algorithm a2c')
            sys.exit(1)
    else:
        first_key = list(checkpoint.keys())[0]
        if '.' in first_key:
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
