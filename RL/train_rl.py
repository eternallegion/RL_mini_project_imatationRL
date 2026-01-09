"""
Yu-Gi-Oh Puzzle Duel - Reinforcement Learning Training
여러 RL 알고리즘 비교 실험 (간소화 버전)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from collections import defaultdict
import argparse

from env_wrapper import YuGiOhPuzzleEnv, make_env
from dqn_agent import DQNAgent, train_dqn, evaluate_agent
from policy_gradient_agent import REINFORCEAgent, A2CAgent, train_policy_gradient, evaluate_pg_agent


def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def smooth(data, window=10):
    """이동 평균으로 데이터 스무딩"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed


def get_action_name(action_obj, game_state):
    """액션을 읽기 쉬운 이름으로 변환"""
    from actions import ActionType
    
    if action_obj.action_type == ActionType.ACTIVATE_TRAP:
        zone = action_obj.zone_index
        card = game_state.player.spell_trap_zones[zone]
        return f'ACTIVATE_TRAP: {card.name if card else "Unknown"} (zone {zone})'
    
    elif action_obj.action_type == ActionType.ACTIVATE_SPELL:
        idx = action_obj.card_index
        card = game_state.player.hand[idx] if idx < len(game_state.player.hand) else None
        return f'ACTIVATE_SPELL: {card.name if card else "Unknown"} (hand {idx})'
    
    elif action_obj.action_type == ActionType.ATTACK:
        attacker_zone = action_obj.card_index
        target_zone = action_obj.target_index
        attacker = game_state.player.monster_zones[attacker_zone]
        if target_zone < 5:
            target = game_state.opponent.monster_zones[target_zone]
            target_name = target.name if target else 'Empty'
        else:
            target_name = 'Direct'
        return f'ATTACK: {attacker.name if attacker else "Unknown"} → {target_name}'
    
    elif action_obj.action_type == ActionType.CHANGE_PHASE:
        return f'CHANGE_PHASE: {game_state.phase} → next'
    
    elif action_obj.action_type == ActionType.END_TURN:
        return 'END_TURN'
    
    return str(action_obj)


def visualize_agent_play(env, agent, puzzle_name: str, num_episodes: int = 1):
    """
    학습된 에이전트의 플레이를 시각화
    
    Args:
        env: 게임 환경
        agent: 학습된 에이전트
        puzzle_name: 퍼즐 이름
        num_episodes: 시각화할 에피소드 수
    """
    from actions import ActionSpace
    
    print("\n" + "=" * 70)
    print(f" TRAINED AGENT PLAYING: {puzzle_name}")
    print("=" * 70)
    
    wins = 0
    total_rewards = []
    
    for ep in range(num_episodes):
        if num_episodes > 1:
            print(f'\n{"=" * 70}')
            print(f' Episode {ep + 1}/{num_episodes}')
            print('=' * 70)
        
        state, info = env.reset()
        env.verbose = True  # 상세 출력 활성화
        
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
        
        # 상대 몬스터 출력 (있는 경우)
        opp_monsters = [(m.name, m.atk, m.position.value) if m else None 
                        for m in env.simulator.game_state.opponent.monster_zones]
        if any(m for m in opp_monsters):
            print('   Opponent Monsters:', opp_monsters)
            print()
        
        # 상대 패 출력 (있는 경우)
        if env.simulator.game_state.opponent.hand:
            print('   Opponent Hand:', [c.name for c in env.simulator.game_state.opponent.hand])
            print()
        
        total_reward = 0
        step = 1
        
        while step <= 30:
            # 에이전트가 액션 선택
            valid_mask = env.get_valid_actions()
            action = agent.select_action(state, valid_mask, training=False)
            
            # 액션 이름 가져오기
            action_obj = ActionSpace.index_to_action(action)
            action_name = get_action_name(action_obj, env.simulator.game_state)
            
            print(f'🔹 Step {step}: {action_name}')
            print('-' * 50)
            
            # 액션 실행
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f'   Reward: {reward:+.2f} (Total: {total_reward:.2f})')
            
            if terminated or truncated:
                print()
                print('=' * 70)
                if info.get('result') == 'win':
                    print(' VICTORY!')
                    wins += 1
                elif info.get('result') == 'lose':
                    print(' DEFEAT')
                else:
                    print(f' {info.get("result", "timeout").upper()}')
                print('=' * 70)
                break
            
            state = next_state
            step += 1
            print()
        
        total_rewards.append(total_reward)
        
        print()
        print(' Episode Summary:')
        print(f'   Result: {info.get("result", "timeout")}')
        print(f'   Total Steps: {step}')
        print(f'   Total Reward: {total_reward:.2f}')
        print(f'   Player LP: {env.simulator.game_state.player.lp}')
        print(f'   Opponent LP: {env.simulator.game_state.opponent.lp}')
    
    # 전체 통계 (여러 에피소드인 경우)
    if num_episodes > 1:
        print()
        print('=' * 70)
        print(' OVERALL STATISTICS')
        print('=' * 70)
        print(f'   Win Rate: {wins}/{num_episodes} ({wins/num_episodes*100:.1f}%)')
        print(f'   Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}')
    
    return wins, total_rewards


def create_agent(algorithm: str, state_size: int, action_size: int, 
                hidden_size: int = 256, lr: float = 0.0005, gamma: float = 0.99):
    """알고리즘에 따른 에이전트 생성"""
    
    if algorithm == 'dqn':
        return DQNAgent(
            state_size, action_size,
            hidden_size=hidden_size, lr=lr, gamma=gamma,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998,
            buffer_size=50000, batch_size=128, target_update_freq=20,
            double_dqn=False, dueling=False
        )
    elif algorithm == 'double_dqn':
        return DQNAgent(
            state_size, action_size,
            hidden_size=hidden_size, lr=lr, gamma=gamma,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998,
            buffer_size=50000, batch_size=128, target_update_freq=20,
            double_dqn=True, dueling=False
        )
    elif algorithm == 'dueling_dqn':
        return DQNAgent(
            state_size, action_size,
            hidden_size=hidden_size, lr=lr, gamma=gamma,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998,
            buffer_size=50000, batch_size=128, target_update_freq=20,
            double_dqn=True, dueling=True
        )
    elif algorithm == 'reinforce':
        return REINFORCEAgent(
            state_size, action_size,
            hidden_size=hidden_size, lr=lr*2, gamma=gamma,
            use_baseline=False
        )
    elif algorithm == 'reinforce_baseline':
        return REINFORCEAgent(
            state_size, action_size,
            hidden_size=hidden_size, lr=lr*2, gamma=gamma,
            use_baseline=True, baseline_lr=lr
        )
    elif algorithm == 'a2c':
        return A2CAgent(
            state_size, action_size,
            hidden_size=hidden_size, lr=lr*2, gamma=gamma,
            entropy_coef=0.01,  # 0.05 → 0.01 (탐험 줄이고 안정화)
            value_coef=0.5
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_agent(env, agent, algorithm: str, num_episodes: int, verbose: bool = True):
    """에이전트 훈련"""
    if algorithm in ['dqn', 'double_dqn', 'dueling_dqn']:
        return train_dqn(env, agent, num_episodes=num_episodes, 
                        verbose=verbose, eval_freq=500)
    else:
        return train_policy_gradient(env, agent, num_episodes=num_episodes,
                                    verbose=verbose, eval_freq=500)


def evaluate(env, agent, algorithm: str, num_episodes: int = 100):
    """에이전트 평가"""
    if algorithm in ['dqn', 'double_dqn', 'dueling_dqn']:
        return evaluate_agent(env, agent, num_episodes=num_episodes)
    else:
        return evaluate_pg_agent(env, agent, num_episodes=num_episodes)


def run_single_experiment(puzzle_id: int, algorithm: str, num_episodes: int,
                          seed: int, lr: float, gamma: float, verbose: bool = True):
    """단일 실험 실행"""
    set_seed(seed)
    
    env = make_env(puzzle_id=puzzle_id)
    state_size = env.observation_size
    action_size = env.action_size
    
    agent = create_agent(algorithm, state_size, action_size, lr=lr, gamma=gamma)
    history = train_agent(env, agent, algorithm, num_episodes, verbose=verbose)
    eval_results = evaluate(env, agent, algorithm)
    
    return {
        'algorithm': algorithm,
        'puzzle_id': puzzle_id,
        'seed': seed,
        'num_episodes': num_episodes,
        'history': history,
        'eval_results': eval_results
    }, agent


def plot_learning_curves(results_list: list, title: str, save_path: str = None):
    """학습 곡선 그래프"""
    plt.figure(figsize=(15, 5))
    
    # Episode Rewards
    plt.subplot(1, 3, 1)
    for results in results_list:
        rewards = smooth(results['history']['episode_rewards'], window=50)
        plt.plot(rewards, label=results['algorithm'])
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Learning Curve (Rewards)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Win Rates
    plt.subplot(1, 3, 2)
    for results in results_list:
        if 'win_rates' in results['history'] and len(results['history']['win_rates']) > 0:
            win_rates = results['history']['win_rates']
            episodes = np.linspace(0, results['num_episodes'], len(win_rates))
            plt.plot(episodes, [w*100 for w in win_rates], label=results['algorithm'], marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final Comparison
    plt.subplot(1, 3, 3)
    algorithms = [r['algorithm'] for r in results_list]
    win_rates = [r['eval_results']['win_rate']*100 for r in results_list]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(algorithms)))
    plt.bar(algorithms, win_rates, color=colors)
    plt.ylabel('Final Win Rate (%)')
    plt.title('Final Performance')
    plt.xticks(rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_comparison_with_ci(summaries: list, title: str, save_path: str = None):
    """신뢰구간 비교 그래프"""
    plt.figure(figsize=(10, 6))
    
    algorithms = [s['algorithm'] for s in summaries]
    means = [s['win_rate_mean']*100 for s in summaries]
    stds = [s['win_rate_std']*100 for s in summaries]
    
    x = np.arange(len(algorithms))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(algorithms)))
    
    bars = plt.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Win Rate (%)')
    plt.title(f'{title}\n(Mean ± Std over {len(summaries[0]["seeds"])} seeds)')
    plt.xticks(x, algorithms, rotation=45, ha='right')
    plt.ylim(0, 110)
    
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Yu-Gi-Oh Puzzle RL Training')
    parser.add_argument('--puzzle', type=str, default='0', 
                       help='Puzzle ID: 0, 1, or all')
    parser.add_argument('--algorithm', type=str, default='all', 
                       choices=['dqn', 'double_dqn', 'dueling_dqn', 
                               'reinforce', 'reinforce_baseline', 'a2c', 'all'])
    parser.add_argument('--episodes', type=int, default=5000, 
                       help='Training episodes')
    parser.add_argument('--seeds', type=int, default=3, 
                       help='Number of seeds for CI')
    parser.add_argument('--lr', type=float, default=0.0005, 
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, 
                       help='Discount factor')
    parser.add_argument('--save_dir', type=str, default='results', 
                       help='Directory to save results')
    parser.add_argument('--skip-ci', action='store_true',
                       help='Skip Phase 2 (multiple seeds CI)')
    parser.add_argument('--ci-episodes', type=int, default=None,
                       help='Episodes for CI (default: episodes/5)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show trained agent playing after training')
    parser.add_argument('--visualize-episodes', type=int, default=1,
                       help='Number of episodes to visualize (default: 1)')
    args = parser.parse_args()
    
    # 퍼즐 목록 결정
    if args.puzzle == 'all':
        puzzle_ids = [0, 1]
    else:
        puzzle_ids = [int(args.puzzle)]
    
    # 각 퍼즐에 대해 실험 실행
    for puzzle_id in puzzle_ids:
        run_puzzle_experiment(puzzle_id, args)


def run_puzzle_experiment(puzzle_id: int, args):
    """단일 퍼즐에 대한 실험 실행"""
    # 설정
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    puzzle_names = {0: 'Oh_Jama', 1: 'Ancient_Kings'}
    puzzle_name = puzzle_names[puzzle_id]
    
    if args.algorithm == 'all':
        algorithms = ['dqn', 'double_dqn', 'reinforce_baseline', 'a2c']
    else:
        algorithms = [args.algorithm]
    
    print("\n" + "="*60)
    print(f"Yu-Gi-Oh Puzzle Duel - RL Experiment")
    print("="*60)
    print(f"Puzzle: {puzzle_name} (ID: {puzzle_id})")
    print(f"Algorithms: {algorithms}")
    print(f"Episodes: {args.episodes}")
    print(f"Skip CI: {args.skip_ci}")
    print("="*60)
    
    # ============================================================
    # Phase 1: Single Seed Training
    # ============================================================
    print("\n" + "="*60)
    print(f"Phase 1: Single Seed Training - {puzzle_name}")
    print("="*60)
    
    single_results = []
    agents = {}
    
    # 알고리즘별 고정 시드 (all이든 개별이든 같은 시드 사용)
    ALGORITHM_SEEDS = {
        'dqn': 42,
        'double_dqn': 142,
        'dueling_dqn': 242,
        'reinforce': 342,
        'reinforce_baseline': 442,
        'a2c': 542
    }
    
    for i, algo in enumerate(algorithms):
        algo_seed = ALGORITHM_SEEDS.get(algo, 42)
        print(f"\n[{i+1}/{len(algorithms)}] Training {algo.upper()} (seed={algo_seed})...")
        
        results, agent = run_single_experiment(
            puzzle_id=puzzle_id,
            algorithm=algo,
            num_episodes=args.episodes,
            seed=algo_seed,
            lr=args.lr,
            gamma=args.gamma,
            verbose=True
        )
        single_results.append(results)
        agents[algo] = agent
        
        print(f"\n--- {algo.upper()} Final Evaluation ---")
        print(f"Win Rate: {results['eval_results']['win_rate']*100:.1f}%")
        print(f"Avg Reward: {results['eval_results']['avg_reward']:.2f}")
    
    # 학습 곡선 저장
    plot_learning_curves(
        single_results,
        f'Learning Curves - {puzzle_name}',
        save_path=f'{args.save_dir}/learning_curves_{puzzle_name}_{timestamp}.png'
    )
    
    # ============================================================
    # Phase 2: Multiple Seeds (Optional)
    # ============================================================
    if not args.skip_ci and args.seeds > 1:
        print("\n" + "="*60)
        print(f"Phase 2: Multiple Seeds ({args.seeds} seeds) - {puzzle_name}")
        print("="*60)
        
        ci_episodes = args.ci_episodes if args.ci_episodes else max(args.episodes // 5, 1000)
        seeds = [42 + i*100 for i in range(args.seeds)]
        summaries = []
        
        total_runs = len(algorithms) * len(seeds)
        current_run = 0
        
        for algo in algorithms:
            algo_win_rates = []
            algo_rewards = []
            
            for seed in seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {algo.upper()} seed={seed} ({ci_episodes} eps)")
                
                results, _ = run_single_experiment(
                    puzzle_id=puzzle_id,
                    algorithm=algo,
                    num_episodes=ci_episodes,
                    seed=seed,
                    lr=args.lr,
                    gamma=args.gamma,
                    verbose=False
                )
                
                algo_win_rates.append(results['eval_results']['win_rate'])
                algo_rewards.append(results['eval_results']['avg_reward'])
                print(f"  Win Rate: {results['eval_results']['win_rate']*100:.1f}%")
            
            summary = {
                'algorithm': algo,
                'seeds': seeds,
                'win_rate_mean': np.mean(algo_win_rates),
                'win_rate_std': np.std(algo_win_rates),
                'reward_mean': np.mean(algo_rewards),
                'reward_std': np.std(algo_rewards)
            }
            summaries.append(summary)
            
            print(f"\n{algo.upper()} Summary:")
            print(f"  Win Rate: {summary['win_rate_mean']*100:.1f}% ± {summary['win_rate_std']*100:.1f}%")
        
        # CI 그래프 저장
        plot_comparison_with_ci(
            summaries,
            f'Algorithm Comparison - {puzzle_name}',
            save_path=f'{args.save_dir}/comparison_{puzzle_name}_{timestamp}.png'
        )
    
    # ============================================================
    # 결과 저장
    # ============================================================
    best_algo = max(single_results, key=lambda x: x['eval_results']['win_rate'])
    best_agent = agents[best_algo['algorithm']]
    best_agent.save(f'{args.save_dir}/best_model_{puzzle_name}_{timestamp}.pth')
    
    # JSON 결과 저장
    results_summary = {
        'puzzle': puzzle_name,
        'puzzle_id': puzzle_id,
        'timestamp': timestamp,
        'num_episodes': args.episodes,
        'results': [{
            'algorithm': r['algorithm'],
            'win_rate': r['eval_results']['win_rate'],
            'avg_reward': r['eval_results']['avg_reward']
        } for r in single_results]
    }
    
    with open(f'{args.save_dir}/results_{puzzle_name}_{timestamp}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print(f"EXPERIMENT COMPLETE - {puzzle_name}")
    print("="*60)
    print(f"\nBest Algorithm: {best_algo['algorithm']}")
    print(f"Best Win Rate: {best_algo['eval_results']['win_rate']*100:.1f}%")
    print(f"Results saved to: {args.save_dir}/")
    
    # ============================================================
    # 시각화 (선택적)
    # ============================================================
    if args.visualize:
        print("\n" + "=" * 60)
        print(" VISUALIZATION MODE")
        print("=" * 60)
        
        # 최고 성능 에이전트로 플레이 시각화
        viz_env = make_env(puzzle_id=puzzle_id)
        visualize_agent_play(
            viz_env, 
            best_agent, 
            puzzle_name, 
            num_episodes=args.visualize_episodes
        )


if __name__ == "__main__":
    main()
