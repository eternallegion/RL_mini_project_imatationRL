"""
DQN (Deep Q-Network) Agent
- 기본 DQN
- Double DQN
- Dueling DQN (선택적)

References:
- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional


class ReplayBuffer:
    """경험 재생 버퍼"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, valid_actions_mask):
        """경험 저장"""
        self.buffer.append((state, action, reward, next_state, done, valid_actions_mask))
    
    def sample(self, batch_size: int) -> Tuple:
        """미니배치 샘플링"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, valid_masks = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(valid_masks)
        )
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network (가치 함수 근사)"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN Network"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DuelingQNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Value stream
        self.value_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.value = nn.Linear(hidden_size // 2, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.advantage = nn.Linear(hidden_size // 2, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Value
        v = F.relu(self.value_fc(x))
        v = self.value(v)
        
        # Advantage
        a = F.relu(self.advantage_fc(x))
        a = self.advantage(a)
        
        # Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    """
    DQN Agent
    
    Features:
    - Experience Replay
    - Target Network
    - Epsilon-greedy exploration
    - Valid action masking
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        double_dqn: bool = False,
        dueling: bool = False,
        device: str = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.learn_step = 0
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        NetworkClass = DuelingQNetwork if dueling else QNetwork
        self.q_network = NetworkClass(state_size, action_size, hidden_size).to(self.device)
        self.target_network = NetworkClass(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.losses = []
    
    def select_action(self, state: np.ndarray, valid_mask: np.ndarray, training: bool = True) -> int:
        """
        액션 선택 (epsilon-greedy with valid action masking)
        """
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return 0  # 기본 액션
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return np.random.choice(valid_indices)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Mask invalid actions
            q_values[~valid_mask] = -np.inf
            
            return int(np.argmax(q_values))
    
    def store_transition(self, state, action, reward, next_state, done, valid_mask):
        """경험 저장"""
        self.replay_buffer.push(state, action, reward, next_state, done, valid_mask)
    
    def learn(self) -> Optional[float]:
        """
        미니배치 학습
        
        Returns:
            loss: 학습 손실 (버퍼가 충분하지 않으면 None)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 샘플링
        states, actions, rewards, next_states, dones, valid_masks = \
            self.replay_buffer.sample(self.batch_size)
        
        # 텐서 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 현재 Q값
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 타겟 Q값
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 액션 선택은 q_network, 평가는 target_network
                next_q_values = self.q_network(next_states)
                next_actions = next_q_values.argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Vanilla DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 손실 계산 (Huber Loss for stability)
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 타겟 네트워크 업데이트
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def decay_epsilon(self):
        """Epsilon 감소"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """모델 저장"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step
        }, path)
    
    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']


def train_dqn(
    env,
    agent: DQNAgent,
    num_episodes: int = 1000,
    max_steps: int = 30,
    verbose: bool = True,
    eval_freq: int = 100
) -> dict:
    """
    DQN 훈련 루프
    
    Returns:
        history: 훈련 기록
    """
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'win_rates': [],
        'losses': [],
        'epsilons': []
    }
    
    wins = 0
    recent_rewards = deque(maxlen=100)
    recent_wins = deque(maxlen=100)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 유효한 액션 마스크
            valid_mask = env.get_valid_actions()
            
            # 액션 선택
            action = agent.select_action(state, valid_mask, training=True)
            
            # 환경 실행
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 다음 상태의 유효 액션 마스크
            next_valid_mask = env.get_valid_actions() if not done else np.zeros(agent.action_size, dtype=bool)
            
            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done, next_valid_mask)
            
            # 학습
            loss = agent.learn()
            if loss is not None:
                history['losses'].append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Epsilon 감소
        agent.decay_epsilon()
        
        # 기록
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(step + 1)
        history['epsilons'].append(agent.epsilon)
        
        recent_rewards.append(episode_reward)
        win = 1 if info.get('result') == 'win' else 0
        recent_wins.append(win)
        wins += win
        
        # 로깅
        if verbose and (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)
            history['win_rates'].append(win_rate)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Win Rate: {win_rate*100:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Total Wins: {wins}")
            print()
    
    return history


def evaluate_agent(env, agent: DQNAgent, num_episodes: int = 100, verbose: bool = False) -> dict:
    """
    에이전트 평가
    """
    wins = 0
    total_reward = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        if verbose and episode == 0:
            env.verbose = True
            env.render()
        
        for step in range(30):
            valid_mask = env.get_valid_actions()
            action = agent.select_action(state, valid_mask, training=False)
            
            if verbose and episode == 0:
                action_obj = ActionSpace.index_to_action(action)
                print(f"Step {step + 1}: {action_obj}")
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        if verbose and episode == 0:
            print(f"Result: {info.get('result', 'unknown')}")
            env.verbose = False
        
        if info.get('result') == 'win':
            wins += 1
        
        total_reward += episode_reward
        episode_lengths.append(step + 1)
    
    return {
        'win_rate': wins / num_episodes,
        'avg_reward': total_reward / num_episodes,
        'avg_length': np.mean(episode_lengths),
        'wins': wins
    }


# Import for evaluate_agent
from actions import ActionSpace
