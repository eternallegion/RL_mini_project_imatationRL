"""
Policy Gradient Agents
- REINFORCE (Monte Carlo Policy Gradient)
- REINFORCE with Baseline
- Actor-Critic (A2C)

References:
- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist RL" (1992)
- Sutton et al., "Policy Gradient Methods for RL with Function Approximation" (1999)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
from typing import Tuple, List, Optional


class PolicyNetwork(nn.Module):
    """정책 네트워크 (액션 확률 출력)"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits (softmax는 나중에)
    
    def get_action_probs(self, x, valid_mask=None):
        """유효한 액션에 대한 확률 분포 반환"""
        logits = self.forward(x)
        
        if valid_mask is not None:
            # 무효한 액션에 -inf 마스킹
            mask = torch.tensor(valid_mask, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, float('-inf'))
        

            # 모두 무효인 경우 방어 코드 (선택 사항)  answpdjqtdmaus wprj
            if not mask.any():
                # 그냥 uniform으로 돌려보내기
                return F.softmax(torch.zeros_like(logits), dim=-1)

        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """가치 네트워크 (상태 가치 추정)"""
    
    def __init__(self, state_size: int, hidden_size: int = 256):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic 공유 네트워크"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        policy_logits = self.actor(x)
        value = self.critic(x)
        
        return policy_logits, value
    
    def get_action_and_value(self, x, valid_mask=None):
        policy_logits, value = self.forward(x)
        
        if valid_mask is not None:
            mask = torch.tensor(valid_mask, dtype=torch.bool, device=policy_logits.device)
            policy_logits = policy_logits.masked_fill(~mask, float('-inf'))
        
        probs = F.softmax(policy_logits, dim=-1)
        return probs, value


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) Agent
    
    loss = -sum(log π(a|s) * G)
    where G is the discounted return from that step
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        lr: float = 0.001,
        gamma: float = 0.99,
        use_baseline: bool = True,
        baseline_lr: float = 0.001,
        device: str = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Policy network
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Baseline (optional)
        if use_baseline:
            self.value_net = ValueNetwork(state_size, hidden_size).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=baseline_lr)
        
        # Episode memory
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_states = []
        
        # Training stats
        self.losses = []
    
    def select_action(self, state: np.ndarray, valid_mask: np.ndarray, training: bool = True) -> int:
        """액션 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.policy_net.get_action_probs(state_tensor, valid_mask)
        
        # 항상 확률적 샘플링 (Policy Gradient는 stochastic policy)
        dist = Categorical(probs)
        action = dist.sample()
        
        if training:
            # 훈련 시에만 log_prob 저장
            with torch.enable_grad():
                probs_grad = self.policy_net.get_action_probs(state_tensor, valid_mask)
                dist_grad = Categorical(probs_grad)
                self.saved_log_probs.append(dist_grad.log_prob(action))
                self.saved_states.append(state)
        
        return action.item()
    
    def store_reward(self, reward: float):
        """보상 저장"""
        self.saved_rewards.append(reward)
    
    def learn(self) -> float:
        """
        에피소드 종료 후 학습
        
        Returns:
            policy_loss: 정책 손실
        """
        if len(self.saved_rewards) == 0:
            return 0.0
        
        # 할인된 리턴 계산
        returns = []
        G = 0
        for r in reversed(self.saved_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 정규화 (학습 안정성)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Baseline 학습 (있는 경우)
        if self.use_baseline:
            states_tensor = torch.FloatTensor(self.saved_states).to(self.device)
            values = self.value_net(states_tensor).squeeze()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # Advantage = Return - Baseline
            with torch.no_grad():
                values = self.value_net(states_tensor).squeeze()
                advantages = returns - values
        else:
            advantages = returns
        
        # Policy loss
        policy_loss = 0
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            policy_loss -= log_prob * advantage
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        loss_value = policy_loss.item()
        self.losses.append(loss_value)
        
        # 메모리 클리어
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_states = []
        
        return loss_value
    
    def save(self, path: str):
        """모델 저장"""
        state_dict = {
            'policy_net': self.policy_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()
        }
        if self.use_baseline:
            state_dict['value_net'] = self.value_net.state_dict()
            state_dict['value_optimizer'] = self.value_optimizer.state_dict()
        torch.save(state_dict, path)
    
    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if self.use_baseline and 'value_net' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent
    
    Actor loss: -log π(a|s) * A(s, a)
    Critic loss: (V(s) - G)²
    where A(s, a) = r + γV(s') - V(s)
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        lr: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Actor-Critic network
        self.ac_net = ActorCriticNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
        # Episode memory
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_rewards = []
        self.saved_entropies = []
        
        # Stats
        self.losses = []
    
    def select_action(self, state: np.ndarray, valid_mask: np.ndarray, training: bool = True) -> int:
        """액션 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.ac_net.get_action_and_value(state_tensor, valid_mask)
        
        # 항상 확률적 샘플링 (Policy Gradient는 stochastic policy)
        dist = Categorical(probs)
        action = dist.sample()
        
        if training:
            # 훈련 시에만 gradient 계산용 값 저장
            probs_grad, value_grad = self.ac_net.get_action_and_value(state_tensor, valid_mask)
            dist_grad = Categorical(probs_grad)
            
            self.saved_log_probs.append(dist_grad.log_prob(action))
            self.saved_values.append(value_grad)
            self.saved_entropies.append(dist_grad.entropy())
        
        return action.item()
    
    def store_reward(self, reward: float):
        """보상 저장"""
        self.saved_rewards.append(reward)
    
    def learn(self) -> float:
        """에피소드 종료 후 학습"""
        if len(self.saved_rewards) == 0:
            return 0.0
        
        # 할인된 리턴 계산
        returns = []
        G = 0
        for r in reversed(self.saved_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.cat(self.saved_values).squeeze()
        log_probs = torch.stack(self.saved_log_probs)
        entropies = torch.stack(self.saved_entropies)
        
        # Advantage
        advantages = returns - values.detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
        self.optimizer.step()
        
        loss_value = total_loss.item()
        self.losses.append(loss_value)
        
        # 메모리 클리어
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_rewards = []
        self.saved_entropies = []
        
        return loss_value
    
    def save(self, path: str):
        torch.save({
            'ac_net': self.ac_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.ac_net.load_state_dict(checkpoint['ac_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def train_policy_gradient(
    env,
    agent,  # REINFORCEAgent or A2CAgent
    num_episodes: int = 1000,
    max_steps: int = 30,
    verbose: bool = True,
    eval_freq: int = 100,
    save_best: bool = True
) -> dict:
    """
    Policy Gradient 훈련 루프
    """
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'win_rates': [],
        'losses': []
    }
    
    wins = 0
    recent_rewards = deque(maxlen=100)
    recent_wins = deque(maxlen=100)
    
    # Best model tracking
    best_win_rate = 0.0
    best_model_state = None
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            valid_mask = env.get_valid_actions()
            action = agent.select_action(state, valid_mask, training=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_reward(reward)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 에피소드 종료 후 학습
        loss = agent.learn()
        history['losses'].append(loss)
        
        # 기록
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(step + 1)
        
        recent_rewards.append(episode_reward)
        win = 1 if info.get('result') == 'win' else 0
        recent_wins.append(win)
        wins += win
        
        # Best model 저장
        if save_best and len(recent_wins) >= 50:
            current_win_rate = np.mean(recent_wins)
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                # 모델 상태 복사
                if hasattr(agent, 'ac_net'):  # A2C
                    best_model_state = {
                        'ac_net': {k: v.clone() for k, v in agent.ac_net.state_dict().items()},
                        'optimizer': agent.optimizer.state_dict()
                    }
                else:  # REINFORCE
                    best_model_state = {
                        'policy_net': {k: v.clone() for k, v in agent.policy_net.state_dict().items()},
                        'policy_optimizer': agent.policy_optimizer.state_dict()
                    }
                    if agent.use_baseline:
                        best_model_state['value_net'] = {k: v.clone() for k, v in agent.value_net.state_dict().items()}
                
                if verbose:
                    print(f"  [NEW BEST] Win Rate: {current_win_rate*100:.1f}%")
        
        # 로깅
        if verbose and (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)
            history['win_rates'].append(win_rate)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Win Rate: {win_rate*100:.1f}%")
            print(f"  Total Wins: {wins}")
            print()
    
    # 훈련 종료 후 best model 복원
    if save_best and best_model_state is not None:
        if verbose:
            print(f"\n[RESTORING BEST MODEL] Win Rate: {best_win_rate*100:.1f}%")
        
        if hasattr(agent, 'ac_net'):  # A2C
            agent.ac_net.load_state_dict(best_model_state['ac_net'])
        else:  # REINFORCE
            agent.policy_net.load_state_dict(best_model_state['policy_net'])
            if agent.use_baseline and 'value_net' in best_model_state:
                agent.value_net.load_state_dict(best_model_state['value_net'])
    
    history['best_win_rate'] = best_win_rate
    return history


def evaluate_pg_agent(env, agent, num_episodes: int = 100) -> dict:
    """Policy Gradient 에이전트 평가"""
    wins = 0
    total_reward = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(30):
            valid_mask = env.get_valid_actions()
            action = agent.select_action(state, valid_mask, training=False)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
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
