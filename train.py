import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch
from trading_env import Nifty500TradingEnv
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Modify network architecture for better performance
        self.policy.mlp_extractor.shared_net = torch.nn.Sequential(
            torch.nn.Linear(self.policy.mlp_extractor.shared_net[0].in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        )

def make_env(api_key: str, train: bool = True):
    """Create and configure the trading environment"""
    def _init():
        env = Nifty500TradingEnv(
            api_key=api_key,
            initial_balance=1000000,
            window_size=60,
            monthly_return_target=0.15,
            stop_loss_pct=0.05
        )
        env = Monitor(env)
        return env
    return _init

def train_agent(api_key: str):
    """Train the trading agent"""
    # Create environments
    env = DummyVecEnv([lambda: make_env(api_key, train=True)()])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(api_key, train=False)()])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Define callbacks
    reward_threshold = 0.15  # 15% monthly return
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        eval_freq=10000,
        n_eval_episodes=10,
        best_model_save_path='./best_model/',
        verbose=1
    )
    
    # Create custom neural network policy
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],  # Policy network
            vf=[512, 512, 256]   # Value function network
        ),
        activation_fn=torch.nn.ReLU
    )
    
    # Initialize agent with custom parameters
    model = CustomPPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,
        tensorboard_log="./tensorboard_log/",
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Train the agent
    total_timesteps = 10_000_000  # Adjust based on convergence
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name=f"ppo_trading_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    
    # Save the final model
    model.save("final_model")
    env.save("final_env_normalize")
    
def evaluate_agent(api_key: str, model_path: str = "best_model/best_model.zip"):
    """Evaluate the trained agent"""
    # Load the saved model and environment
    env = DummyVecEnv([lambda: make_env(api_key, train=False)()])
    env = VecNormalize.load("final_env_normalize", env)
    model = CustomPPO.load(model_path)
    
    # Run evaluation episodes
    n_eval_episodes = 12  # One year of monthly evaluations
    episode_rewards = []
    episode_lengths = []
    monthly_returns = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Track monthly returns
            if 'monthly_return' in info[0]:
                monthly_returns.append(info[0]['monthly_return'])
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"Episode {episode + 1}")
        logger.info(f"Total Reward: {episode_reward}")
        logger.info(f"Monthly Return: {monthly_returns[-1] if monthly_returns else 'N/A'}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_monthly_return = np.mean(monthly_returns)
    success_rate = sum(1 for r in monthly_returns if r >= 0.15) / len(monthly_returns)
    max_drawdown = min(monthly_returns) if monthly_returns else 0
    
    logger.info("\nEvaluation Results:")
    logger.info(f"Average Episode Reward: {avg_reward}")
    logger.info(f"Average Monthly Return: {avg_monthly_return:.2%}")
    logger.info(f"Success Rate (15% Target): {success_rate:.2%}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
    logger.info(f"Risk-Adjusted Return: {avg_monthly_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Train the agent
    train_agent(api_key)
    
    # Evaluate the agent
    evaluate_agent(api_key)