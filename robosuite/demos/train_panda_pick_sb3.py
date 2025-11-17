"""
Training script for Panda robot to pick a box using stable-baselines3.

This script extends demo_gym_functionality.py to train a policy using PPO or SAC.
Follows the same pattern as the existing demo.

Usage:
    python train_panda_pick_sb3.py --algorithm PPO --total_timesteps 100000
    python train_panda_pick_sb3.py --algorithm SAC --total_timesteps 100000 --render
"""

import argparse
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def create_env(render=False):
    """
    Create the Lift environment with Panda robot.
    Same pattern as demo_gym_functionality.py
    """
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="Panda",
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=render,
            reward_shaping=True,
            control_freq=20,
        )
    )
    return env


def train(algorithm="PPO", total_timesteps=100000, render=False, save_path="./panda_pick_model"):
    """
    Train a policy using stable-baselines3.
    
    Args:
        algorithm: "PPO" or "SAC"
        total_timesteps: Total training timesteps
        render: Whether to render during training (slower)
        save_path: Path to save the trained model
    """
    print(f"Creating environment...")
    env = create_env(render=render)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"\nTraining {algorithm} for {total_timesteps} timesteps...")
    
    # Create the agent
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./tensorboard_logs/",
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            tensorboard_log="./tensorboard_logs/",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'PPO' or 'SAC'")
    
    # Create callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"./best_{algorithm.lower()}_model/",
        log_path=f"./logs/{algorithm.lower()}/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./checkpoints/{algorithm.lower()}/",
        name_prefix=f"{algorithm.lower()}_model"
    )
    
    # Train the agent
    print(f"\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save(save_path)
    print(f"\nTraining completed! Model saved to {save_path}")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_episodes = 5
    for episode in range(test_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    print(f"\nTo visualize training progress, run: tensorboard --logdir ./tensorboard_logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Panda robot to pick box using stable-baselines3")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="RL algorithm to use (default: PPO)"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during training (slower)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./panda_pick_model",
        help="Path to save the trained model (default: ./panda_pick_model)"
    )
    
    args = parser.parse_args()
    
    train(
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        render=args.render,
        save_path=args.save_path
    )

