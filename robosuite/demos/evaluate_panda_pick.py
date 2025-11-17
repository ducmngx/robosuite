"""
Evaluate a trained model for Panda pick task.

This script evaluates a trained model and reports:
- Success rate (box lifted >4cm above table)
- Average reward per episode
- Episode length statistics
- Visual demonstration (optional)

Usage:
    python evaluate_panda_pick.py --model_path ./panda_pick_model.zip --num_episodes 50
    python evaluate_panda_pick.py --model_path ./panda_pick_model.zip --num_episodes 10 --render
"""

import argparse
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO, SAC


def create_env(render=False):
    """Create the Lift environment with Panda robot."""
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


def check_success(env):
    """
    Check if the task was successful (box lifted >4cm above table).
    
    Note: This accesses the underlying robosuite environment, not the wrapped one.
    """
    unwrapped_env = env.env  # Get the unwrapped robosuite environment
    if hasattr(unwrapped_env, '_check_success'):
        return unwrapped_env._check_success()
    return False


def evaluate_model(model_path, num_episodes=50, render=False, deterministic=True):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the saved model (.zip file)
        num_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
        deterministic: Use deterministic policy (recommended for evaluation)
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"Loading model from {model_path}...")
    
    # Try to load as PPO first, then SAC
    try:
        model = PPO.load(model_path)
        print("Loaded PPO model")
    except:
        try:
            model = SAC.load(model_path)
            print("Loaded SAC model")
        except Exception as e:
            raise ValueError(f"Could not load model. Error: {e}")
    
    # Create environment
    env = create_env(render=render)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Track if success was achieved at any point in the episode
        episode_success = False
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Check for success (check periodically to avoid overhead)
            if episode_length % 10 == 0:
                if check_success(env):
                    episode_success = True
            
            if render:
                env.render()
        
        # Final success check
        if check_success(env):
            episode_success = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_success:
            success_count += 1
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            current_success_rate = success_count / (episode + 1) * 100
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Success={'✓' if episode_success else '✗'}, "
                  f"Success Rate={current_success_rate:.1f}%")
    
    env.close()
    
    # Calculate statistics
    metrics = {
        'num_episodes': num_episodes,
        'success_rate': success_count / num_episodes * 100,
        'success_count': success_count,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }
    
    return metrics


def print_metrics(metrics):
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes evaluated: {metrics['num_episodes']}")
    print(f"\nSuccess Metrics:")
    print(f"  Success Rate: {metrics['success_rate']:.1f}% ({metrics['success_count']}/{metrics['num_episodes']})")
    print(f"\nReward Statistics:")
    print(f"  Average: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print(f"\nEpisode Length:")
    print(f"  Average: {metrics['avg_length']:.0f} ± {metrics['std_length']:.0f} steps")
    print("=" * 60)
    
    # Performance interpretation
    print("\nPerformance Interpretation:")
    if metrics['success_rate'] >= 90:
        print("  ✓ Excellent! Model consistently solves the task.")
    elif metrics['success_rate'] >= 70:
        print("  ✓ Good! Model solves the task most of the time.")
    elif metrics['success_rate'] >= 50:
        print("  ~ Moderate. Model solves the task about half the time.")
    elif metrics['success_rate'] >= 30:
        print("  ~ Needs improvement. Model occasionally solves the task.")
    else:
        print("  ✗ Poor. Model rarely solves the task. Consider more training.")
    
    # Reward interpretation (with reward_shaping=True, max is 1.0)
    print(f"\nReward Interpretation:")
    if metrics['avg_reward'] >= 0.8:
        print("  ✓ High average reward. Model is performing well.")
    elif metrics['avg_reward'] >= 0.5:
        print("  ~ Moderate reward. Model is learning but can improve.")
    else:
        print("  ✗ Low reward. Model needs more training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Panda pick model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate (default: 50)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation (slower but visual)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)"
    )
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        render=args.render,
        deterministic=args.deterministic
    )
    
    # Print results
    print_metrics(metrics)

