import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from pettingzoo.butterfly import pistonball_v6

# Import custom models
from ray_models import PistonParametricModel, GovernanceModel

# Define assumed training parameters (or retrieve them if stored in checkpoint/config)
# PistonBallRLlibEnvironment defaults: n_pistons=10, max_cycles=1000
ASSUMED_TRAINING_N_PISTONS = 10
ASSUMED_TRAINING_MAX_CYCLES = 10000

# Predefined list of 100 seeds for reproducible evaluation
EVALUATION_SEEDS = [
    42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021,
    2223, 2425, 2627, 2829, 3031, 3233, 3435, 3637, 3839, 4041,
    4243, 4445, 4647, 4849, 5051, 5253, 5455, 5657, 5859, 6061,
    6263, 6465, 6667, 6869, 7071, 7273, 7475, 7677, 7879, 8081,
    8283, 8485, 8687, 8889, 9091, 9293, 9495, 9697, 9899, 10101,
    10203, 10405, 10607, 10809, 11011, 11213, 11415, 11617, 11819, 12021,
    12223, 12425, 12627, 12829, 13031, 13233, 13435, 13637, 13839, 14041,
    14243, 14445, 14647, 14849, 15051, 15253, 15455, 15657, 15859, 16061,
    16263, 16465, 16667, 16869, 17071, 17273, 17475, 17677, 17879, 18081,
    18283, 18485, 18687, 18889, 19091, 19293, 19495, 19697, 19899, 20101
]

def evaluate_trained_model(checkpoint_path: str, num_episodes: int = 100, max_episode_length: int = 10000, experiment_name: str = None):
    """Load and test a trained model using native PettingZoo environment"""
    
    # Convert to absolute path
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Register custom models BEFORE loading checkpoint
        ModelCatalog.register_custom_model("piston_parametric", PistonParametricModel)
        ModelCatalog.register_custom_model("governance_model", GovernanceModel)
        
        # Load the trained algorithm
        algo = PPO.from_checkpoint(checkpoint_path)
        
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        successful_episode_rewards = []  # Track rewards for successful episodes only
        
        print("Starting evaluation...")
        
        for episode in range(num_episodes):
            # Get seed for this episode (cycle through if more episodes than seeds)
            seed = EVALUATION_SEEDS[episode % len(EVALUATION_SEEDS)]
            
            # Create native PettingZoo environment
            env = pistonball_v6.env(
                render_mode="rgb_array", # None leads to terrible model performance? so we use rgb_array
                n_pistons=ASSUMED_TRAINING_N_PISTONS,
                max_cycles=ASSUMED_TRAINING_MAX_CYCLES
            )
            env.reset(seed=seed)
            
            episode_reward = 0
            step_count = 0
            is_success = False
            
            # Use PettingZoo's agent iteration pattern
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                
                episode_reward += reward if reward is not None else 0
                step_count += 1
                
                if termination or truncation:
                    action = None
                else:
                    # Determine which policy to use
                    if "governance" in agent:
                        policy_id = "governance"
                    else:
                        policy_id = "piston"
                    
                    # Get action from trained model
                    action = algo.compute_single_action(observation, policy_id=policy_id)
                
                env.step(action)
                
                # Safety check to prevent infinite episodes
                if step_count > max_episode_length:
                    break
            
            # Episode completed - check if it was natural completion (success)
            # If we didn't hit the max step limit, it's a natural completion
            is_success = step_count <= max_episode_length
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            episode_success.append(is_success)
            
            # Track rewards for successful episodes only
            if is_success:
                successful_episode_rewards.append(episode_reward)
            
            # Close environment
            env.close()
            
            # Print progress
            current_success_rate = sum(episode_success) / len(episode_success)
            print(f"Episode {episode + 1}/{num_episodes} (seed={seed}) - Success: {'✓' if is_success else '✗'}, Reward: {episode_reward:.0f}, Length: {step_count}, Success Rate: {current_success_rate:.1%}")
        
        # Calculate final statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        var_reward = np.var(episode_rewards)
        
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        var_length = np.var(episode_lengths)
        
        success_count = sum(episode_success)
        success_rate = success_count / num_episodes
        
        # Calculate statistics for successful episodes only
        if successful_episode_rewards:
            mean_reward_success = np.mean(successful_episode_rewards)
            std_reward_success = np.std(successful_episode_rewards)
            var_reward_success = np.var(successful_episode_rewards)
        else:
            mean_reward_success = 0
            std_reward_success = 0
            var_reward_success = 0
        
        print(f"\nFinal Evaluation Results ({num_episodes} episodes):")
        print(f"Mean Reward (All): {mean_reward:.2f} ± {std_reward:.2f} (var: {var_reward:.2f})")
        print(f"Mean Reward (Success Only): {mean_reward_success:.2f} ± {std_reward_success:.2f} (var: {var_reward_success:.2f})")
        print(f"Mean Episode Length: {mean_length:.2f} ± {std_length:.2f} (var: {var_length:.2f})")
        print(f"Success Rate: {success_rate:.2%} ({success_count}/{num_episodes})")
        
        # Create visualizations
        results = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'var_reward': var_reward,
            'mean_reward_success': mean_reward_success,
            'std_reward_success': std_reward_success,
            'var_reward_success': var_reward_success,
            'mean_length': mean_length,
            'std_length': std_length,
            'var_length': var_length,
            'success_rate': success_rate,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'successful_episode_rewards': successful_episode_rewards,
            'episode_success': episode_success
        }
        
        # Create and save plots
        _create_evaluation_plots(results, num_episodes, experiment_name or "unknown")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        ray.shutdown()

def _create_evaluation_plots(results, num_episodes, experiment_name):
    """Create and save evaluation plots"""
    
    # Create results directory
    os.makedirs("eval_results", exist_ok=True)
    
    plt.style.use('default')
    
    # Plot 1: Episode rewards over time with std bands
    fig, ax = plt.subplots(figsize=(12, 6))
    episodes = range(1, len(results['episode_rewards']) + 1)
    rewards = results['episode_rewards']
    
    # Calculate rolling mean and std for visualization
    window = 10  # Fixed window size
    rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
    rolling_episodes = episodes[window-1:]
    
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')
    ax.plot(rolling_episodes, rolling_mean, color='blue', linewidth=2, label=f'Rolling Mean (window={window})')
    
    # Add horizontal lines for overall statistics
    ax.axhline(y=results['mean_reward'], color='red', linestyle='--', alpha=0.7, 
               label=f"Mean All: {results['mean_reward']:.1f}±{results['std_reward']:.1f}")
    ax.fill_between(episodes, 
                    results['mean_reward'] - results['std_reward'],
                    results['mean_reward'] + results['std_reward'],
                    alpha=0.2, color='red')
    
    if results['successful_episode_rewards']:
        ax.axhline(y=results['mean_reward_success'], color='green', linestyle='--', alpha=0.7,
                   label=f"Mean Success: {results['mean_reward_success']:.1f}±{results['std_reward_success']:.1f}")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"eval_results/{experiment_name}_rewards_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Episode lengths over time
    fig, ax = plt.subplots(figsize=(12, 6))
    lengths = results['episode_lengths']
    rolling_mean_length = np.convolve(lengths, np.ones(window)/window, mode='valid')
    
    ax.plot(episodes, lengths, alpha=0.3, color='orange', label='Episode Lengths')
    ax.plot(rolling_episodes, rolling_mean_length, color='orange', linewidth=2, label=f'Rolling Mean (window={window})')
    
    # Add horizontal lines for statistics
    ax.axhline(y=results['mean_length'], color='red', linestyle='--', alpha=0.7,
               label=f"Mean: {results['mean_length']:.1f}±{results['std_length']:.1f}")
    ax.fill_between(episodes,
                    results['mean_length'] - results['std_length'],
                    results['mean_length'] + results['std_length'],
                    alpha=0.2, color='red')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Lengths Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"eval_results/{experiment_name}_lengths_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Reward distribution histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # All episodes
    ax1.hist(results['episode_rewards'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(results['mean_reward'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {results['mean_reward']:.1f}")
    ax1.axvline(results['mean_reward'] - results['std_reward'], color='red', linestyle=':', alpha=0.7)
    ax1.axvline(results['mean_reward'] + results['std_reward'], color='red', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution (All Episodes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Successful episodes only
    if results['successful_episode_rewards']:
        ax2.hist(results['successful_episode_rewards'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(results['mean_reward_success'], color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {results['mean_reward_success']:.1f}")
        ax2.axvline(results['mean_reward_success'] - results['std_reward_success'], color='red', linestyle=':', alpha=0.7)
        ax2.axvline(results['mean_reward_success'] + results['std_reward_success'], color='red', linestyle=':', alpha=0.7)
        ax2.set_title(f'Reward Distribution (Successful Episodes, n={len(results["successful_episode_rewards"])})')
    else:
        ax2.text(0.5, 0.5, 'No Successful Episodes', transform=ax2.transAxes, 
                 ha='center', va='center', fontsize=16)
        ax2.set_title('Reward Distribution (Successful Episodes)')
    
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"eval_results/{experiment_name}_reward_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Success Rate over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative success rate over episodes
    cumulative_success = np.cumsum(results['episode_success']) / np.arange(1, len(results['episode_success']) + 1)
    
    ax.plot(episodes, [s * 100 for s in cumulative_success], color='green', linewidth=2, label='Cumulative Success Rate')
    ax.axhline(y=results['success_rate'] * 100, color='red', linestyle='--', alpha=0.7,
               label=f"Final Success Rate: {results['success_rate']:.1%}")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"eval_results/{experiment_name}_success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Mean Length over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative mean length over episodes
    cumulative_mean_length = np.cumsum(results['episode_lengths']) / np.arange(1, len(results['episode_lengths']) + 1)
    rolling_mean_length_display = np.convolve(results['episode_lengths'], np.ones(window)/window, mode='valid')
    
    ax.plot(episodes, results['episode_lengths'], alpha=0.3, color='orange', label='Episode Lengths')
    ax.plot(episodes, cumulative_mean_length, color='blue', linewidth=2, label='Cumulative Mean Length')
    ax.plot(rolling_episodes, rolling_mean_length_display, color='orange', linewidth=2, label=f'Rolling Mean (window={window})')
    ax.axhline(y=results['mean_length'], color='red', linestyle='--', alpha=0.7,
               label=f"Final Mean: {results['mean_length']:.1f}±{results['std_length']:.1f}")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"eval_results/{experiment_name}_mean_length.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Summary statistics bar chart (simplified)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['Mean Reward\n(All)', 'Mean Reward\n(Success)']
    values = [results['mean_reward'], results['mean_reward_success']]
    errors = [results['std_reward'], results['std_reward_success']]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax.bar(metrics, values, yerr=errors, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value, error in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + error + max(values)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Reward Summary Statistics')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"eval_results/{experiment_name}_summary_stats.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to eval_results/{experiment_name}_*.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained PistonBall model')
    parser.add_argument('--experiment', type=str, choices=['baseline', 'rl_governance'],
                        required=True, help='Which experiment type the model was trained on')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--max-length', type=int, default=10000, help='Maximum episode length')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
        exit(1)
    
    print(f"Evaluating {args.experiment} model")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Max episode length: {args.max_length}")
    print("-" * 50)
    
    evaluate_trained_model(args.checkpoint, args.episodes, args.max_length, args.experiment)

