import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
import argparse
import os
from pettingzoo.butterfly import pistonball_v6

# Import custom models
from ray_models import PistonParametricModel, GovernanceModel

# Define assumed training parameters (or retrieve them if stored in checkpoint/config)
# PistonBallRLlibEnvironment defaults: n_pistons=20, max_cycles=1000
ASSUMED_TRAINING_N_PISTONS = 10
ASSUMED_TRAINING_MAX_CYCLES = 10000

def test_trained_model(checkpoint_path: str, render_delay: float = 0.05):
    """Load and test a trained model with visual rendering using native PettingZoo environment"""
    
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
        
        # Create native PettingZoo environment with rendering
        # Align with assumed training parameters
        env = pistonball_v6.env(
            render_mode="human",
            n_pistons=ASSUMED_TRAINING_N_PISTONS,
            max_cycles=ASSUMED_TRAINING_MAX_CYCLES
        )
        env.reset()#(seed=42)
        
        print("Starting visual test run...")
        print("Close the render window to end the episode early")
        
        episode_reward = 0
        step_count = 0
        
        # Use PettingZoo's agent iteration pattern
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            # Print the iteration number
            print(f"Step: {step_count}")
            # Print the total reward of all agents
            if reward is not None:
                print(f"Agent {agent} received reward: {reward:.2f}")
            
            episode_reward += reward if reward is not None else 0.0 # Fix: Handle None reward
            step_count += 1
            
            if termination or truncation:
                action = None
            else:
                # Use trained model instead of random action
                # Determine which policy to use
                if "governance" in agent:
                    policy_id = "governance"
                else:
                    policy_id = "piston"
                
                # Get action from trained model
                action = algo.compute_single_action(observation, policy_id=policy_id)
            
            env.step(action)
            
            # # Add delay to make visualization easier to follow
            # if not (termination or truncation):
            #     time.sleep(render_delay)
            
            # Safety check to prevent infinite episodes
            if step_count > 10000:
                print(f"Episode exceeded maximum steps, ending...")
                break
        
        print(f"Episode completed!")
        print(f"Final Episode Reward: {episode_reward:.2f}")
        print(f"Episode Length: {step_count} steps")
        
        # Close environment
        env.close()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    finally:
        ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained PistonBall model with visual rendering')
    parser.add_argument('checkpoint', type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--experiment', type=str, choices=['baseline', 'rl_governance'],
                        default='baseline', help='Which experiment type the model was trained on')
    parser.add_argument('--delay', type=float, default=0.05, 
                        help='Delay between steps in seconds (default: 0.05)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
        exit(1)
    
    print(f"Testing {args.experiment} model with visual rendering")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Step delay: {args.delay}s")
    print("-" * 50)
    
    test_trained_model(args.checkpoint, args.delay)
