import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import os
from ray_models import PistonParametricModel, GovernanceModel

def setup_training():
    """Setup training configurations for different experiments"""
    
    # Register custom models
    from ray.rllib.models import ModelCatalog
    from ray_env import PistonBallRLlibEnvironment
    
    ModelCatalog.register_custom_model("piston_parametric", PistonParametricModel)
    ModelCatalog.register_custom_model("governance_model", GovernanceModel)
    
    # Policy mapping function for multi-agent scenarios
    def policy_mapping_fn(agent_id, episode, **kwargs):
        if "governance" in agent_id:
            return "governance"
        else:
            return "piston"
    
    configs = {
        # Baseline: Only pistons, no governance
        'baseline': {
            'env': PistonBallRLlibEnvironment,
            'env_config': {
                'N_PISTONS': 10,
                'MAX_CYCLES': 125,
                'CONTINUOUS': True,
                'ENABLE_GOVERNANCE': False
            },
            'multiagent': {
                'policies': {
                    'piston': (None, None, None, {
                        'model': {
                            'custom_model': 'piston_parametric',
                            'custom_model_config': {}
                        }
                    })
                },
                'policy_mapping_fn': lambda agent_id, episode, **kwargs: 'piston',
                'policies_to_train': ['piston']
            }
        },
        
        # RL-based governance: Train both pistons and governance
        'rl_governance': {
            'env': PistonBallRLlibEnvironment,
            'env_config': {
                'N_PISTONS': 10,
                'MAX_CYCLES': 125,
                'CONTINUOUS': True,
                'ENABLE_GOVERNANCE': True,
                'GOVERNANCE_TYPE': 'learning'  # Learning governance
            },
            'multiagent': {
                'policies': {
                    'piston': (None, None, None, {
                        'model': {
                            'custom_model': 'piston_parametric',
                            'custom_model_config': {}
                        }
                    }),
                    'governance': (None, None, None, {
                        'model': {
                            'custom_model': 'governance_model',
                            'custom_model_config': {}
                        }
                    })
                },
                'policy_mapping_fn': policy_mapping_fn,
                'policies_to_train': ['piston', 'governance']  # Train both
            }
        }
    }
    
    return configs

def run_experiment(experiment_name: str, num_timesteps: int = 125):
    """Run a specific experiment"""
    
    ray.init(ignore_reinit_error=True)
    
    configs = setup_training()
    config = configs[experiment_name]
    
    # PPO configuration - using PyTorch
    ppo_config = (PPOConfig()
                  .environment(**{k: v for k, v in config.items() if k in ['env', 'env_config']})
                  .multi_agent(**config['multiagent'])
                  .api_stack(
                      enable_rl_module_and_learner=False,
                      enable_env_runner_and_connector_v2=False
                  )
                  .training(
                      lr=3e-4,
                      train_batch_size=500,
                      minibatch_size=64,
                      num_epochs=3,
                      gamma=0.99, # Discount factor
                      lambda_=0.95, # GAE parameter (generalized advantage estimate)
                      clip_param=0.2 # PPO clipping parameter
                  )
                  .env_runners(
                      num_env_runners=2,
                      rollout_fragment_length="auto"
                  )
                  .framework('torch')
                  .debugging(log_level="INFO"))
    
    # Create absolute path for results
    results_dir = os.path.abspath("./results/")
    os.makedirs(results_dir, exist_ok=True)
    
    # Run training
    results = tune.run(
        "PPO",
        config=ppo_config.to_dict(),
        stop={'timesteps_total': num_timesteps},
        num_samples=1,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        name=f"pistonball_{experiment_name}",
        storage_path=results_dir,
        verbose=1
    )
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PistonBall with PPO')
    parser.add_argument('--experiment', type=str, choices=['baseline', 'rl_governance', 'all'],
                        default='all', help='Which experiment to run')
    parser.add_argument('--timesteps', type=int, default=125, help='Number of training timesteps')
    
    args = parser.parse_args()
    
    if args.experiment == 'all':
        experiments = ['baseline', 'rl_governance']
    else:
        experiments = [args.experiment]
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp} experiment")
        print(f"{'='*60}")
        
        results = run_experiment(exp, args.timesteps)
        print(f"Completed {exp} experiment")
        
        # Save best checkpoint path for later evaluation
        best_trial = results.get_best_trial(metric="episode_reward_mean", mode="max")
        if best_trial:
            print(f"Best checkpoint: {best_trial.checkpoint.path}")
    
    ray.shutdown()