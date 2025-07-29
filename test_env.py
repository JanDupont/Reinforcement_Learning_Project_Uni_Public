import numpy as np
from ray_env import PistonBallRLlibEnvironment

def test_environment():
    """Test the environment wrapper"""
    
    print("Creating environment...")
    env = PistonBallRLlibEnvironment({
        'N_PISTONS': 3,  # Smaller for testing
        'MAX_CYCLES': 50,
        'CONTINUOUS': True,
        'ENABLE_GOVERNANCE': False
    })
    
    print(f"Possible agents: {env.possible_agents}")
    print(f"Observation spaces: {list(env.observation_spaces.keys())}")
    print(f"Action spaces: {list(env.action_spaces.keys())}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Initial observations: {list(obs.keys())}")
    
    # Test a few steps
    print("\nTesting steps...")
    for step_num in range(10):
        # Create random actions for all agents
        actions = {}
        for agent in obs.keys():
            if env.continuous:
                actions[agent] = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
            else:
                actions[agent] = np.random.randint(0, 3)
        
        print(f"\nStep {step_num + 1}")
        print(f"Actions: {actions}")
        
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        print(f"Observations: {list(obs.keys())}")
        print(f"Rewards: {rewards}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        
        if terminated.get('__all__', False) or truncated.get('__all__', False):
            print("Episode ended!")
            break
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_environment()