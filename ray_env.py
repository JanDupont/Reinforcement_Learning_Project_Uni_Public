import numpy as np
from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pettingzoo.butterfly import pistonball_v6

# Custom RLlib environment for PistonBall with multi-agent support
class PistonBallRLlibEnvironment(MultiAgentEnv):
    def __init__(self, env_config: dict = {}):
        # Initialize environment configuration
        self.n_pistons = env_config.get('N_PISTONS', 10)
        self.max_cycles = env_config.get('MAX_CYCLES', 125)
        self.continuous = env_config.get('CONTINUOUS', True)
        self.enable_governance = env_config.get('ENABLE_GOVERNANCE', False)
        
        # Create base PistonBall environment
        self.env = pistonball_v6.env(
            n_pistons=self.n_pistons,
            max_cycles=self.max_cycles,
            continuous=self.continuous,
            render_mode=None
        )
        
        # IMPORTANT: Reset before using agent_iter()
        self.env.reset()
        
        # Get agent names from the environment
        self.possible_agents = list(self.env.agents)
        self._agent_ids = self.possible_agents.copy()
        
        # Define observation and action spaces
        self._setup_spaces()
        
    def _setup_spaces(self):
        # Reset environment to ensure we can get valid observations
        self.env.reset()
        
        # Get the first agent and their observation/action space
        if len(self.env.agents) > 0:
            first_agent = self.env.agents[0]
            
            # Get observation space from environment's observation_space
            if hasattr(self.env, 'observation_space'):
                obs_space = self.env.observation_space(first_agent)
                if hasattr(obs_space, 'shape'):
                    obs_shape = obs_space.shape
                else:
                    obs_shape = (457, 120, 3)  # Default PistonBall shape
            else:
                obs_shape = (457, 120, 3)  # Default PistonBall shape
        else:
            obs_shape = (457, 120, 3)  # Default PistonBall shape
        
        # Define observation space (default from PistonBall)
        piston_obs_space = Box(
            low=0, high=255, 
            shape=obs_shape, 
            dtype=np.uint8
        )
        
        # Define action space - PistonBall expects shape (1,) for continuous
        piston_action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Set spaces for all agents
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.possible_agents:
            self.observation_spaces[agent] = piston_obs_space
            self.action_spaces[agent] = piston_action_space


        # Add governance agent when enabled
        if self.enable_governance:
            self.possible_agents.append("governance_0")
            
            # Define governance observation space (global state)
            governance_obs_space = Box(
                low=-np.inf, high=np.inf, 
                shape=(1000,),  # governance model expects a flat observation of size 1000
                dtype=np.float32
            )
            
            # Define governance action space
            governance_action_space = Box(low=-1.0, high=1.0, shape=(self.n_pistons,), dtype=np.float32)
            
            # Add governance spaces
            self.observation_spaces["governance_0"] = governance_obs_space
            self.action_spaces["governance_0"] = governance_action_space
        
    def reset(self, *, seed=None, options=None):
        # Reset the environment
        self.env.reset(seed=seed, options=options)
        self.step_count = 0
        
        # Collect initial observations
        obs = {}
        
        # Use observe method if available
        if hasattr(self.env, 'observe'):
            for agent in self.env.agents:
                agent_obs = self.env.observe(agent)
                if agent_obs is not None:
                    obs[agent] = agent_obs.astype(np.uint8)
                else:
                    obs[agent] = np.zeros((457, 120, 3), dtype=np.uint8)
        else:
            # Fallback: create dummy observations
            for agent in self.env.agents:
                obs[agent] = np.zeros((457, 120, 3), dtype=np.uint8)
        
        # Add governance observation
        if self.enable_governance and "governance_0" in self.possible_agents:
            # Create global state observation for governance
            obs["governance_0"] = self._get_governance_observation()
        
        return obs, {}
    
    def _get_governance_observation(self):
        """Create global state observation for governance agent"""
        piston_states = [self.env.observe(agent).flatten() for agent in self.env.agents if agent != "governance_0"]
        
        # If all piston agents are done (end of episode), piston_states will be empty.
        if not piston_states:
            # Return a zero array of the correct shape.
            return np.zeros((1000,), dtype=np.float32)
            
        global_obs = np.concatenate(piston_states, axis=0)
        if global_obs.shape[0] > 1000: # Truncate to 1000 elements
            global_obs = global_obs[:1000]
        else:
            global_obs = np.pad(global_obs, (0, 1000 - global_obs.shape[0]))
        return global_obs.astype(np.float32)

    
    def step(self, actions):
        obs = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}
        
        # Convert actions to proper format for PettingZoo
        processed_actions = {}
        for agent, action in actions.items():
            if isinstance(action, (list, np.ndarray)): # handle list or numpy array actions
                if len(action) >= 1:
                    processed_actions[agent] = np.array([float(action[0])], dtype=np.float32) # use first element
                else:
                    processed_actions[agent] = np.array([0.0], dtype=np.float32) # default to 0.0 if empty
            else: # handle single float actions
                processed_actions[agent] = np.array([float(action)], dtype=np.float32)

        # --- Governance Logic ---
        # The governance agent's actions are applied *after* the piston agents have chosen their actions.
        # This works by clipping the piston's chosen action before it's executed in the environment.
        # The piston agent itself is not aware of the restriction when it makes its decision, but learns
        # from the consequences of the clipped action.
        if self.enable_governance and "governance_0" in actions:
            governance_actions = actions["governance_0"]
            
            # Use governance action to create action restrictions for each piston.
            # The i-th governance action corresponds to the i-th piston agent.
            for i, agent_id in enumerate(self._agent_ids):
                if agent_id in processed_actions:
                    # The i-th action from governance corresponds to the i-th piston
                    restriction = governance_actions[i]
                    original_action = processed_actions[agent_id]
                    
                    # Example: governance action determines allowed action range.
                    # A high restriction value leads to a smaller allowed action range.
                    max_allowed_action = 1.0 - (restriction * 0.5)  # reduce max action by up to 50%
                    
                    # The piston's original action is clipped to the new restricted range.
                    processed_actions[agent_id] = np.clip(original_action, -max_allowed_action, max_allowed_action)
        
        # Step through all agents using PettingZoo's iteration pattern
        agents_to_step = list(self.env.agents)
        agents_stepped = []
        
        for agent in self.env.agent_iter():
            agents_stepped.append(agent)
            observation, reward, termination, truncation, info = self.env.last()
            
            if termination or truncation:
                self.env.step(None)
            else:
                if agent in processed_actions:
                    action_to_send = processed_actions[agent]
                else:
                    action_to_send = np.array([0.0], dtype=np.float32)
                self.env.step(action_to_send)
            
            if observation is not None:
                obs[agent] = observation.astype(np.uint8)
            else:
                obs[agent] = np.zeros((457, 120, 3), dtype=np.uint8)

            rewards[agent] = float(reward) if reward is not None else 0.0
            terminated[agent] = bool(termination)
            truncated[agent] = bool(truncation)
            infos[agent] = info if info is not None else {}
        
            if len(agents_stepped) >= len(agents_to_step):
                break

        # Episode termination flags
        all_terminated = len(self.env.agents) == 0
        all_truncated = self.step_count >= self.max_cycles

        # Assign reward, observation, and termination status to governance agent
        if self.enable_governance:
            piston_rewards = [rewards.get(agent, 0.0) for agent in self._agent_ids if agent in rewards]
            rewards["governance_0"] = sum(piston_rewards)
            obs["governance_0"] = self._get_governance_observation()
            terminated["governance_0"] = all_terminated
            truncated["governance_0"] = all_truncated
        
        terminated['__all__'] = all_terminated
        truncated['__all__'] = all_truncated
        
        self.step_count += 1
        
        return obs, rewards, terminated, truncated, infos