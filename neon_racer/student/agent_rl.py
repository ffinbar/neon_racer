from .agent import Agent
from . import environment
from stable_baselines3 import PPO
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
from gymnasium import spaces

import gymnasium as gym
class MockEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    def reset(self): return np.zeros(6, dtype=np.float32), {}
    def step(self, action): return np.zeros(6, dtype=np.float32), 0.0, False, False, {}
    def render(self): pass

class PPOAgent(Agent):
    def __init__(self, model_path='rl_model'):
        base_path = model_path.replace('.zip', '')
        check_path = base_path + ".zip"
        
        if not os.path.exists(check_path):
             print(f"Warning: RL Model {check_path} not found.")
             self.model = None
             self.norm_env = None
        else:
             print(f"Loading PPO Model from {model_path}...")
             self.model = PPO.load(model_path)
             self.name = f"PPO_{os.path.basename(model_path)}"
             vec_path = base_path + "_vecnormalize.pkl"
             if os.path.exists(vec_path):
                 print(f"Loading Normalization Stats from {vec_path}...")
                 dummy_env = DummyVecEnv([lambda: MockEnv()])
                 self.norm_env = VecNormalize.load(vec_path, dummy_env)
                 self.norm_env.training = False
                 self.norm_env.norm_reward = False
             else:
                 print("Warning: No normalization stats found. Inference may be incorrect.")
                 self.norm_env = None

    # ─────────────────────────────────────────────────────────────
    # STUDENT ZONE: PPO Output Translation
    # ─────────────────────────────────────────────────────────────
             
    def get_action(self, state):
        if self.model is None:
            return {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}

        inputs = np.array(state.get('neural_inputs', []), dtype=np.float32)
        if self.norm_env:
            inputs = self.norm_env.normalize_obs(inputs)
        
        # The PPO model produces 3 raw outputs
        action, _ = self.model.predict(inputs, deterministic=True)
        
        # Use the shared translation function to convert to car controls
        return environment.translate_neural_output(action)

    # ─────────────────────────────────────────────────────────────
