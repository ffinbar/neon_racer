import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from ..game_engine import GameEngine
from ..dynamic_track import DynamicTrack
from . import environment
import random

class NeonRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, track_name='default', enable_sound=True, mode='manual'):
        super().__init__()
        is_headless = (render_mode is None)
        should_enable_sound = enable_sound and (not is_headless)
        self.engine = GameEngine(mode=mode, track_name=track_name, headless=is_headless, enable_sound=should_enable_sound)
        self.engine.sim_speed = 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.clock = pygame.time.Clock()
        self.current_step = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        if hasattr(self.engine, 'track') and hasattr(self.engine.track, 'reset'):
             if isinstance(self.engine.track, DynamicTrack):
                 new_seed = random.randint(0, 1000000)
                 self.engine.track.reset(new_seed)
        
        self.engine.reset_game()
        self.current_step = 0
        self.max_cumulative_reward = -float('inf')
        self.cumulative_reward = 0.0
        self.engine.car.current_track_fitness = 0.0
        if hasattr(self.engine.car, 'last_score_time'):
            delattr(self.engine.car, 'last_score_time')
        if hasattr(self.engine.car, 'last_max_score'):
            delattr(self.engine.car, 'last_max_score')
        
        obs = self._get_obs()
        info = {}
        
        return obs, info

    def step(self, action):
        self.current_step += 1

        inputs = environment.translate_neural_output(action)
        
        dt = self.engine.fixed_dt
        current_time = self.current_step * dt
        
        reward_delta, passed_checkpoint, stagnated = environment.perform_step(
            self.engine.car, inputs['throttle'], inputs['steering'], inputs['brake'], 
            self.engine.track, dt, current_time
        )
        
        if hasattr(self.engine.track, 'update'):
            self.engine.track.update([self.engine.car])
        
        terminated = False
        truncated = False
        
        if self.engine.car.crashed:
            terminated = True
        
        if stagnated:
            terminated = True

        self.cumulative_reward += reward_delta
        if self.cumulative_reward > self.max_cumulative_reward:
            self.max_cumulative_reward = self.cumulative_reward

        obs = self._get_obs()
        info = {}
        
        return obs, reward_delta, terminated, truncated, info

    def _get_obs(self):
        inputs = environment.get_inputs(self.engine.car, self.engine.track)
        return np.array(inputs, dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            self.engine.render()
            pygame.display.flip()

    def close(self):
        pygame.quit()
