from abc import ABC, abstractmethod
import random
import pygame

class Agent(ABC):
    @abstractmethod
    def get_action(self, state):
        """
        state: dict containing sensors, speed, etc.
        returns: [steering, accel, brake]
        """
        pass
        
    @property
    def name(self):
        return getattr(self, '_name', 'Unknown')
        
    @name.setter
    def name(self, value):
        self._name = value

class ManualAgent(Agent):
    def __init__(self):
        self.name = "Manual"

    def get_action(self, state):
        steering = 0.0
        accel = 0.0
        brake = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            steering = -1.0
        elif keys[pygame.K_RIGHT]:
            steering = 1.0
        if keys[pygame.K_UP]:
            accel = 1.0
        elif keys[pygame.K_DOWN]:
            accel = -0.5
            
        if keys[pygame.K_SPACE]:
            brake = True
            
        # This agent returns a proper dict since we do not predict outputs from a neural network
        return {'steering': steering, 'throttle': accel, 'brake': brake}


# ─────────────────────────────────────────────────────────────
# STUDENT ZONE: RoverAgent
# ─────────────────────────────────────────────────────────────

class RoverAgent(Agent):
    """
    The Rover (Weeks 1-3)
    Rule-Based Logic using Raycast Sensors.
    """
    def __init__(self):
        self.name = "Rover"

    def get_action(self, state):
        steering = 0.0
        throttle = 1.0
        brake = False
        
        sensors = state['sensors']
        speed = state['speed']
        
        left, front_left, front, front_right, right = sensors

        # The amount of throttle is inversely proportional to the distance to the front wall
        if front < speed:
            throttle = 1.0 - (speed / front)
        
        # Use a weighted sum of all sensors to find the best path
        # We give higher priority to the 45-degree sensors (front_left, front_right) for general steering
        # and use far-side sensors (left, right) to help detect the exit of sharp curves.
        left_val = front_left * 1.0 + left * 0.5
        right_val = front_right * 1.0 + right * 0.5
        
        if right_val > left_val:
            steering = 1.0
        elif left_val > right_val:
            steering = -1.0
        
        return [steering, throttle, brake]

# ─────────────────────────────────────────────────────────────


class RandomAgent(Agent):
    def __init__(self):
         self.name = "Random"
         
    def get_action(self, state):
        return [
            # steering
            random.uniform(-1.0, 1.0),
            # throttle
            random.uniform(0.0, 1.0),
            # brake
            random.choice([True, False])
        ]
