from pygame.math import Vector2
from . import settings

def get_relative_vector(pos, target, track):
    """
    Calculates vector from pos to target, handling toroidal wrapping if supported.
    target can be dict or Vector2.
    """
    tx = target['x'] if isinstance(target, dict) else target.x
    ty = target['y'] if isinstance(target, dict) else target.y
    
    dx = tx - pos.x
    dy = ty - pos.y
    if hasattr(track, 'width') and hasattr(track, 'height'):
        w, h = track.width, track.height
        if dx > w / 2: dx -= w
        elif dx < -w / 2: dx += w
        
        if dy > h / 2: dy -= h
        elif dy < -h / 2: dy += h
        
    return Vector2(dx, dy)


# ─────────────────────────────────────────────────────────────
# STUDENT ZONE: Perception (What the AI sees)
# ─────────────────────────────────────────────────────────────

def get_inputs(car, track):
    """
    Returns the standard 6 inputs for AI agents.
    [Ray1, Ray2, Ray3, Ray4, Ray5, Speed]
    All normalized to appropriate ranges.
    """
    inputs = []
    max_dist = settings.MAX_DIST
    max_speed_input = settings.MAX_SPEED_INPUT
    
    # 5 ray sensors (normalized 0-1)
    for s in car.rays:
        inputs.append(s[2] / max_dist)
    
    # Speed (normalized 0-1)
    inputs.append(car.speed / max_speed_input)
        
    return inputs

# ─────────────────────────────────────────────────────────────


def check_progress(car, track):
    """
    Updates car.checkpoint_idx, car.laps based on track gates.
    Returns reward for progress made this step.
    """
    reward = 0.0
    
    if track.check_checkpoint(car, car.checkpoint_idx):
        car.checkpoint_idx += 1
        reward += 1.0
    else:
        if car.checkpoint_idx >= len(track.nodes):
            if track.check_start_finish(car):
                car.laps += 1
                car.checkpoint_idx = 0
                reward += 50.0
                
    return reward


# ─────────────────────────────────────────────────────────────
# STUDENT ZONE: Reward Function
# ─────────────────────────────────────────────────────────────

def calculate_step_reward(car):
    """
    Calculates high-frequency rewards (called every frame).
    This is where we define what "good driving" looks like.
    """
    reward = 0.0
    
    speed_ratio = (car.speed / settings.MAX_SPEED_INPUT)
    reward += speed_ratio ** 3

    if speed_ratio < 0.5:
        reward -= 0.25  # Penalty for very low speed

    # Crash penalty
    if car.crashed:
        reward -= 10.0
    
    reward -= car.slip * 0.5

    return reward

# ─────────────────────────────────────────────────────────────
# TRANSLATION LAYER - Used by both NEAT and RL agents
# ─────────────────────────────────────────────────────────────

def translate_neural_output(raw_output):
    """
    Universal translation layer: converts raw neural network outputs to car controls.
    Works for any AI algorithm (NEAT, PPO, or future algorithms).
    
    Neural networks output values in range [-1, 1].
    We need to convert them to the car's expected ranges.
    
    Args:
        raw_output: Array-like with 3 values [steering, throttle, brake]
    
    Returns:
        dict with 'steering', 'throttle', 'brake' properly bounded
    """
    
    steering = raw_output[0]  # Left (-1) to Right (1)
    
    # convert from [-1, 1] to [0, 1]
    throttle = (raw_output[1] + 1.0) / 2.0  # No gas (0) to Full gas (1)
    
    # convert from [-1, 1] to boolean true / false
    brake = raw_output[2] > 0.5
    throttle = 0.0 if brake else throttle  # No throttle if braking
    
    return {'steering': steering, 'throttle': throttle, 'brake': brake}

# ─────────────────────────────────────────────────────────────


def check_stagnation(car, progress, current_time, current_score):
    """
    Checks if car has stalled (no progress for too long).
    Returns True if stagnated (should die/reset).
    """
    if not hasattr(car, 'last_score_time'):
        car.last_score_time = current_time
        car.last_max_score = current_score
        
    if current_score > car.last_max_score:
        car.last_max_score = current_score
        car.last_score_time = current_time
    
    if progress > 0.0:
        car.last_score_time = current_time
        
    if current_time - car.last_score_time > 5.0:
        return True
        
    return False


def perform_step(car, throttle, steering, brake, track, dt, current_time):
    """
    Unified step function for training loops.
    Returns (reward_delta, passed_checkpoint, stagnated)
    """
    car.update(dt, throttle, steering, brake, track)
    
    reward_progress = check_progress(car, track)
    reward_step = calculate_step_reward(car)

    passed_checkpoint = reward_progress > 0.0
    
    car.current_track_fitness = getattr(car, 'current_track_fitness', 0.0) + reward_progress + reward_step
    
    stagnated = check_stagnation(car, reward_progress, current_time, car.current_track_fitness)
    
    return reward_progress + reward_step, passed_checkpoint, stagnated
