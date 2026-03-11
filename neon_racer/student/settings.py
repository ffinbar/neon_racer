
"""
Shared configuration and constants for Neon Racer.
"""

# Physics Constants
BASE_LENGTH = 45
BASE_MAX_SPEED = 300.0
BASE_ACCEL = 200.0
BASE_FRICTION = 5.0
BASE_TURN_SPEED = 180.0
BASE_DRAG = 0.5
BASE_BRAKE_FORCE = 1.0
RAY_LENGTH = 400.0
RAY_STEP = 10.0

# Normalization Constants
MAX_DIST = RAY_LENGTH
MAX_SPEED_INPUT = 300.0

# General
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
FIXED_DT = 1.0 / FPS

# Training Constants
EVAL_SEEDS = 3  # Number of seeds to evaluate each genome on for dynamic tracks
MAX_FRAMES = 60 * 60 * 5  # Maximum frames per evaluation (5 minutes at 60 FPS)

# Dynamic track
LAP_TILES = 10 # How many tiles passed should constitute a lap