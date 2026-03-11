from .physics import Car

class Racer:
    def __init__(self, agent, car_config, name=None, color=(255, 255, 255), type='manual'):
        self.agent = agent
        self.car = Car(car_config['x'], car_config['y'], car_config['angle'], car_config['scale'])
        self.name = name if name else agent.name
        self.color = color
        self.type = type
        self.lap = 0
        self.checkpoint_idx = 0
        self.current_lap_time = 0.0
        self.last_lap_time = 0.0
        self.best_lap_time = float('inf')
        self.total_checkpoints_passed = 0
        self.tile_checkpoints_since_lap = 0
        self.time_alive = 0.0
        self.finished = False
        self.eliminated = False
        self.low_speed_timer = 0.0
        self.splits = {}
        self.best_lap_splits = {}
        self.personal_best_splits = {}
        self.last_split_text = ""
        self.last_split_color = (255, 255, 255)
        self.split_display_timer = 0.0

    def reset_state(self, start_pos, start_angle, scale):
        self.car = Car(start_pos[0], start_pos[1], angle=start_angle, scale=scale)
        self.lap = 0
        self.checkpoint_idx = 0
        self.current_lap_time = 0.0
        self.last_lap_time = 0.0
        self.best_lap_time = float('inf')
        self.splits = {}
        self.best_lap_splits = {}
        self.personal_best_splits = {}
        self.last_split_text = ""
        self.split_display_timer = 0.0
        self.total_checkpoints_passed = 0
        self.tile_checkpoints_since_lap = 0
        self.time_alive = 0.0
        self.finished = False
        self.eliminated = False
        self.low_speed_timer = 0.0
