import pygame
from pygame.math import Vector2
import sys
import os
from .physics import Car
from .student import environment
from .track import Track
from .dynamic_track import DynamicTrack
from .student.agent import ManualAgent, RandomAgent, RoverAgent
from .student.agent_neat import NeatAgent
from .student.agent_rl import PPOAgent
from .highscore_manager import HighscoreManager
from .racer import Racer
from .sound_manager import SoundManager
from .student.settings import SCREEN_WIDTH, SCREEN_HEIGHT, FIXED_DT
from .utils import angle_difference, generate_color_from_string, render_text_with_outline

class TireMarkSegment:
    def __init__(self, p1, p2, alpha, life=12.0):
        self.p1 = p1
        self.p2 = p2
        self.alpha = alpha
        self.max_alpha = alpha
        self.life = life
        self.max_life = life

    def update(self, dt):
        self.life -= dt
        self.alpha = (self.life / self.max_life) * self.max_alpha
        return self.life > 0

class GameEngine:
    def __init__(self, mode='manual', track_name='default', headless=False, agents_config=None, enable_sound=True):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        pygame.init()
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        if headless:
             pygame.display.set_caption("Neon-Racer (Headless)")
        else:
             pygame.display.set_caption("Neon-Racer")
        self.clock = pygame.time.Clock()
        self.running = True
        self.mode = mode
        self.highscore_manager = HighscoreManager(os.path.join(os.path.dirname(__file__), "student", "tracks"))
        
        self.cars = []
        if track_name == 'dynamic':
            self.track = DynamicTrack(self.width, self.height)
        else:
            track_path = os.path.join(os.path.dirname(__file__), "student", "tracks", f"{track_name}.json")
            self.track = Track(track_path)
        self.load_assets()
        assets_dir = os.path.join(os.path.dirname(__file__), "student", "assets")
        self.sound_manager = SoundManager(assets_dir, enabled=enable_sound)
        self.enable_sound = enable_sound
        self.user_muted = False
        self.agents_config = agents_config
        if not self.agents_config:
            self.agents_config = [{'type': mode, 'name': 'Player'}]
        self.racers = []
        for i, conf in enumerate(self.agents_config):
             agent = self.create_agent(conf)
             display_name = conf.get('name')
             if hasattr(agent, 'name') and agent.name:
                display_name = agent.name
             racer = Racer(agent, {'x':0, 'y':0, 'angle':0, 'scale':1.0}, name=display_name, type=conf.get('type'))
             self.racers.append(racer)
        self.agent = self.racers[0].agent 
        self.reset_game()
        self.current_inputs = [0.0, 0.0, False]
        self.current_checkpoint = 0
        self.current_lap = 0
        self.current_lap_time = 0.0
        self.last_lap_time = 0.0
        self.best_lap_time = float('inf')
        self.best_lap_splits = {}
        self.current_lap_splits = {}
        self.last_split_text = ""
        self.last_split_color = (255, 255, 255)
        self.split_display_timer = 0.0
        self.show_hud = True
        self.show_hud = True
        self.show_debug = False
        self.camera_locked = False
        self.camera_angle = 0.0
        self.camera_lock_init = True
        self.external_stats = {}
        self.paused = False
        self.single_step = False
        self.sim_speed = 1.0
        self.accumulator = 0.0
        self.fixed_dt = FIXED_DT
        self.tire_marks = []
        self.max_tire_marks = 2000
        self.tire_marks_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.last_dt = FIXED_DT
        
    def load_assets(self):
        assets_dir = os.path.join(os.path.dirname(__file__), "student", "assets")
        try:
            self.car_sprite = pygame.image.load(os.path.join(assets_dir, "car.png")).convert_alpha()
            self.car_brake_sprite = pygame.image.load(os.path.join(assets_dir, "car_brake.png")).convert_alpha()
            self.car_crashed_sprite = pygame.image.load(os.path.join(assets_dir, "car_crashed.png")).convert_alpha()
            self.car_mask_sprite = pygame.image.load(os.path.join(assets_dir, "mask.png")).convert_alpha()
            self.crown_sprite = pygame.image.load(os.path.join(assets_dir, "crown.png")).convert_alpha()
            
            scale = self.track.car_scale if hasattr(self, 'track') else 1.0
            sprite_scale = scale/10
            
            self.car_sprite = pygame.transform.rotozoom(self.car_sprite, 0, sprite_scale)
            self.car_brake_sprite = pygame.transform.rotozoom(self.car_brake_sprite, 0, sprite_scale)
            self.car_crashed_sprite = pygame.transform.rotozoom(self.car_crashed_sprite, 0, sprite_scale)
            self.car_mask_sprite = pygame.transform.rotozoom(self.car_mask_sprite, 0, sprite_scale)
            
            self.crown_sprite = pygame.transform.smoothscale(self.crown_sprite, (24, 24))
            
            if not hasattr(self, 'font'):
                self.font = pygame.font.SysFont("Consolas", 24)
                if not self.font:
                     self.font = pygame.font.SysFont("Courier New", 24)
                if not self.font:
                     self.font = pygame.font.SysFont("Courier", 24)
                if not self.font:
                     self.font = pygame.font.SysFont(None, 24)
        except Exception as e:
            print(f"Error loading assets: {e}")
            sys.exit()

    def get_focus_car(self):
        """Return the car object used as camera focus (manual focus, manual agent, or first racer)."""
        if hasattr(self, 'manual_focus_racer') and self.manual_focus_racer:
            if not self.manual_focus_racer.car.crashed:
                return self.manual_focus_racer.car
            else:
                self.manual_focus_racer = None
        manual_racer = next((r for r in self.racers if getattr(r, 'type', None) == 'manual'), None)
        if manual_racer:
            return manual_racer.car
        if self.racers:
            return self.racers[0].car
        return None

    def world_to_screen(self, world_pos):
        """Convert a world position (Vector2 or tuple) to screen coordinates accounting for camera lock and zoom."""
        if getattr(self, 'camera_locked', False):
            focus_car = self.get_focus_car()
            if focus_car is None:
                return Vector2(world_pos)
            surf_angle = self.camera_angle + 90
            vector_angle = -surf_angle
            cam_zoom = getattr(self.track, 'cam_zoom', 1.0)
            screen_center = Vector2(self.width // 2, self.height // 2)
            rel = Vector2(world_pos) - focus_car.pos
            rot = rel.rotate(vector_angle)
            return screen_center + (rot * cam_zoom)
        else:
            return Vector2(world_pos)

    def change_track(self, track_name, seed=None):
        """Switches the current track, reloads data, and re-scales assets."""
        if track_name == 'dynamic':
            if isinstance(self.track, DynamicTrack):
                self.track.reset(seed)
            else:
                self.track = DynamicTrack(self.width, self.height)
                self.track.reset(seed)
        else:
            track_path = os.path.join(os.path.dirname(__file__), "student", "tracks", f"{track_name}.json")
            if not os.path.exists(track_path):
                print(f"Error: Track file {track_path} not found.")
                return False
            self.track = Track(track_path)
        self.load_assets()
        self.highscore_manager.load_highscores(track_name)
        self.reset_game()
        return True

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    clicked_racer = None
                    min_dist = 50
                    for r in self.racers:
                        if not r.car.crashed:
                            car_screen_pos = self.world_to_screen(r.car.pos)
                            dist = car_screen_pos.distance_to(Vector2(mx, my))
                            if dist < min_dist:
                                min_dist = dist
                                clicked_racer = r
                    
                    self.manual_focus_racer = clicked_racer
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    if self.paused:
                        self.sound_manager.stop_all()
                elif event.key == pygame.K_c:
                    self.camera_locked = not getattr(self, 'camera_locked', False)
                    self.camera_lock_init = True
                elif event.key == pygame.K_PERIOD:
                    self.single_step = True
                    self.paused = True
                elif event.key == pygame.K_LEFTBRACKET:
                    if self.sim_speed > 0.15:
                        self.sim_speed = max(0.1, round(self.sim_speed - 0.1, 1))
                    else:
                        self.sim_speed = max(0.01, round(self.sim_speed - 0.01, 2))
                elif event.key == pygame.K_RIGHTBRACKET:
                    if self.sim_speed < 0.1:
                        self.sim_speed = min(0.1, round(self.sim_speed + 0.01, 2))
                    else:
                        self.sim_speed = min(5.0, round(self.sim_speed + 0.1, 1))
                elif event.key == pygame.K_m:
                    self.set_sound(not self.enable_sound)
                    
    def set_sound(self, enabled, manual=False):
        """Enable/disable sound. If manual=True this was triggered by the user (toggle)
        and will update the user_muted flag. Automatic callers should respect user_muted."""
        if manual:
            self.user_muted = not enabled
        if self.user_muted and enabled:
            return
        self.enable_sound = enabled
        self.sound_manager.enabled = enabled
        if not enabled:
            self.sound_manager.stop_all()

    def create_agent(self, config):
        """Helper to create agent from config dict."""
        atype = config.get('type', 'manual')
        if atype == 'manual':
            return ManualAgent()
        elif atype == 'random':
            return RandomAgent()
        elif atype == 'rover':
            return RoverAgent()
        elif atype == 'neat':
            path = config.get('path', 'best_genome.pkl')
            if path.endswith('.pkl'):
                path = path[:-4]
            if not path.startswith('genomes/') and not os.path.isabs(path):
                genome_path = os.path.join('genomes', path)
                if os.path.exists(genome_path + '.pkl'):
                    path = genome_path
            if not path.endswith('.pkl'):
                path = path + '.pkl'
            return NeatAgent('config-feedforward.txt', path)
        elif atype == 'rl':
            path = config.get('path', 'rl_model')
            if path.endswith('.zip'):
                path = path[:-4]
            if not path.startswith('models/') and not os.path.isabs(path):
                model_path = os.path.join('models', path)
                if os.path.exists(model_path + '.zip'):
                    path = model_path
            return PPOAgent(path)
        return ManualAgent()

    def generate_racer_sprite(self, racer, use_existing_color=False):
        if not use_existing_color:
            racer.color = generate_color_from_string(racer.name, saturation=90, value=100)
            
        base = self.car_sprite.copy()
        paint_surf = pygame.Surface(base.get_size(), pygame.SRCALPHA)
        paint_surf.fill(racer.color)
        paint_surf.blit(self.car_mask_sprite, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        base.blit(paint_surf, (0, 0))
        
        racer.cached_sprite = base


    def reset_game(self):
        """Resets the game state and returns the initial state dict."""
        if hasattr(self.track, 'reset'):
             self.track.reset()
        start_pos = (100, 100)
        start_angle = 0
        if self.track.start_line:
            start_pos = (self.track.start_line['x'], self.track.start_line['y'])
            start_angle = self.track.start_line.get('angle', 0)
        
        scale = self.track.car_scale if self.track else 1.0
        for i, racer in enumerate(self.racers):
            racer.reset_state(start_pos, start_angle, scale)
            racer.is_braking = False
            self.generate_racer_sprite(racer)
            pass
            
        self.cars = [r.car for r in self.racers]
        self.car = self.racers[0].car
        self.manual_focus_racer = None
        
        self.current_checkpoint = 0
        self.current_lap = 0
        
        self.tire_marks = []
        track_name = 'dynamic' if isinstance(self.track, DynamicTrack) else os.path.splitext(os.path.basename(self.track.file_path))[0]
        data = self.highscore_manager.load_highscores(track_name)
        for racer in self.racers:
             racer.personal_best = data['agent_bests'].get(racer.agent.name)
             if racer.personal_best and 'metrics' in racer.personal_best:
                 racer.personal_best_splits = racer.personal_best['metrics']
             
             if racer.personal_best and 'score' in racer.personal_best:
                 racer.best_lap_time = racer.personal_best['score']
        
        self.global_best = data['global_best']
        self.is_dynamic = isinstance(self.track, DynamicTrack)
        for car in self.cars:
            car.cast_rays(self.track)
        
        return self.get_state()


    def _advance_game(self, dt):
        """
        Internal method to advance physics and game logic (checkpoints/laps).
        Returns reward for primary racer (index 0).
        """
        primary_reward = 0.0
        active_racers = [r for r in self.racers if not r.eliminated]
        leader = max(active_racers, key=lambda r: r.total_checkpoints_passed) if active_racers else None
        
        for racer in active_racers:
            if racer == self.racers[0] and self.mode in ['manual', 'train_rl', 'train_neat']:
                steering, throttle, brake = self.current_inputs
            else:
                r_state = {
                    'sensors': [ray[2] for ray in racer.car.rays],
                    'speed': racer.car.speed,
                    'pos': racer.car.pos,
                    'heading': racer.car.heading,
                    'checkpoint_idx': racer.checkpoint_idx,
                    'track_nodes': self.track.nodes,
                    'neural_inputs': environment.get_inputs(racer.car, self.track)
                }
                decision = racer.agent.get_action(r_state)
                steering, throttle, brake = decision['steering'], decision['throttle'], decision['brake']
            racer.car.update(dt, throttle, steering, brake, self.track)
            try:
                racer.is_braking = bool(brake) if isinstance(brake, bool) else (float(brake) > 0.5)
            except Exception:
                racer.is_braking = bool(brake)
            if not racer.car.crashed:
                racer.current_lap_time += dt
                if racer.split_display_timer > 0:
                    racer.split_display_timer -= dt
                racer.time_alive += dt
                if isinstance(self.track, DynamicTrack):
                    if racer.car.speed < 10.0:
                        racer.low_speed_timer += dt
                    else:
                        racer.low_speed_timer = 0.0
                    
                    if racer.low_speed_timer > 5.0:
                        racer.car.crashed = True
            r_reward = 0.0
            racer.car.checkpoint_idx = racer.checkpoint_idx
            
            if not racer.car.crashed:
                if isinstance(self.track, DynamicTrack):
                    if self.track.check_checkpoint(racer.car, None):
                        racer.total_checkpoints_passed += 1
                        racer.checkpoint_idx += 1
                        racer.tile_checkpoints_since_lap = getattr(racer, 'tile_checkpoints_since_lap', 0) + 1
                        r_reward += 1.0
                        if racer.tile_checkpoints_since_lap >= getattr(self.track, 'lap_tiles', 10):
                            racer.lap += 1
                            racer.tile_checkpoints_since_lap = 0
                            racer.checkpoint_idx = 0
                            r_reward += 100.0
                            racer.last_lap_time = racer.current_lap_time
                            if racer.last_lap_time < racer.best_lap_time:
                                racer.best_lap_time = racer.last_lap_time
                            racer.current_lap_time = 0.0
                            racer.splits = {}
                elif racer.checkpoint_idx < len(self.track.nodes):
                    if self.track.check_checkpoint(racer.car, racer.checkpoint_idx):
                        racer.splits[racer.checkpoint_idx] = racer.current_lap_time
                        ref_time = None
                        if racer.checkpoint_idx in racer.best_lap_splits:
                            ref_time = racer.best_lap_splits[racer.checkpoint_idx]
                        elif str(racer.checkpoint_idx) in racer.personal_best_splits:
                            ref_time = racer.personal_best_splits[str(racer.checkpoint_idx)]
                            
                        if ref_time:
                            delta = racer.current_lap_time - float(ref_time)
                            sign = "+" if delta > 0 else "-"
                            racer.last_split_text = f"{sign}{abs(delta):.2f}"
                            racer.last_split_color = (255, 0, 0) if delta > 0 else (0, 255, 0)
                            racer.split_display_timer = 3.0
                            
                        racer.checkpoint_idx += 1
                        racer.total_checkpoints_passed += 1
                        r_reward += 1.0
                else:
                    if self.track.check_start_finish(racer.car):
                         racer.lap += 1
                         racer.checkpoint_idx = 0
                         r_reward += 100.0
                         
                         racer.last_lap_time = racer.current_lap_time
                         if racer.last_lap_time < racer.best_lap_time:
                             racer.best_lap_time = racer.last_lap_time
                             racer.best_lap_splits = racer.splits.copy()
                         if not isinstance(self.track, DynamicTrack):
                             is_new_global, _ = self.highscore_manager.update_highscore(
                                 os.path.splitext(os.path.basename(self.track.file_path))[0],
                                 racer.agent.name,
                                 racer.last_lap_time,
                                 racer.splits.copy(),
                                 mode='time_trial'
                             )
                             if is_new_global:
                                 self.global_best = {
                                     'score': racer.last_lap_time,
                                     'agent': racer.agent.name
                                 }
                             
                         racer.current_lap_time = 0.0
                         racer.splits = {}
            else:
                r_reward -= 10.0
                if isinstance(self.track, DynamicTrack) and not racer.finished:
                     is_new_global, _ = self.highscore_manager.update_highscore(
                         'dynamic', racer.agent.name, racer.time_alive, {'checkpoints': racer.total_checkpoints_passed}, mode='survival'
                     )
                     if is_new_global:
                         self.global_best = {
                             'score': racer.time_alive,
                             'agent': racer.agent.name
                         }
                     racer.finished = True
            r_reward += environment.calculate_step_reward(racer.car)
            
            if racer == self.racers[0]:
                primary_reward = r_reward
        
        self.last_dt = dt
        if hasattr(self.track, 'update') and self.racers:
            active_cars = [r.car for r in self.racers if not getattr(r, 'eliminated', False)]
            if active_cars:
                self.track.update(active_cars)
            
        return primary_reward

    def update(self, dt):
        self._advance_game(dt)
        state = self.get_state()
        action = self.agent.get_action(state)
        if isinstance(action, dict):
            steering = action.get('steering', 0.0)
            throttle = action.get('throttle', 0.0)
            brake = action.get('brake', False)
        else:
            steering, throttle, brake = action
        self.current_inputs = [steering, throttle, brake]

    def render(self):
        dt = getattr(self, 'last_dt', FIXED_DT)
        if not self.paused:
             for racer in self.racers:
                 is_eliminated = getattr(racer, 'eliminated', False)
                 self.sound_manager.update_racer(
                     id(racer),
                     racer.car.speed,
                     racer.car.slip,
                     is_alive=(not is_eliminated),
                     is_crashed=racer.car.crashed,
                     dt=dt
                 )
        
        for racer in self.racers:
            car_obj = racer.car
            if not car_obj.crashed:
                corners = car_obj.get_corners()
                bl, br = corners[2], corners[3]
                
                if car_obj.slip > 0.05:
                    alpha = int(min(255, car_obj.slip * 600))
                    if car_obj.prev_back_left and car_obj.prev_back_right:
                        if bl.distance_to(car_obj.prev_back_left) < 50:
                            self.tire_marks.append(TireMarkSegment(car_obj.prev_back_left, bl, alpha))
                            self.tire_marks.append(TireMarkSegment(car_obj.prev_back_right, br, alpha))
                    while len(self.tire_marks) > self.max_tire_marks:
                        self.tire_marks.pop(0)
                        
                car_obj.prev_back_left = Vector2(bl.x, bl.y)
                car_obj.prev_back_right = Vector2(br.x, br.y)
            else:
                car_obj.prev_back_left = None
                car_obj.prev_back_right = None

        self.tire_marks = [tm for tm in self.tire_marks if tm.update(dt)]

        self.screen.set_clip(None)
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.width, self.height))
        
        camera_locked = getattr(self, 'camera_locked', False)
        
        focus_racer = None
        if hasattr(self, 'manual_focus_racer') and self.manual_focus_racer:
             focus_racer = self.manual_focus_racer
        else:
             focus_racer = next((r for r in self.racers if getattr(r, 'type', None) == 'manual'), None)
             if not focus_racer and self.racers:
                 focus_racer = self.racers[0]
        
        focus_car = focus_racer.car if focus_racer else None
        
        if isinstance(self.track, DynamicTrack):
             if hasattr(self.track, 'render_update'):
                 self.track.render_update()
                 
        if camera_locked and focus_car:

            target_angle = focus_car.heading
            
            if getattr(self, 'camera_lock_init', False):
                self.camera_angle = target_angle
                self.camera_lock_init = False
                
            diff = angle_difference(self.camera_angle, target_angle)
            
            dt = getattr(self, 'last_dt', FIXED_DT)
            lerp_speed = 5.0
            
            self.camera_angle += diff * min(1.0, lerp_speed * dt)
            self.camera_angle %= 360
            
            surf_angle = self.camera_angle + 90
            vector_angle = -surf_angle
            
            cam_zoom = getattr(self.track, 'cam_zoom', 1.0)
            
            screen_center = Vector2(self.width // 2, self.height // 2)
            
            def to_screen(world_pos):
                rel = Vector2(world_pos) - focus_car.pos
                rot = rel.rotate(vector_angle)
                return screen_center + (rot * cam_zoom)
            
            bg_surf = None
            if isinstance(self.track, DynamicTrack):
                bg_surf = self.track.master_surface
            elif self.track.image:
                bg_surf = self.track.image
            
            track_offsets = [(0, 0)]
            if isinstance(self.track, DynamicTrack):
                tw, th = self.width, self.height
                track_offsets = [
                    (0, 0), (tw, 0), (-tw, 0),
                    (0, th), (0, -th),
                    (tw, th), (tw, -th), (-tw, th), (-tw, -th)
                ]

            if bg_surf:
                rotated_bg = pygame.transform.rotozoom(bg_surf, surf_angle, cam_zoom)
                
                bg_rect = bg_surf.get_rect()
                base_cx, base_cy = bg_rect.centerx, bg_rect.centery
                base_cx, base_cy = bg_rect.centerx, bg_rect.centery
                
                for ox, oy in track_offsets:
                    tile_center_world = Vector2(base_cx + ox, base_cy + oy)
                    
                    tile_center_screen = to_screen(tile_center_world)
                    
                    rot_rect = rotated_bg.get_rect(center=(int(tile_center_screen.x), int(tile_center_screen.y)))
                    
                    if self.screen.get_rect().colliderect(rot_rect):
                        self.screen.blit(rotated_bg, rot_rect)
                
            if self.tire_marks:
                self.tire_marks_surf.fill((0, 0, 0, 0))
                for tm in self.tire_marks:
                    if tm.alpha < 3: continue
                    color = (10, 10, 10, int(tm.alpha))
                    pygame.draw.line(self.tire_marks_surf, color, tm.p1, tm.p2, 3)
                
                rotated_marks = pygame.transform.rotozoom(self.tire_marks_surf, surf_angle, cam_zoom)
                
                tm_rect = self.tire_marks_surf.get_rect()
                base_tm_cx, base_tm_cy = tm_rect.centerx, tm_rect.centery
                
                for ox, oy in track_offsets:
                     tile_center_world = Vector2(base_tm_cx + ox, base_tm_cy + oy)
                     tile_center_screen = to_screen(tile_center_world)
                     
                     rot_tm_rect = rotated_marks.get_rect(center=(int(tile_center_screen.x), int(tile_center_screen.y)))
                     
                     if self.screen.get_rect().colliderect(rot_tm_rect):
                         self.screen.blit(rotated_marks, rot_tm_rect)

            if self.track.start_line:
                start_node = self.track.start_line
                sl_width = start_node.get('width', 100)
                sl_height = 20
                sl_surf = pygame.Surface((sl_width, sl_height), pygame.SRCALPHA)
                sq_size = 10
                rows = int(sl_height / sq_size)
                cols = int(sl_width / sq_size)
                for c in range(cols + 1):
                    for r in range(rows):
                        x = c * sq_size
                        y = r * sq_size
                        if x >= sl_width: continue
                        is_black = (c + r) % 2 == 1
                        color_rect = (0, 0, 0) if is_black else (255, 255, 255)
                        pygame.draw.rect(sl_surf, color_rect, (x, y, sq_size, sq_size))
                
                track_angle = start_node.get('angle', 0)
                base_angle = -track_angle - 90
                final_angle = base_angle + surf_angle
                
                rotated_sl = pygame.transform.rotozoom(sl_surf, final_angle, cam_zoom)
                
                base_sl_pos = Vector2(start_node['x'], start_node['y'])
                
                for ox, oy in track_offsets:
                    sl_world_pos = base_sl_pos + Vector2(ox, oy)
                    sl_screen_pos = to_screen(sl_world_pos)
                    
                    sl_rect = rotated_sl.get_rect(center=(int(sl_screen_pos.x), int(sl_screen_pos.y)))
                    if self.screen.get_rect().colliderect(sl_rect):
                         self.screen.blit(rotated_sl, sl_rect)

            if self.show_debug:
                debug_car_disp = self.get_focus_car() or (self.racers[0].car if self.racers else None)

                if debug_car_disp:
                    for start, end, dist in debug_car_disp.rays:
                        color = (0, 255, 0)
                        if dist < 50: color = (255, 0, 0)
                        s = to_screen(start)
                        e = to_screen(end)
                        pygame.draw.line(self.screen, color, s, e, 1)
                    
                for node in self.track.nodes:
                    base_np = Vector2(node['x'], node['y'])
                    for ox, oy in track_offsets:
                        np_world = base_np + Vector2(ox, oy)
                        np = to_screen(np_world)
                        if -50 < np.x < self.width + 50 and -50 < np.y < self.height + 50:
                            pygame.draw.circle(self.screen, (0, 0, 255), (int(np.x), int(np.y)), int(5 * cam_zoom))

            leader = max(self.racers, key=lambda r: (r.lap, r.total_checkpoints_passed)) if self.racers else None
            
            manual_racer = next((r for r in self.racers if getattr(r, 'type', None) == 'manual'), None)
            
            draw_list = list(self.racers)
            if manual_racer:
                draw_list.remove(manual_racer)
                draw_list.append(manual_racer)
            
            for racer in draw_list:
                sprite = getattr(racer, 'cached_sprite', self.car_sprite)
                if racer.car.crashed:
                    sprite = self.car_crashed_sprite
                    sprite.set_alpha(100)
                else:
                    sprite.set_alpha(255)
                
                car_rot = -racer.car.heading + surf_angle
                rotated_car = pygame.transform.rotozoom(sprite, car_rot, cam_zoom)
                
                base_r_pos = racer.car.pos
                
                for ox, oy in track_offsets:
                    r_pos_world = base_r_pos + Vector2(ox, oy)
                    r_pos = to_screen(r_pos_world)
                    
                    if not (-100 < r_pos.x < self.width + 100 and -100 < r_pos.y < self.height + 100):
                        continue
                        
                    rect = rotated_car.get_rect(center=(int(r_pos.x), int(r_pos.y)))
                    self.screen.blit(rotated_car, rect)
                    
                    if not racer.car.crashed and getattr(racer, 'is_braking', False) and ox == 0 and oy == 0:
                        brake_sprite = self.car_brake_sprite
                        rotated_brake = pygame.transform.rotozoom(brake_sprite, car_rot, cam_zoom)
                        b_rect = rotated_brake.get_rect(center=(int(r_pos.x), int(r_pos.y)))
                        self.screen.blit(rotated_brake, b_rect)
                     
                    if self.show_debug and isinstance(self.track, DynamicTrack) and not racer.car.crashed:
                        current_timer = getattr(racer, 'low_speed_timer', 0.0)
                        if current_timer > 0:
                            max_time = 5.0
                            ratio = max(0.0, max_time - current_timer) / max_time
                            bar_w = 40
                            bar_h = 6
                            x = r_pos.x - bar_w // 2
                            y = r_pos.y - 55 
                            pygame.draw.rect(self.screen, (255, 0, 0), (x, y, bar_w, bar_h))
                            if ratio > 0:
                                 pygame.draw.rect(self.screen, (0, 255, 0), (x, y, bar_w * ratio, bar_h))

                    if not racer.car.crashed and self.show_hud:
                        label = render_text_with_outline(self.font, racer.name, racer.color, outline_color=(0,0,0), outline_width=2, aa=True)
                        l_pos_x = r_pos.x - label.get_width() // 2
                        l_pos_y = r_pos.y - 40
                        
                        if (racer == leader) and len(self.racers) > 1:
                            cw = self.crown_sprite.get_width()
                            self.screen.blit(self.crown_sprite, (l_pos_x - cw - 5, l_pos_y))
                             
                        self.screen.blit(label, (l_pos_x, l_pos_y))
                        
                        if racer.split_display_timer > 0:
                             stext = render_text_with_outline(self.font, racer.last_split_text, racer.last_split_color, outline_color=(0,0,0), outline_width=2, aa=True)
                             self.screen.blit(stext, (r_pos.x + 20, r_pos.y - 20))

        else:
            if isinstance(self.track, DynamicTrack):
                 self.screen.fill((20, 20, 20))
                 self.screen.blit(self.track.master_surface, (0, 0))
            elif self.track.image:
                 self.screen.blit(self.track.image, (0, 0))
                 
            if self.tire_marks:
                self.tire_marks_surf.fill((0, 0, 0, 0))
                width = 3
                for tm in self.tire_marks:
                    if tm.alpha < 3: continue
                    color = (10, 10, 10, int(tm.alpha))
                    pygame.draw.line(self.tire_marks_surf, color, tm.p1, tm.p2, width)
                self.screen.blit(self.tire_marks_surf, (0, 0))
                
            if self.track.start_line:
                start_node = self.track.start_line
                sl_width = start_node.get('width', 100)
                sl_height = 20
                sl_surf = pygame.Surface((sl_width, sl_height), pygame.SRCALPHA)
                sq_size = 10
                rows = int(sl_height / sq_size)
                cols = int(sl_width / sq_size)
                for c in range(cols + 1):
                    for r in range(rows):
                        x = c * sq_size
                        y = r * sq_size
                        if x >= sl_width: continue
                        is_black = (c + r) % 2 == 1
                        color = (0, 0, 0) if is_black else (255, 255, 255)
                        pygame.draw.rect(sl_surf, color, (x, y, sq_size, sq_size))
                track_angle = start_node.get('angle', 0)
                final_angle = -track_angle - 90
                rotated_sl = pygame.transform.rotate(sl_surf, final_angle)
                rect = rotated_sl.get_rect(center=(start_node['x'], start_node['y']))
                self.screen.blit(rotated_sl, rect)
            
            leader = max(self.racers, key=lambda r: (r.lap, r.total_checkpoints_passed)) if self.racers else None

            if self.show_debug:
                 debug_car = self.get_focus_car() or (self.racers[0].car if self.racers else None)
                 
                 if debug_car:
                     for start, end, dist in debug_car.rays:
                         color = (0, 255, 0)
                         if dist < 50: color = (255, 0, 0)
                         pygame.draw.line(self.screen, color, start, end, 1)
                 for i, node in enumerate(self.track.nodes):
                     pygame.draw.circle(self.screen, (0, 0, 255), (int(node['x']), int(node['y'])), 5)
                 if isinstance(self.track, DynamicTrack):
                     self.track.render_debug(self.screen, debug_car)
            
            for racer in self.racers:
                sprite = getattr(racer, 'cached_sprite', self.car_sprite)
                
                if racer.car.crashed:
                    sprite = self.car_crashed_sprite
                    sprite.set_alpha(100)
                else:
                    sprite.set_alpha(255)
                rotated_car = pygame.transform.rotate(sprite, -racer.car.heading)
                rect = rotated_car.get_rect(center=(racer.car.pos.x, racer.car.pos.y))
                self.screen.blit(rotated_car, rect)
                if not racer.car.crashed and getattr(racer, 'is_braking', False):
                     brake_sprite = self.car_brake_sprite
                     rotated_brake = pygame.transform.rotate(brake_sprite, -racer.car.heading)
                     b_rect = rotated_brake.get_rect(center=(racer.car.pos.x, racer.car.pos.y))
                     self.screen.blit(rotated_brake, b_rect)
                if self.show_debug and isinstance(self.track, DynamicTrack) and not racer.car.crashed:
                    max_time = 5.0
                    current_timer = getattr(racer, 'low_speed_timer', 0.0)
                    if current_timer > 0:
                        remaining = max(0.0, max_time - current_timer)
                        ratio = remaining / max_time
                        
                        bar_w = 40
                        bar_h = 6
                        x = racer.car.pos.x - bar_w // 2
                        y = racer.car.pos.y - 55 
                        pygame.draw.rect(self.screen, (255, 0, 0), (x, y, bar_w, bar_h))
                        if ratio > 0:
                             pygame.draw.rect(self.screen, (0, 255, 0), (x, y, bar_w * ratio, bar_h))
                if not racer.car.crashed and self.show_hud:
    
                    label = render_text_with_outline(self.font, racer.name, racer.color, outline_color=(0,0,0), outline_width=2, aa=True)
                    if racer == leader:
                         cw = self.crown_sprite.get_width()
                         lx = racer.car.pos.x - label.get_width()//2
                         ly = racer.car.pos.y - 40
                         self.screen.blit(self.crown_sprite, (lx - cw - 5, ly))
                         self.screen.blit(label, (lx, ly))
                    else:
                         self.screen.blit(label, (racer.car.pos.x - label.get_width()//2, racer.car.pos.y - 40))
                    
                    if racer.split_display_timer > 0:
                         stext = render_text_with_outline(self.font, racer.last_split_text, racer.last_split_color, outline_color=(0,0,0), outline_width=2, aa=False)
                         self.screen.blit(stext, (racer.car.pos.x + 20, racer.car.pos.y - 20))
        if self.show_hud:
            lb_h = 180
            if self.racers:
                lb_h = 60 + len(self.racers[:5]) * 25
            
            s = pygame.Surface((450, lb_h))
            s.set_alpha(0)
            s.fill((0, 0, 0))
            self.screen.blit(s, (0, 0))
            
            stats_h = 160
            s2 = pygame.Surface((450, stats_h))
            s2.set_alpha(0)
            s2.fill((0, 0, 0))
            self.screen.blit(s2, (0, self.height - stats_h))

            cam_text = "CAM: LOCK" if getattr(self, 'camera_locked', False) else "CAM: MAP"
            c_surf = render_text_with_outline(self.font, cam_text, (0, 255, 255), outline_color=(0,0,0), outline_width=2, aa=False)
            self.screen.blit(c_surf, (self.width - c_surf.get_width() - 10, 10))
            
            sim_speed_text = f"Sim Speed: {self.sim_speed:.2f}x"
            col = (0, 255, 0) if self.sim_speed == 1.0 else (255, 165, 0)
            spd_surf = render_text_with_outline(self.font, sim_speed_text, col, outline_color=(0,0,0), outline_width=2, aa=False)
            self.screen.blit(spd_surf, (self.width - spd_surf.get_width() - 10, 40))

            sorted_racers = sorted(self.racers, key=lambda r: (r.lap, r.total_checkpoints_passed), reverse=True)
            
            y_off = 10
            lb_surf = render_text_with_outline(self.font, "Leaderboard:", (255, 255, 0), outline_color=(0,0,0), outline_width=2, aa=False)
            self.screen.blit(lb_surf, (10, y_off))
            
            y_off += 30
            
            for i, r in enumerate(sorted_racers[:5]):
                valid = not r.car.crashed
                status = "" if valid else "(C)"
                color = r.color if valid else (100, 100, 100)

                cp_count = getattr(r, 'total_checkpoints_passed', None)
                if cp_count is None:
                    cp_count = getattr(r.car, 'checkpoint_idx', 0)

                last_tile = getattr(r.car, 'last_tile', None)
                last_tile_str = f" T:{last_tile}" if last_tile is not None else ""

                suffix = ""
                if self.mode == 'train_neat':
                    suffix = f" | Fitness: {max(0, r.car.current_track_fitness):.0f}"
                elif self.mode == 'train_rl':
                    suffix = f" | Reward: {r.car.current_track_fitness:.2f}"

                text = f"{i+1}. {r.name} {status} - L{r.lap} CP:{cp_count}{last_tile_str}{suffix}"
                
                if self.global_best and self.global_best.get('agent') == r.name and not self.is_dynamic:
                     if r.best_lap_time == self.global_best.get('score'):
                         color = (255, 100, 255)

                entry_surf = render_text_with_outline(self.font, text, color, outline_color=(0,0,0), outline_width=1, aa=False)
                self.screen.blit(entry_surf, (10, y_off))
                y_off += 25
            is_multi_agent = len(self.racers) > 1
            if is_multi_agent and self.global_best:
                gb_score = self.global_best.get('score', 0)
                gb_agent = self.global_best.get('agent', 'Unknown')
                gb_text = f"Global Best: {gb_score:.2f} ({gb_agent})"
                gb_surf = render_text_with_outline(self.font, gb_text, (255, 100, 255), outline_color=(0,0,0), outline_width=1, aa=False)
                self.screen.blit(gb_surf, (10, y_off + 5))
                y_off += 30
            p = next((r for r in self.racers if r.car == self.car), self.racers[0])
            
            y_bot = self.height - 150
            
            if not is_multi_agent:
                spd_out = render_text_with_outline(self.font, f"Speed: {p.car.speed:.1f}", (255, 255, 255), outline_color=(0,0,0), outline_width=1, aa=False)
                self.screen.blit(spd_out, (10, y_bot))
                rew_out = render_text_with_outline(self.font, f"Reward: {p.car.current_track_fitness:.2f}", (200, 200, 200), outline_color=(0,0,0), outline_width=1, aa=False)
                self.screen.blit(rew_out, (10, y_bot + 30))
                cp_idx = getattr(p.car, 'checkpoint_idx', None)
                last_tile = getattr(p.car, 'last_tile', None)
                cp_text = f"CP: {cp_idx}" if cp_idx is not None else (f"Tile: {last_tile}" if last_tile is not None else "")
                if cp_text:
                    cp_out = render_text_with_outline(self.font, cp_text, (255, 215, 0), outline_color=(0,0,0), outline_width=1, aa=False)
                    self.screen.blit(cp_out, (10, y_bot + 60))
                else:
                    best_out = render_text_with_outline(self.font, f"Best Lap: {p.best_lap_time:.2f}", (255, 215, 0), outline_color=(0,0,0), outline_width=1, aa=False)
                    self.screen.blit(best_out, (10, y_bot + 60))
            
            y_extra = 90
            if not is_multi_agent and self.global_best:
                gb_score = self.global_best.get('score', 0)
                gb_agent = self.global_best.get('agent', 'Unknown')
                gb_text = f"Global Best: {gb_score:.2f} ({gb_agent})"
                gb_out = render_text_with_outline(self.font, gb_text, (255, 100, 255), outline_color=(0,0,0), outline_width=1, aa=False)
                self.screen.blit(gb_out, (10, y_bot + y_extra))
                y_extra += 30

            if self.is_dynamic and self.mode != 'train_neat' and self.mode != 'train_rl':
                 alive_out = render_text_with_outline(self.font, f"Alive: {p.time_alive:.1f}s", (0, 255, 0), outline_color=(0,0,0), outline_width=1, aa=False)
                 self.screen.blit(alive_out, (10, y_bot + y_extra))
            if self.external_stats:
                ex_y = 70
                ex_x = self.width - 250
                for label, value in self.external_stats.items():
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    ex_surf = render_text_with_outline(self.font, f"{label}: {val_str}", (0, 255, 255), outline_color=(0,0,0), outline_width=1, aa=False)
                    self.screen.blit(ex_surf, (ex_x, ex_y))
                    ex_y += 30

        if self.mode != 'train_neat' and self.mode != 'train_rl':
            all_crashed = (all(r.car.crashed for r in self.racers) and is_multi_agent)
            manual_agent_crashed = any(r.type == 'manual' for r in self.racers if r.car.crashed)
            msg = None
            if all_crashed:
                msg = "ALL CRASHED! Press R to Reset"
            elif manual_agent_crashed or (len(self.racers) == 1 and self.racers[0].car.crashed):
                msg = "CRASHED! Press R to Reset"

            if msg:
                crash_text = render_text_with_outline(self.font, msg, (255, 0, 0), outline_color=(0,0,0), outline_width=2, aa=False)
                center_x = self.width // 2 - crash_text.get_width() // 2
                center_y = self.height // 2 - crash_text.get_height() // 2
                self.screen.blit(crash_text, (center_x, center_y))

    def step(self, action):
        """
        Advance the game by one step (1/60s).
        action: [steering, throttle, brake] or None (if None, use internal agent)
        Focuses on Primary Racer (Index 0).
        """
        dt = FIXED_DT
        if action is not None:
             self.current_inputs = action
        else:
             pass
        reward = self._advance_game(dt)
        
        state = self.get_state()
        done = self.racers[0].car.crashed
        info = {
            'lap': self.racers[0].lap,
            'checkpoint': self.racers[0].checkpoint_idx,
            'time': self.racers[0].current_lap_time
        }
        
        return state, reward, done, info

    def get_state(self):
        """Returns attributes of the primary racer (index 0)."""
        r = self.racers[0]
        return {
            'speed': r.car.speed,
            'sensors': [req[2] for req in r.car.rays],
            'angle': r.car.heading,
            'pos': r.car.pos,
            'heading': r.car.heading,
            'checkpoint_idx': r.checkpoint_idx,
            'track_nodes': self.track.nodes,
            'neural_inputs': environment.get_inputs(r.car, self.track)
        }

    def run(self):
        self.running = True
        self.accumulator = 0.0
        self.render()
        pygame.display.flip()
        
        while self.running:
            raw_dt = self.clock.tick(60) / 1000.0
            
            self.handle_input()
            
            if not self.paused:
                self.accumulator += raw_dt * self.sim_speed
                
                while self.accumulator >= self.fixed_dt:
                    self.update(self.fixed_dt)
                    self.accumulator -= self.fixed_dt
            
            elif self.single_step:
                self.update(self.fixed_dt)
                self.single_step = False
                
            self.render()
            pygame.display.flip()
        if hasattr(self, 'track'):
             track_name = 'dynamic' if isinstance(self.track, DynamicTrack) else os.path.splitext(os.path.basename(self.track.file_path))[0]
             if isinstance(self.track, DynamicTrack):
                 for racer in self.racers:
                     if not racer.finished and not racer.car.crashed:
                         self.highscore_manager.update_highscore(
                             'dynamic',
                             racer.name,
                             racer.time_alive,
                             {'checkpoints': racer.total_checkpoints_passed},
                             mode='survival'
                         )
             self.highscore_manager.save_highscores(track_name)

        pygame.quit()
        sys.exit()


