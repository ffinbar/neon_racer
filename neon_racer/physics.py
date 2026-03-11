from pygame.math import Vector2
from .student import settings
from .utils import normalize_angle

class Car:
    def __init__(self, x, y, angle=0.0, scale=1.0):
        self.pos = Vector2(x, y)
        self.vel = Vector2(0, 0)
        self.heading = angle
        self.speed = 0.0
        self.crashed = False
        self.rays = []
        self.scale = scale
        base_length = settings.BASE_LENGTH
        self.width = base_length * scale * 2
        self.height = base_length * scale
        
        self.max_speed = settings.BASE_MAX_SPEED * scale
        self.acceleration = settings.BASE_ACCEL * scale
        self.friction = settings.BASE_FRICTION * scale
        
        self.brake_force = settings.BASE_BRAKE_FORCE * scale
        self.turn_speed = settings.BASE_TURN_SPEED
        self.drag = settings.BASE_DRAG
        self.turn_threshold = 200.0 * scale
        self.brake_threshold = 10.0 * scale
        self.drift_threshold = 50.0 * scale
        
        self.ray_length = settings.RAY_LENGTH * scale
        self.ray_step = settings.RAY_STEP * scale
        self.checkpoint_idx = 0
        self.laps = 0
        self.current_track_fitness = 0.0
        self.peak_fitness = -float('inf')
        self.slip = 0.0
        self.pivot_offset_ratio = -0.3
        self.prev_back_left = None
        self.prev_back_right = None


    def update(self, dt, throttle, steering, brake, track=None):
        """
        Update physics state.
        throttle: 0.0 to 1.0
        steering: -1.0 (left) to 1.0 (right)
        brake: bool
        track: Track object for collision
        """
        if self.crashed:
            return

        throttle = max(0.0, min(1.0, throttle))
        steering = max(-1.0, min(1.0, steering))
        
        speed = self.vel.length()
        if abs(steering) > 0.01:
            low_speed_factor = min(1.0, speed / self.turn_threshold)
            high_speed_dampening = 1.0 / (1.0 + (speed / self.turn_threshold))
            
            turn_rate = steering * self.turn_speed * low_speed_factor * high_speed_dampening
            old_heading = self.heading
            delta_heading = turn_rate * dt
            new_heading = old_heading + delta_heading
            
            self.heading = normalize_angle(new_heading)
            pivot_dist = self.width * self.pivot_offset_ratio
            
            center_to_pivot = Vector2(pivot_dist, 0).rotate(old_heading)
            
            center_to_pivot_new = center_to_pivot.rotate(delta_heading)
            
            shift = center_to_pivot - center_to_pivot_new
            self.pos += shift
            
        else:
            self.heading = normalize_angle(self.heading)
        heading_vec = Vector2(1, 0).rotate(self.heading)
        if brake:
            if speed > self.brake_threshold:
                braking_force = -self.vel.normalize() * self.acceleration * self.brake_force
                self.vel += braking_force * dt
            else:
                self.vel = Vector2(0, 0)
        elif throttle > 0:
            force = heading_vec * throttle * self.acceleration
            self.vel += force * dt
        if speed > 0:
            self.vel -= self.vel * self.drag * dt
        self.slip = 0.0
        if speed > 0:
            forward_velocity = heading_vec * (self.vel.dot(heading_vec))
            lateral_velocity = self.vel - forward_velocity
            current_lateral_speed = lateral_velocity.length()
            traction = 5.0
            if current_lateral_speed > self.drift_threshold: 
                traction = 1.5
                self.slip = min(1.0, current_lateral_speed / (self.drift_threshold * 3))
                
            if current_lateral_speed > 0:
                friction_force = -lateral_velocity.normalize() * current_lateral_speed * traction
                self.vel += friction_force * dt
        if brake and speed > self.brake_threshold:
            self.slip = max(self.slip, 0.5)
        self.prev_pos = Vector2(self.pos.x, self.pos.y)
        self.pos += self.vel * dt
        
        if hasattr(track, 'width') and hasattr(track, 'height') and track.__class__.__name__ == 'DynamicTrack':
             self.pos.x %= track.width
             self.pos.y %= track.height
             if self.pos.distance_to(self.prev_pos) > 100:
                  self.prev_pos = Vector2(self.pos.x, self.pos.y)
        self.speed = self.vel.length() / self.scale
        if track:
             if self.check_wall_collision(track):
                 self.crashed = True
                 self.vel = Vector2(0, 0)
                 self.speed = 0
             self.cast_rays(track)

    def get_corners(self):
        """
        Returns a list of 4 Vector2 points representing corners of the car.
        """
        fwd = Vector2(1, 0).rotate(self.heading) * (self.width / 2)
        right = Vector2(0, 1).rotate(self.heading) * (self.height / 2)
        
        c1 = self.pos + fwd + right
        c2 = self.pos + fwd - right
        c3 = self.pos - fwd - right
        c4 = self.pos - fwd + right
        
        return [c1, c2, c3, c4]

    def check_wall_collision(self, track):
        if track.get_collision(self.pos.x, self.pos.y): return True
        corners = self.get_corners()
        for p in corners:
            if track.get_collision(p.x, p.y):
                return True
                
        return False
        
             
    def cast_rays(self, track):
        self.rays = []
        angles = [-90, -45, 0, 45, 90]
        max_dist = self.ray_length
        step = self.ray_step
        
        for angle in angles:
            cast_angle = self.heading + angle
            direction = Vector2(1, 0).rotate(cast_angle)
            dist = 0
            curr_pos = Vector2(self.pos.x, self.pos.y)
            hit = False
            
            while dist < max_dist:
                curr_pos += direction * step
                dist += step
                if track.get_collision(curr_pos.x, curr_pos.y):
                    hit = True
                    break
            normalized_dist = dist / self.scale
            self.rays.append((self.pos, curr_pos, normalized_dist))
