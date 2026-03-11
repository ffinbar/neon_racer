import json
import os
import pygame
from .utils import check_gate_crossing
from .student.settings import SCREEN_WIDTH, SCREEN_HEIGHT
_TRACK_IMAGE_CACHE = {}

class Track:
    def __init__(self, track_file):
        """
        track_file: Path to the .json file for the track.
        """
        self.track_data = {}
        self.image = None
        self.mask_image = None
        self.boundary_color = (255, 0, 0)
        self.nodes = []
        self.start_line = None
        self.file_path = track_file
        
        self.load_track(track_file)

    def load_track(self, json_path):
        if not os.path.exists(json_path):
            print(f"Track file not found: {json_path}")
            return

        with open(json_path, 'r') as f:
            data = json.load(f)
            
        self.track_data = data
        image_name = data.get('image_file', 'track.png')
        base_dir = os.path.dirname(json_path)
        
        image_path = os.path.join(base_dir, image_name)
        
        if not os.path.exists(image_path):
            student_parent = os.path.dirname(base_dir)
            possible_path = os.path.join(student_parent, "assets", "tracks", image_name)
            if os.path.exists(possible_path):
                image_path = possible_path
            else:
                project_alt = os.path.join(os.path.abspath(os.path.join(base_dir, "..", "..")), "neon_racer", "student", "assets", "tracks", image_name)
                if os.path.exists(project_alt):
                    image_path = project_alt
                elif os.path.exists(image_name):
                    image_path = image_name
                else:
                    checked = [os.path.join(base_dir, image_name), possible_path, project_alt, image_name]
                    print(f"Track image not found: {image_name}. Checked: {checked}")
        
        if os.path.exists(image_path):
            target_size = (SCREEN_WIDTH, SCREEN_HEIGHT)
            cache_key = (os.path.abspath(image_path), target_size)
            
            if cache_key in _TRACK_IMAGE_CACHE:
                self.image = _TRACK_IMAGE_CACHE[cache_key]
            else:
                loaded_image = pygame.image.load(image_path).convert()
                if loaded_image.get_size() != target_size:
                    print(f"Resampling track image from {loaded_image.get_size()} to {target_size}")
                    self.image = pygame.transform.smoothscale(loaded_image, target_size)
                else:
                    self.image = loaded_image
                _TRACK_IMAGE_CACHE[cache_key] = self.image
                
            self.mask_image = self.image
        else:
            print(f"Track image not found: {image_name} looked in {base_dir}")
            
        self.boundary_color = tuple(data.get('boundary_color', [255, 0, 0]))
        self.nodes = data.get('nodes', [])
        self.start_line = data.get('start_line', None)
        self.car_scale = data.get('car_scale', 1.0)
        self.cam_zoom = data.get('cam_zoom', 1.0)

    def get_collision(self, x, y):
        """
        Check if the pixel at (x, y) matches the boundary color.
        """
        if not self.image:
            return False
        if x < 0 or x >= self.image.get_width() or y < 0 or y >= self.image.get_height():
            return True

        try:
            color = self.image.get_at((int(x), int(y)))
            return color[:3] == self.boundary_color
        except IndexError:
            return True

    def check_checkpoint(self, car, checkpoint_index):
        """
        Checks if the car has crossed checkpoint
        Returns True if crossed.
        """
        if not self.nodes or checkpoint_index >= len(self.nodes):
            return False
            
        node = self.nodes[checkpoint_index]
        return self._check_crossing(car, node)

    def check_start_finish(self, car):
        if not self.start_line:
            return False
        return self._check_crossing(car, self.start_line)

    def _check_crossing(self, car, gate):
        """
        Generic line crossing check.
        gate: {x, y, width, angle}
        """
        return check_gate_crossing(car, gate)

