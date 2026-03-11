import pygame
import sys
import json
import os
import math
import argparse
from .student.settings import SCREEN_WIDTH, SCREEN_HEIGHT
from .utils import render_text_with_outline

class TrackEditor:
    def __init__(self):
        pygame.init()
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Neon-Racer Track Editor")
        self.clock = pygame.time.Clock()
        self.running = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--track", type=str, default="default", help="Name of the track to load/save")
        args = parser.parse_args()
        
        self.track_name = args.track
        self.track_dir = os.path.join("neon_racer", "student", "tracks")
        self.assets_dir = os.path.join("neon_racer", "student", "assets", "tracks")
        
        self.json_path = os.path.join(self.track_dir, f"{self.track_name}.json")
        self.image_path = os.path.join(self.assets_dir, f"{self.track_name}.png")
        
        self.image = None
        self.nodes = []
        self.start_line = None
        self.boundary_color = (255, 0, 0)
        self.car_scale = 1.0
        
        self.selected_node_index = None
        self.mode = "NODES"
        self.font = pygame.font.SysFont(None, 24)
        try:
            self.car_sprite = pygame.image.load("neon_racer/student/assets/car.png").convert_alpha()
            self.car_sprite = pygame.transform.rotozoom(self.car_sprite, 0, 0.1)
        except:
            print("Warning: Could not load car.png for visualizer")
            self.car_sprite = pygame.Surface((100, 50))
            self.car_sprite.fill((0, 0, 255))
        if os.path.exists(self.json_path):
             self.load_track(self.json_path)
        else:
            print(f"Track {self.json_path} not found. Starting new.")
            if os.path.exists(self.image_path):
                 self.load_image(self.image_path)
            else:
                print(f"Image {self.image_path} not found. Creating dummy.")
                self.image = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.image.fill((0, 0, 0))

    def load_image(self, path):
        try:
            loaded_image = pygame.image.load(path).convert()
            target_size = (self.width, self.height)
            if loaded_image.get_size() != target_size:
                print(f"Scaling track image from {loaded_image.get_size()} to {target_size}")
                self.image = pygame.transform.smoothscale(loaded_image, target_size)
            else:
                self.image = loaded_image
        except Exception as e:
            print(f"Error loading image: {e}")

    def load_track(self, json_path):
        if not os.path.exists(json_path):
            return
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            self.boundary_color = tuple(data.get('boundary_color', [255, 0, 0]))
            self.start_line = data.get('start_line', None)
            self.nodes = data.get('nodes', [])
            self.car_scale = data.get('car_scale', 1.0)
            image_name = data.get('image_file', f"{self.track_name}.png")
            possible_path = os.path.join(self.assets_dir, image_name)
            if os.path.exists(possible_path):
                self.image_path = possible_path
                self.load_image(self.image_path)
            else:
                 print(f"Image {image_name} not found in {self.assets_dir}")
            
            print(f"Loaded track from {json_path}")
        except Exception as e:
            print(f"Error loading track: {e}")

    def save_track(self):
        os.makedirs(self.track_dir, exist_ok=True)
        
        data = {
            "image_file": os.path.basename(self.image_path),
            "boundary_color": self.boundary_color,
            "car_scale": self.car_scale,
            "start_line": self.start_line,
            "nodes": self.nodes
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Track saved to {self.json_path}")

    def handle_input(self):
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s and (keys[pygame.K_LGUI] or keys[pygame.K_LCTRL]):
                    self.save_track()
                elif event.key == pygame.K_1:
                    self.mode = "NODES"
                elif event.key == pygame.K_2:
                    self.mode = "START"
                elif event.key == pygame.K_3:
                    self.mode = "COLOR"
                elif event.key == pygame.K_4:
                    self.mode = "ERASER"
                elif event.key == pygame.K_BACKSPACE:
                    if self.selected_node_index is not None:
                        if self.selected_node_index == "START":
                            self.start_line = None
                        elif self.nodes:
                             self.nodes.pop(self.selected_node_index)
                        self.selected_node_index = None
                elif event.key == pygame.K_LEFTBRACKET:
                    self.car_scale = max(0.1, self.car_scale - 0.1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.car_scale = min(1.0, self.car_scale + 0.1)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.mode == "NODES":
                        clicked_node = self.get_node_at(mouse_pos)
                        if clicked_node is not None:
                            self.selected_node_index = clicked_node
                        else:
                            if self.start_line and self.check_dist(self.start_line, mouse_pos) < 20:
                                self.selected_node_index = "START"
                            elif self.selected_node_index == "START":
                                self.nodes.append({
                                    'x': mouse_pos[0],
                                    'y': mouse_pos[1],
                                    'width': 100,
                                    'angle': 0
                                })
                                self.selected_node_index = len(self.nodes) - 1
                            else:
                                self.nodes.append({
                                    'x': mouse_pos[0],
                                    'y': mouse_pos[1],
                                    'width': 100,
                                    'angle': 0
                                })
                                self.selected_node_index = len(self.nodes) - 1

                    elif self.mode == "START":
                        if self.start_line and self.check_dist(self.start_line, mouse_pos) < 20:
                             self.selected_node_index = "START"
                        else:
                            self.start_line = {
                                'x': mouse_pos[0],
                                'y': mouse_pos[1],
                                'width': 100,
                                'angle': 0
                            }
                            self.selected_node_index = "START"
                    
                    elif self.mode == "COLOR":
                        if self.image:
                             self.boundary_color = self.image.get_at(mouse_pos)[:3]
                             print(f"Selected Boundary Color: {self.boundary_color}")
                             
                    elif self.mode == "ERASER":
                        if self.start_line and self.check_dist(self.start_line, mouse_pos) < 20:
                            self.start_line = None
                            self.selected_node_index = None
                            return
                        clicked_node = self.get_node_at(mouse_pos)
                        if clicked_node is not None:
                            self.nodes.pop(clicked_node)
                            self.selected_node_index = None

            if event.type == pygame.MOUSEWHEEL:
                if self.selected_node_index is not None:
                    if self.selected_node_index == "START":
                        node = self.start_line
                    else:
                        node = self.nodes[self.selected_node_index]
                    
                    if keys[pygame.K_LSHIFT]:
                        node['width'] += event.y * 5
                    else:
                        node['angle'] += event.y * 5
        if mouse_buttons[0]:
            if self.selected_node_index == "START" and self.start_line and (self.mode == "NODES" or self.mode == "START"):
                 self.start_line['x'] = mouse_pos[0]
                 self.start_line['y'] = mouse_pos[1]
            elif isinstance(self.selected_node_index, int) and self.mode == "NODES":
                 self.nodes[self.selected_node_index]['x'] = mouse_pos[0]
                 self.nodes[self.selected_node_index]['y'] = mouse_pos[1]

    def check_dist(self, node, pos):
        return math.hypot(node['x'] - pos[0], node['y'] - pos[1])

    def get_node_at(self, pos):
        for i, node in enumerate(self.nodes):
            if self.check_dist(node, pos) < 20:
                return i
        return None

    def draw_node(self, node, color=(0, 255, 0), selected=False):
        center = (node['x'], node['y'])
        angle = node['angle']
        width = node['width']
        pygame.draw.circle(self.screen, color, center, 5)
        rad = math.radians(angle + 90)
        dx = math.cos(rad) * (width/2)
        dy = math.sin(rad) * (width/2)
        
        p1 = (center[0] + dx, center[1] + dy)
        p2 = (center[0] - dx, center[1] - dy)
        
        thickness = 3 if selected else 1
        pygame.draw.line(self.screen, color, p1, p2, thickness)
        dir_rad = math.radians(angle)
        arrow_end = (center[0] + math.cos(dir_rad)*20, center[1] + math.sin(dir_rad)*20)
        pygame.draw.line(self.screen, (255, 255, 0), center, arrow_end, 2)

    def render(self):
        self.screen.fill((50, 50, 50))
        
        if self.image:
             self.screen.blit(self.image, (0, 0))
        if self.start_line and self.nodes:
            start_pos = (self.start_line['x'], self.start_line['y'])
            first_node_pos = (self.nodes[0]['x'], self.nodes[0]['y'])
            pygame.draw.line(self.screen, (0, 255, 255), start_pos, first_node_pos, 2)
            if len(self.nodes) > 1:
                points = [(n['x'], n['y']) for n in self.nodes]
                pygame.draw.lines(self.screen, (0, 255, 255), False, points, 2)
            last_node_pos = (self.nodes[-1]['x'], self.nodes[-1]['y'])
            pygame.draw.line(self.screen, (0, 255, 255), last_node_pos, start_pos, 2)
            
        elif len(self.nodes) > 1:
            points = [(n['x'], n['y']) for n in self.nodes]
            pygame.draw.lines(self.screen, (0, 255, 255), False, points, 2)
        for i, node in enumerate(self.nodes):
            selected = (i == self.selected_node_index)
            self.draw_node(node, selected=selected)
        if self.start_line:
             selected = (self.selected_node_index == "START")
             self.draw_node(self.start_line, color=(255, 255, 255), selected=selected)
        scaled_sprite = pygame.transform.rotozoom(self.car_sprite, 0, self.car_scale)
        sprite_rect = scaled_sprite.get_rect()
        x_pos = self.width - sprite_rect.width - 20
        y_pos = 20
        
        self.screen.blit(scaled_sprite, (x_pos, y_pos))
        ui_text = [
            f"Mode: {self.mode} (1: Nodes, 2: Start, 3: Color, 4: Eraser)",
            f"Boundary Color: {self.boundary_color}",
            "Left Click: Add/Select/Drag (Nodes) | Click (Eraser)",
            "Scroll: Rotate | Shift+Scroll: Width",
            "S: Save",
            f"Car Scale: {self.car_scale:.2f} (Use [ and ] to adjust)",
            f"Nodes: {len(self.nodes)}"
        ]
        
        for i, line in enumerate(ui_text):
            surf = render_text_with_outline(self.font, line, (255, 255, 255), outline_color=(0,0,0), outline_width=1, aa=True)
            self.screen.blit(surf, (10, 10 + i * 20))
        pygame.draw.rect(self.screen, self.boundary_color, (200, 30, 20, 20))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_input()
            self.render()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    editor = TrackEditor()
    editor.run()
