import pygame
import random
from .utils import bezier_curve_points, check_gate_crossing, render_text_with_outline
from .student.settings import SCREEN_WIDTH, SCREEN_HEIGHT, LAP_TILES



class TileType:
    EMPTY = 0
    STRAIGHT = 1
    CORNER_LEFT = 2
    CORNER_RIGHT = 3


class DynamicTrack:
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, max_tiles_behind=3):
        self.width = width
        self.height = height
        self.cols = 4
        self.rows = 3
        self.cell_w = self.width // self.cols
        self.cell_h = self.height // self.rows 
        self.grid = {} 
        
        self.car_scale = 0.5
        self.cam_zoom = 1.5
        self.start_line = {'x': self.cell_w // 2, 'y': self.cell_h // 2, 'angle': 0, 'width': 50}
        self.boundary_color = (0, 0, 0)
        self.road_colour = (100, 100, 100)
        self.track_width = 80
        self.lookahead = 2
        
        self.max_tiles_behind = max_tiles_behind
        self.lap_tiles = LAP_TILES
        
        self.master_surface = pygame.Surface((self.width, self.height))
        self.master_surface.set_colorkey((0,0,0))
        
        self.tile_variations = [
            'STRAIGHT',
            'WAVE_LEFT',
            'WAVE_RIGHT', 
            'CHICANE_LR',
            'CHICANE_RL',
        ]
        
        self.corner_variations = [
            'CORNER',
            'TIGHT_CORNER',
            'WIDE_CORNER',
            'HAIRPIN',
        ]
        
        self.reset()
        
    def reset(self, seed=None):
        if seed is not None:
             random.seed(seed)
             
        self.grid = {}
        self.active_path = []
        self.nodes = []
        self.last_path_index = 0
        self.dirty = True

        self.start_line = {'x': self.cell_w // 2, 'y': self.cell_h // 2, 'angle': 0, 'width': 50}
        cx, cy = 0, 0
        entry_vec = (-1, 0)
        exit_vec = (1, 0)
        tile_type = 'STRAIGHT'
        
        entry_offset = 0
        exit_offset = 0
        
        self.grid[(cx, cy)] = {
            'type': tile_type,
            'variation': 'STRAIGHT',
            'entry': entry_vec,
            'exit': exit_vec,
            'entry_offset': entry_offset,
            'exit_offset': exit_offset,
            'surface': self._render_tile(tile_type, entry_vec, exit_vec, entry_offset, exit_offset, 'STRAIGHT')
        }
        self.active_path.append((cx, cy))
                
        current = (cx, cy)
        for _ in range(3):
            current = self.extend_path(current)
            
        self.dirty = True

    def render_update(self):
        """Lazy update of master surface."""
        if getattr(self, 'dirty', False):
            self._redraw_master()
            self.dirty = False

    def extend_path(self, from_cell, force=False):
        """Generates the NEXT tile based on the exit of from_cell.
        
        Args:
            from_cell: The cell to extend from
            force: If True, will overwrite protected cells if no other option exists
            
        Returns:
            The cell that was created, or if a protected cell was overwritten, returns that cell
            so the caller can crash any cars inside it.
        """
        if from_cell not in self.grid:
            return None
            
        prev_tile = self.grid[from_cell]
        move_vec = prev_tile['exit']
        
        nx = (from_cell[0] + move_vec[0]) % self.cols
        ny = (from_cell[1] + move_vec[1]) % self.rows
        next_cell = (nx, ny)
        
        protected = getattr(self, 'protected_cells', set())
        
        should_overwrite = True
        overwritten_protected = None
        
        if next_cell in self.grid:
             if next_cell in protected:
                 if force:
                     overwritten_protected = next_cell
                     should_overwrite = True
                 else:
                     should_overwrite = False
        else:
             should_overwrite = True
             
        if not should_overwrite:
             self.active_path.append(next_cell)
             return None
             
        entry_vec = (-move_vec[0], -move_vec[1])
        
        exit_options = []
        exit_options.append(('STRAIGHT', move_vec))
        l_dir = (move_vec[1], -move_vec[0])
        exit_options.append(('CORNER_LEFT', l_dir))
        r_dir = (-move_vec[1], move_vec[0])
        exit_options.append(('CORNER_RIGHT', r_dir))
        random.shuffle(exit_options)
        
        candidates_empty = []
        candidates_overwrite = []
        candidates_protected = []

        for e_type, e_dir in exit_options:
            fx = (nx + e_dir[0]) % self.cols
            fy = (ny + e_dir[1]) % self.rows
            future = (fx, fy)
            
            if future not in self.grid:
                candidates_empty.append((e_type, e_dir))
            elif future not in protected:
                candidates_overwrite.append((e_type, e_dir))
            else:
                candidates_protected.append((e_type, e_dir))
                
        best_exit = None
        if candidates_empty:
             scored_empty = []
             for c_type, c_dir in candidates_empty:
                 fx = (nx + c_dir[0]) % self.cols
                 fy = (ny + c_dir[1]) % self.rows
                 open_edges = 0
                 for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                     nx_n = (fx + dx) % self.cols
                     ny_n = (fy + dy) % self.rows
                     if (nx_n, ny_n) not in self.grid:
                         open_edges += 1
                 scored_empty.append((open_edges, (c_type, c_dir)))
             scored_empty.sort(key=lambda x: x[0], reverse=True)
             best_exit = scored_empty[0][1]
        elif candidates_overwrite:
             best_exit = candidates_overwrite[0]
        elif candidates_protected:
             if force:
                 best_exit = candidates_protected[0]
             else:
                 best_exit = exit_options[0]
        if not best_exit: best_exit = exit_options[0]
            
        choice_type, choice_dir = best_exit
        max_off = int(min(self.cell_w, self.cell_h) * 0.2)
        exit_offset = random.randint(-max_off, max_off)
        prev_exit_offset = prev_tile.get('exit_offset', 0)
        entry_offset = prev_exit_offset
        tile_type = self._determine_tile_type(entry_vec, choice_dir)
        
        variation = self._choose_variation(tile_type)
        
        self.grid[next_cell] = {
            'type': tile_type,
            'variation': variation,
            'entry': entry_vec,
            'exit': choice_dir,
            'entry_offset': entry_offset,
            'exit_offset': exit_offset,
            'surface': self._render_tile(tile_type, entry_vec, choice_dir, entry_offset, exit_offset, variation)
        }
        self.active_path.append(next_cell)
        
        self.dirty = True
        return overwritten_protected

    def _choose_variation(self, tile_type):
        """Choose a random variation for the given tile type."""
        if tile_type == 'STRAIGHT':
            weights = [0.4, 0.15, 0.15, 0.15, 0.15]
            return random.choices(self.tile_variations, weights=weights)[0]
        else:
            weights = [0.2, 0.4, 0.1, 0.3]
            return random.choices(self.corner_variations, weights=weights)[0]

    def _get_edge_local(self, vec, offset):
        """Helper to get local edge coordinate for a vector and offset."""
        cx, cy = self.cell_w // 2, self.cell_h // 2
        if vec == (0, -1): return (cx + offset, 0)
        if vec == (0, 1): return (cx + offset, self.cell_h)
        if vec == (-1, 0): return (0, cy + offset)
        if vec == (1, 0): return (self.cell_w, cy + offset)
        return (cx, cy)
        
    def _determine_tile_type(self, entry, exit):
        if entry[0] == -exit[0] and entry[1] == -exit[1]:
            return 'STRAIGHT'
        return 'CORNER'
    
    def _render_tile(self, t_type, entry, exit, entry_off, exit_off, variation='STRAIGHT'):
        surf = pygame.Surface((self.cell_w, self.cell_h))
        w_col = self.boundary_color
        surf.fill(w_col)
        
        cx, cy = self.cell_w // 2, self.cell_h // 2
        
        p_entry = self._get_edge_local(entry, entry_off)
        p_exit = self._get_edge_local(exit, exit_off)
        
        track_width = self.track_width
        road_col = self.road_colour
        
        if t_type == 'STRAIGHT':
            self._render_straight_variation(surf, p_entry, p_exit, entry, exit, track_width, road_col, variation)
        elif t_type == 'CORNER':
            self._render_corner_variation(surf, p_entry, p_exit, entry, exit, entry_off, exit_off, track_width, road_col, variation)
                
        used_edges = [entry, exit]
        all_edges = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for edge in all_edges:
            if edge not in used_edges:
                sw, sh = self.cell_w, self.cell_h
                start, end = (0,0), (0,0)
                
                if edge == (0, -1):
                    start, end = (0,0), (sw, 0)
                elif edge == (0, 1):
                    start, end = (0, sh), (sw, sh)
                elif edge == (-1, 0):
                    start, end = (0, 0), (0, sh)
                elif edge == (1, 0):
                    start, end = (sw, 0), (sw, sh)
                    
                pygame.draw.line(surf, w_col, start, end, 5)

        return surf

    def _render_straight_variation(self, surf, p_entry, p_exit, entry, exit, track_width, road_col, variation):
        """Render straight tile with various curve patterns."""
        cx, cy = self.cell_w // 2, self.cell_h // 2
        
        if variation == 'STRAIGHT':
            pygame.draw.line(surf, road_col, p_entry, p_exit, track_width)
            pygame.draw.circle(surf, road_col, p_entry, track_width//2)
            pygame.draw.circle(surf, road_col, p_exit, track_width//2)
            
        elif variation in ['WAVE_LEFT', 'WAVE_RIGHT']:
            self._draw_wave(surf, p_entry, p_exit, entry, exit, track_width, road_col, variation == 'WAVE_LEFT')
            
        elif variation in ['CHICANE_LR', 'CHICANE_RL']:
            self._draw_chicane(surf, p_entry, p_exit, entry, exit, track_width, road_col, variation == 'CHICANE_LR')

    def _draw_wave(self, surf, p_entry, p_exit, entry, exit, track_width, road_col, wave_left):
        """Draw a smooth S-curve wave pattern with tangent alignment at edges."""
        amplitude = min(self.cell_w, self.cell_h) * 0.35
        if not wave_left:
            amplitude = -amplitude
        
        entry_dir = (-entry[0], -entry[1])
        exit_dir = (exit[0], exit[1])
        
        normal = (-entry_dir[1], entry_dir[0])
        
        tangent_len = min(self.cell_w, self.cell_h) * 0.4
        
        ctrl1 = (
            p_entry[0] + entry_dir[0] * tangent_len + normal[0] * amplitude,
            p_entry[1] + entry_dir[1] * tangent_len + normal[1] * amplitude
        )
        ctrl2 = (
            p_exit[0] - exit_dir[0] * tangent_len - normal[0] * amplitude,
            p_exit[1] - exit_dir[1] * tangent_len - normal[1] * amplitude
        )
        
        points = bezier_curve_points(p_entry, ctrl1, ctrl2, p_exit, steps=32)
        
        self._draw_thick_curve(surf, points, track_width, road_col)

    def _draw_chicane(self, surf, p_entry, p_exit, entry, exit, track_width, road_col, chicane_lr):
        """Draw a sharp chicane pattern with tangent alignment at edges."""
        amplitude = min(self.cell_w, self.cell_h) * 0.8
        if not chicane_lr:
            amplitude = -amplitude
        
        entry_dir = (-entry[0], -entry[1])
        exit_dir = (exit[0], exit[1])
        
        tangent_len = min(self.cell_w, self.cell_h) * 0.5
        normal = (-entry_dir[1], entry_dir[0])
        
        ctrl1 = (
            p_entry[0] + entry_dir[0] * tangent_len + normal[0] * amplitude,
            p_entry[1] + entry_dir[1] * tangent_len + normal[1] * amplitude
        )
        
        ctrl2 = (
            p_exit[0] - exit_dir[0] * tangent_len - normal[0] * amplitude,
            p_exit[1] - exit_dir[1] * tangent_len - normal[1] * amplitude
        )
        
        points = bezier_curve_points(p_entry, ctrl1, ctrl2, p_exit, steps=32)
        
        self._draw_thick_curve(surf, points, track_width, road_col)

    def _render_corner_variation(self, surf, p_entry, p_exit, entry, exit, entry_off, exit_off, track_width, road_col, variation):
        """Render corner tile with various curve patterns using tangent-aligned control points."""
        entry_dir = (-entry[0], -entry[1])
        exit_dir = (exit[0], exit[1])
        
        base_len = min(self.cell_w, self.cell_h) * 0.5
        
        if variation == 'CORNER':
            tangent_len = base_len * 0.7
            ctrl1 = (
                p_entry[0] + entry_dir[0] * tangent_len,
                p_entry[1] + entry_dir[1] * tangent_len
            )
            ctrl2 = (
                p_exit[0] - exit_dir[0] * tangent_len,
                p_exit[1] - exit_dir[1] * tangent_len
            )
            points = bezier_curve_points(p_entry, ctrl1, ctrl2, p_exit, steps=24)
            self._draw_thick_curve(surf, points, track_width, road_col)
            
        elif variation == 'TIGHT_CORNER':
            tangent_len = base_len * 0.25
            ctrl1 = (
                p_entry[0] + entry_dir[0] * tangent_len,
                p_entry[1] + entry_dir[1] * tangent_len
            )
            ctrl2 = (
                p_exit[0] - exit_dir[0] * tangent_len,
                p_exit[1] - exit_dir[1] * tangent_len
            )
            points = bezier_curve_points(p_entry, ctrl1, ctrl2, p_exit, steps=32)
            self._draw_thick_curve(surf, points, track_width, road_col)
            
        elif variation == 'WIDE_CORNER':
            tangent_len = base_len * 1.0
            ctrl1 = (
                p_entry[0] + entry_dir[0] * tangent_len,
                p_entry[1] + entry_dir[1] * tangent_len
            )
            ctrl2 = (
                p_exit[0] - exit_dir[0] * tangent_len,
                p_exit[1] - exit_dir[1] * tangent_len
            )
            points = bezier_curve_points(p_entry, ctrl1, ctrl2, p_exit, steps=24)
            self._draw_thick_curve(surf, points, track_width, road_col)

        elif variation == 'HAIRPIN':
            tangent_len = base_len * 1.6
            ctrl1 = (
                p_entry[0] + entry_dir[0] * tangent_len,
                p_entry[1] + entry_dir[1] * tangent_len
            )
            ctrl2 = (
                p_exit[0] - exit_dir[0] * tangent_len,
                p_exit[1] - exit_dir[1] * tangent_len
            )
            points = bezier_curve_points(p_entry, ctrl1, ctrl2, p_exit, steps=40)
            self._draw_thick_curve(surf, points, track_width, road_col)

    def _draw_thick_curve(self, surf, points, width, color):
        """Draw a thick curve by drawing circles along bezier points and connecting lines."""
        radius = width // 2
        edge_bleed = 5
        
        for i in range(len(points) - 1):
            p1 = (int(points[i][0]), int(points[i][1]))
            p2 = (int(points[i+1][0]), int(points[i+1][1]))
            pygame.draw.line(surf, color, p1, p2, width)
        
        for p in points:
            pygame.draw.circle(surf, color, (int(p[0]), int(p[1])), radius)
        
    def _redraw_master(self):
        self.master_surface.fill((0, 0, 0))
        
        for (gx, gy), tile in self.grid.items():
             px = gx * self.cell_w
             py = gy * self.cell_h
             self.master_surface.blit(tile['surface'], (px, py))

    def update(self, cars):
        if not cars: return
        if not isinstance(cars, list): cars = [cars]

        active_cars = [c for c in cars if not c.crashed]
        if not active_cars: return

        if not hasattr(self, 'car_path_indices'):
            self.car_path_indices = {}

        car_progress = []
        for car in active_cars:
            col = int(car.pos.x // self.cell_w) % self.cols
            row = int(car.pos.y // self.cell_h) % self.rows
            cell = (col, row)

            cid = id(car)
            last_index = self.car_path_indices.get(cid, 0)

            try:
                candidates = [i for i, x in enumerate(self.active_path) if x == cell]
                if not candidates:
                    found_idx = last_index
                else:
                    found_idx = min(candidates, key=lambda x: abs(x - last_index))
            except:
                found_idx = last_index

            if found_idx == -1:
                found_idx = last_index

            self.car_path_indices[cid] = found_idx
            car_progress.append((car, found_idx, cell))

        if not car_progress: return
        car_progress.sort(key=lambda x: x[1], reverse=True)
        leader_car, leader_idx, leader_cell = car_progress[0]

        max_tiles_behind = getattr(self, 'max_tiles_behind', 4)

        for car, idx, cell in car_progress:
            tiles_behind = leader_idx - idx
            if tiles_behind > max_tiles_behind:
                car.crashed = True
                if hasattr(car, 'straggler_crash'):
                    car.straggler_crash = True
            elif tiles_behind > max_tiles_behind // 2:
                car.crashed = True
                if hasattr(car, 'straggler_crash'):
                    car.straggler_crash = True

        active_cars = [c for c in cars if not c.crashed]
        if not active_cars: return

        car_progress = [(car, idx, cell) for car, idx, cell in car_progress if not car.crashed]

        max_protected = min(2, len(car_progress))
        leader_cars = car_progress[:max_protected]

        self.protected_cells = set()
        for car, idx, cell in leader_cars:
            self.protected_cells.add(cell)

        current_indices = [x[1] for x in car_progress]
        leader_idx = max(current_indices) if current_indices else 0
        trailer_idx = min(current_indices) if current_indices else 0

        margin = 1
        pop_count = 0

        while trailer_idx > margin and len(self.active_path) > margin:
            to_pop = self.active_path[0]
            if to_pop in self.protected_cells:
                break

            popped_cell = self.active_path.pop(0)

            for car in active_cars:
                col = int(car.pos.x // self.cell_w) % self.cols
                row = int(car.pos.y // self.cell_h) % self.rows
                if (col, row) == popped_cell:
                    car.crashed = True

            for car in cars:
                if hasattr(car, 'checkpoint_idx'):
                    car.checkpoint_idx = max(0, car.checkpoint_idx - 1)
                if getattr(car, 'last_tile', None) == popped_cell:
                    car.last_tile = None
                if getattr(car, 'last_tiles', None) and popped_cell in car.last_tiles:
                    car.last_tiles = None

            pop_count += 1
            trailer_idx -= 1
            leader_idx -= 1

            if popped_cell not in self.active_path:
                if popped_cell in self.grid:
                    del self.grid[popped_cell]

        if pop_count > 0:
            for cid in self.car_path_indices:
                self.car_path_indices[cid] = max(0, self.car_path_indices[cid] - pop_count)
            for car in cars:
                if hasattr(car, 'last_lap_path_index'):
                    car.last_lap_path_index = max(0, car.last_lap_path_index - pop_count)

        lookahead = self.lookahead
        while len(self.active_path) <= leader_idx + lookahead:
            if self.active_path:
                overwritten_cell = self.extend_path(self.active_path[-1], force=True)
                if overwritten_cell:
                    for car in active_cars:
                        col = int(car.pos.x // self.cell_w) % self.cols
                        row = int(car.pos.y // self.cell_h) % self.rows
                        if (col, row) == overwritten_cell:
                            car.crashed = True
            else:
                break

        self.dirty = True

        for cell in self.protected_cells:
            if cell in self.grid:
                self.ensure_neighbors(cell)

    def ensure_neighbors(self, cell_coords):
        if cell_coords not in self.grid: return
        
        tile = self.grid[cell_coords]
        
        exit_vec = tile['exit']
        next_cell = ((cell_coords[0] + exit_vec[0]) % self.cols, 
                     (cell_coords[1] + exit_vec[1]) % self.rows)
        if next_cell not in self.grid:
             if self.active_path and self.active_path[-1] == cell_coords:
                 self.extend_path(cell_coords)
                 self.dirty = True
        
        entry_vec = tile['entry']
        prev_cell = ((cell_coords[0] + entry_vec[0]) % self.cols,
                     (cell_coords[1] + entry_vec[1]) % self.rows)
                     
        if prev_cell not in self.grid:
            self.prepend_path(cell_coords)
            self.dirty = True
            
    def prepend_path(self, to_cell):
        """Generates a tile that CONNECTS TO to_cell's entry."""
        if to_cell not in self.grid: return
        
        target_tile = self.grid[to_cell]
        t_entry = target_tile['entry']
        t_entry_offset = target_tile['entry_offset']
        
        prev_cell = ((to_cell[0] + t_entry[0]) % self.cols,
                     (to_cell[1] + t_entry[1]) % self.rows)
        
        exit_vec = (-t_entry[0], -t_entry[1])
        exit_offset = t_entry_offset
        
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        valid_entries = []
        for d in dirs:
            if d != exit_vec: 
                 valid_entries.append(d)
                 
        entry_vec = random.choice(valid_entries)
        
        max_off = int(min(self.cell_w, self.cell_h) * 0.2)
        entry_offset = random.randint(-max_off, max_off)
        
        tile_type = self._determine_tile_type(entry_vec, exit_vec)
        variation = self._choose_variation(tile_type)
        
        self.grid[prev_cell] = {
            'type': tile_type,
            'variation': variation,
            'entry': entry_vec,
            'exit': exit_vec,
            'entry_offset': entry_offset,
            'exit_offset': exit_offset,
            'surface': self._render_tile(tile_type, entry_vec, exit_vec, entry_offset, exit_offset, variation)
        }
        
        self.active_path.insert(0, prev_cell)
        self.last_path_index += 1 
        self.dirty = True
        
        return prev_cell

    def get_collision(self, x, y):
        wx = x % self.width
        wy = y % self.height
        
        col = int(wx // self.cell_w)
        row = int(wy // self.cell_h)
        
        if (col, row) not in self.grid:
            return True
            
        tile = self.grid[(col, row)]
        
        lx = int(wx % self.cell_w)
        ly = int(wy % self.cell_h)
        
        try:
            color = tile['surface'].get_at((lx, ly))
            return color[:3] == self.boundary_color
        except:
            return True
            
    def check_checkpoint(self, car, idx):
        """
        Returns True when the car moves into a different tile (cell).
        idx is ignored; checkpointing is per-tile. We consider both the car
        center and its corners to detect tile transitions more robustly.
        The first time we see a car we initialize its last_tiles and do NOT
        count that as a checkpoint pass.
        """
        def pos_to_tile(x, y):
            col = int(x // self.cell_w) % self.cols
            row = int(y // self.cell_h) % self.rows
            return (col, row)

        current_tiles = set()
        try:
            current_tiles.add(pos_to_tile(car.pos.x, car.pos.y))
        except Exception:
            pass

        if hasattr(car, 'get_corners'):
            try:
                for c in car.get_corners():
                    current_tiles.add(pos_to_tile(c.x, c.y))
            except Exception:
                pass

        last_tiles = getattr(car, 'last_tiles', None)
        if last_tiles is None:
            car.last_tiles = current_tiles
            car.last_tile = next(iter(current_tiles)) if current_tiles else None
            return False

        new_tiles = current_tiles - last_tiles
        if new_tiles:
            car.last_tiles = current_tiles
            car.last_tile = next(iter(current_tiles)) if current_tiles else None
            return True

        return False
        
    def check_start_finish(self, car):
        """
        For dynamic tracks: determine if the car has completed a lap based on
        tiles crossed. Uses the index of the car's current tile in
        self.active_path and compares it to the last lap index stored on the car.
        Returns True if lap threshold is reached and resets bookkeeping.
        """
        if not self.active_path:
            return False

        current_tile = getattr(car, 'last_tile', None)
        if current_tile is None:
            col = int(car.pos.x // self.cell_w) % self.cols
            row = int(car.pos.y // self.cell_h) % self.rows
            current_tile = (col, row)

        try:
            idx = self.active_path.index(current_tile)
        except ValueError:
            return False

        last_idx = getattr(car, 'last_lap_path_index', None)
        if last_idx is None:
            car.last_lap_path_index = idx
            return False

        if idx >= last_idx:
            delta = idx - last_idx
        else:
            delta = (len(self.active_path) - last_idx) + idx

        if delta >= getattr(self, 'lap_tiles', 10):
            # completed a lap
            car.last_lap_path_index = idx
            return True

        return False

    def _check_crossing(self, car, gate):
        """
        Generic line crossing check.
        gate: {x, y, width, angle}
        """
        return check_gate_crossing(car, gate, track_width=self.width, track_height=self.height)


    def render_debug(self, surface, car):
        """Draws grid lines and cell highlights for debugging."""
        for c in range(self.cols + 1):
            x = c * self.cell_w
            pygame.draw.line(surface, (100, 100, 100), (x, 0), (x, self.height), 2)
            
        for r in range(self.rows + 1):
            y = r * self.cell_h
            pygame.draw.line(surface, (100, 100, 100), (0, y), (self.width, y), 2)
            
        cx = int(car.pos.x // self.cell_w)
        cy = int(car.pos.y // self.cell_h)
        rect = (cx * self.cell_w, cy * self.cell_h, self.cell_w, self.cell_h)
        pygame.draw.rect(surface, (0, 255, 0), rect, 4)
        
        if self.active_path:
             tx, ty = self.active_path[-1]
             t_rect = (tx * self.cell_w, ty * self.cell_h, self.cell_w, self.cell_h)
             pygame.draw.rect(surface, (0, 0, 255), t_rect, 4)
             
        font = pygame.font.SysFont(None, 24)
        tip_str = f"{self.active_path[-1]}" if self.active_path else "None"
        
        if self.active_path and self.active_path[-1] in self.grid:
            tile_info = self.grid[self.active_path[-1]]
            var = tile_info.get('variation', 'unknown')
            tip_str = f"{self.active_path[-1]} ({var})"
            
        text = render_text_with_outline(font, f"Car: {cx},{cy}  Tip: {tip_str}", (255, 255, 0), outline_color=(0,0,0), outline_width=1, aa=True)
        surface.blit(text, (10, 120))
