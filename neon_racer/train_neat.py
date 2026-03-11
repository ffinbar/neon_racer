import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
import pygame
import neat
import os
import sys
import pickle
import argparse
import multiprocessing
import time
import subprocess
import signal
import threading
import torch
import json
try:
    import tkinter as tk
except ImportError:
    tk = None
from neon_racer.metrics import MetricsLogger
from neon_racer.game_engine import GameEngine
from neon_racer.physics import Car
from neon_racer.racer import Racer
from neon_racer.student.agent import ManualAgent
from neon_racer.student import environment
from .student.settings import EVAL_SEEDS, MAX_FRAMES, FIXED_DT
from neon_racer.utils import is_process_running, render_text_with_outline

engine = None
target_tracks = ['default']
generation = 0
best_fitness_ever = -float('inf')
visuals_enabled = True
MEASURE_FRAMES = False
MEASURE_FILE = 'frame_measurements.jsonl'


def init_worker(track_list):
    """
    Initializer for worker processes.
    Creates a SINGLE GameEngine instance that persists for the life of the worker.
    """
    global engine
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    engine = GameEngine(mode='train_neat', track_name=track_list[0], headless=True)
    engine.sim_speed = 1.0
    print(f"[Worker {os.getpid()}] Initialized with tracks: {track_list}")

def eval_genome_worker(genome, config, generation_seed):
    """
    Worker function to evaluate a single genome using the persistent engine.
    Returns either a float fitness, or (fitness, frames_used) when MEASURE_FRAMES is enabled.
    """
    global engine, MEASURE_FRAMES

    genome.track_scores = []
    tracks_to_test = getattr(engine, 'worker_tracks', ['default'])
    frames_total = 0

    for track_name in tracks_to_test:
        current_eval_seeds = EVAL_SEEDS if track_name == 'dynamic' else 1
        for seed_idx in range(current_eval_seeds):
            current_seed = generation_seed + seed_idx
            if not engine.change_track(track_name, seed=current_seed):
                 continue
            start_pos = (engine.track.start_line['x'], engine.track.start_line['y'])
            start_angle = engine.track.start_line.get('angle', 0)
            scale = engine.track.car_scale
            
            car = Car(start_pos[0], start_pos[1], angle=start_angle, scale=scale)
            car.cast_rays(engine.track)
            current_track_fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            max_frames = MAX_FRAMES
            frame = 0
            dt = FIXED_DT
            
            while frame < max_frames:
                frame += 1
                if car.crashed:
                    break
                inputs = environment.get_inputs(car, engine.track)
                output = net.activate(inputs)
                controls = environment.translate_neural_output(output)
                steering = controls['steering']
                throttle = controls['throttle']
                brake = controls['brake']
                current_time = frame * dt
                reward_delta, passed_checkpoint, stagnated = environment.perform_step(
                    car, throttle, steering, brake, engine.track, dt, current_time
                )
                if hasattr(engine.track, 'update'):
                    engine.track.update([car])
                if stagnated:
                    car.current_track_fitness -= 5
                    break
            

            frames_total += frame
            genome.track_scores.append(max(0.01, car.current_track_fitness))

    if genome.track_scores:
        fitness = sum(genome.track_scores) / len(genome.track_scores)
    else:
        fitness = 0.0

    if MEASURE_FRAMES:
        return (fitness, frames_total)
    return fitness

def eval_genome_chunk_worker(genome_chunk, config, generation_seed):
    """
    Worker function to evaluate multiple genomes in one call (reduces IPC overhead).
    Returns a list of tuples (genome_id, fitness) or (genome_id, fitness, frames) when MEASURE_FRAMES is enabled.
    """
    results = []
    for genome_id, genome in genome_chunk:
        try:
            res = eval_genome_worker(genome, config, generation_seed)
            if isinstance(res, tuple):
                fitness, frames = res
                results.append((genome_id, fitness, frames))
            else:
                results.append((genome_id, res))
        except Exception as e:
            results.append((genome_id, 0.0))
    return results

def init_worker_helper(tracks):
    """Helper to initialize worker global variables"""
    global engine
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        threads = int(os.environ.get('NEON_THREADS_PER_WORKER', '1'))
    except Exception:
        threads = 1
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if engine is None: 
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        engine = GameEngine(mode='train_neat', track_name=tracks[0], headless=True, enable_sound=False)

def eval_genomes(genomes, config):
    global engine, generation, best_fitness_ever, visuals_enabled, target_tracks
    generation += 1
    if engine is None:
        engine = GameEngine(mode='train_neat', track_name=target_tracks[0])
    racers = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome.track_scores = []
    for track_idx, track_name in enumerate(target_tracks):
        current_eval_seeds = EVAL_SEEDS if track_name == 'dynamic' else 1
        seed_scores_display = []
        for seed_idx in range(current_eval_seeds):
            seed_status = f"{seed_idx+1}/{current_eval_seeds}"
            if seed_scores_display:
                prev_scores = ", ".join([f"{s:.0f}" for s in seed_scores_display])
                seed_status += f" (Prev: {prev_scores})"
            sys.stdout.write(f"\rGen: {generation:<4} | Track: '{track_name}' ({track_idx+1}/{len(target_tracks)}) | Seed: {seed_status}".ljust(100))
            sys.stdout.flush()
            gen_seed = generation * 1000 + track_idx * 10 + seed_idx 
            if not engine.change_track(track_name, seed=gen_seed):
                seed_scores_display.append(0.0)
                print(f"\n[!] Track Generation Failed for Seed {gen_seed}")
                continue
            nets = []
            ge = []
            racers = []
            start_pos = (engine.track.start_line['x'], engine.track.start_line['y'])
            start_angle = engine.track.start_line.get('angle', 0)
            scale = engine.track.car_scale
        
            for i, (genome_id, genome) in enumerate(genomes):
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                nets.append(net)
                racer = Racer(ManualAgent(), {'x': start_pos[0], 'y': start_pos[1], 'angle': start_angle, 'scale': scale}, name=f"G{genome_id}", type='neat')
                hue = (i / len(genomes)) * 360
                c = pygame.Color(0)
                c.hsla = (hue, 100, 50, 100)
                racer.color = (c.r, c.g, c.b)
                engine.generate_racer_sprite(racer, use_existing_color=True)
                racer.car.cast_rays(engine.track)
                racer.car.current_track_fitness = 0.0
                racer.car.frames_used = 0
                
                racers.append(racer)
                ge.append(genome)
            engine.racers = racers
            engine.cars = [r.car for r in racers]
            if racers:
                engine.car = racers[0].car 
            run_sim = True
            max_frames = MAX_FRAMES
            frame = 0
            dt = FIXED_DT
        
            while run_sim and frame < max_frames:
                frame += 1
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        save_best_so_far(genomes)
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                             mx, my = pygame.mouse.get_pos()
                             min_dist = 50
                             clicked = None
                             for r in racers:
                                 if not r.car.crashed:
                                     car_scr = engine.world_to_screen(r.car.pos)
                                     d = car_scr.distance_to(pygame.math.Vector2(mx, my))
                                     if d < min_dist:
                                         min_dist = d
                                         clicked = r
                             engine.manual_focus_racer = clicked
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_v:
                            visuals_enabled = not visuals_enabled
                            if not engine.user_muted:
                                engine.set_sound(visuals_enabled and engine.sim_speed < 2.0)
                            print(f"Visuals: {'ENABLED' if visuals_enabled else 'DISABLED'}")
                        elif event.key == pygame.K_LEFTBRACKET:
                            if engine.sim_speed > 0.15:
                                engine.sim_speed = max(0.1, round(engine.sim_speed - 0.5, 1))
                            else:
                                engine.sim_speed = max(0.05, round(engine.sim_speed - 0.01, 2))
                        elif event.key == pygame.K_RIGHTBRACKET:
                            if engine.sim_speed < 0.1:
                                engine.sim_speed = min(0.1, round(engine.sim_speed + 0.01, 2))
                            else:
                                engine.sim_speed = min(50.0, round(engine.sim_speed + 0.5, 1))
                            if engine.sim_speed > 2.0:
                                engine.set_sound(False)
                            elif visuals_enabled and not engine.user_muted:
                                engine.set_sound(True)
                        elif event.key == pygame.K_p:
                            engine.paused = not engine.paused
                        elif event.key == pygame.K_d:
                            engine.show_debug = not engine.show_debug
                        elif event.key == pygame.K_h:
                            engine.show_hud = not engine.show_hud
                        elif event.key == pygame.K_m:
                            engine.set_sound(not engine.enable_sound, manual=True)
                            print(f"Sound: {'ENABLED' if engine.enable_sound else 'DISABLED'}")
                alive_count = 0
                best_idx = -1
                current_max_f = -float('inf')
    
                if not engine.paused:
                    for x, racer in enumerate(racers):
                        car = racer.car
                        
                        if car.crashed:
                            racer.eliminated = True
                            continue
    
                        alive_count += 1
                        inputs = environment.get_inputs(car, engine.track)
                        output = nets[x].activate(inputs)
                        controls = environment.translate_neural_output(output)
                        steering = controls['steering']
                        throttle = controls['throttle']
                        brake = controls['brake']
                        current_time = frame * dt
                        reward_delta, passed_checkpoint, stagnated = environment.perform_step(
                            car, throttle, steering, brake, engine.track, dt, current_time
                        )
                        
                        old_laps = racer.lap
                        racer.checkpoint_idx = car.checkpoint_idx
                        racer.lap = car.laps
                        racer.current_lap_time += dt
                        if passed_checkpoint:
                            racer.total_checkpoints_passed += 1
                            
                        if racer.lap > old_laps:
                                racer.last_lap_time = racer.current_lap_time
                                racer.current_lap_time = 0.0
                        
                        if stagnated:
                            car.current_track_fitness -= 5
                            car.crashed = True
                            try:
                                if getattr(car, 'frames_used', 0) == 0:
                                    car.frames_used = frame
                            except Exception:
                                car.frames_used = frame
                            racer.eliminated = True
                            continue
                            
                        if car.current_track_fitness < -100:
                            car.crashed = True
                            try:
                                if getattr(car, 'frames_used', 0) == 0:
                                    car.frames_used = frame
                            except Exception:
                                car.frames_used = frame
                            racer.eliminated = True
                            continue
                        
                        score = car.current_track_fitness
                        if score > current_max_f:
                            current_max_f = score
                            best_idx = x

                    if best_idx != -1:
                        leader_racer = racers[best_idx]
                        
                        target_car = leader_racer.car
                        if hasattr(engine, 'manual_focus_racer') and engine.manual_focus_racer:
                            if not engine.manual_focus_racer.car.crashed:
                                target_car = engine.manual_focus_racer.car
                            else:
                                engine.manual_focus_racer = None
                                
                        engine.car = target_car
                        
                        if hasattr(engine.track, 'update'):
                            active_cars_list = [r.car for r in racers if not getattr(r, 'eliminated', False)]
                            if active_cars_list:
                                engine.track.update(active_cars_list)
                            
                    if alive_count == 0:
                        run_sim = False
                if visuals_enabled:
                    engine.render()
                    focus_fit = racers[best_idx].car.current_track_fitness if best_idx != -1 else 0.0
                    info_str = f"G:{generation} T:{track_name} A:{alive_count} S:{engine.sim_speed:.1f} F:{focus_fit:.0f}"
                    
                    surf = render_text_with_outline(engine.font, info_str, (255, 255, 0), outline_color=(0,0,0), outline_width=2, aa=True)
                    engine.screen.blit(surf, (10, engine.height - 35))
                    pygame.display.flip()
                    
                    if engine.sim_speed < 10.0:
                        engine.clock.tick(60 * engine.sim_speed)
                else:
                    if frame % 120 == 0:
                        pygame.display.set_caption(f"NEAT Training - Gen {generation} - Track {track_name} - Fast Mode")
            current_seed_max = 0
            for x, racer in enumerate(racers):
                if getattr(racer.car, 'frames_used', 0) == 0:
                    racer.car.frames_used = frame
                val = max(0.01, racer.car.current_track_fitness)
                if val > current_seed_max:
                    current_seed_max = val
                ge[x].track_scores.append(val)
                if MEASURE_FRAMES:
                    if not hasattr(ge[x], 'frames'):
                        ge[x].frames = 0
                    ge[x].frames += getattr(racer.car, 'frames_used', 0)
                
            seed_scores_display.append(current_seed_max)
    for genome_id, genome in genomes:
        if genome.track_scores:
            genome.fitness = sum(genome.track_scores) / len(genome.track_scores)
        else:
            genome.fitness = 0
    if MEASURE_FRAMES:
        try:
            total_frames = 0
            per_genome = {}
            for genome_id, genome in genomes:
                frames = getattr(genome, 'frames', 0)
                per_genome[genome_id] = frames
                total_frames += frames
            avg_frames = total_frames / len(per_genome) if per_genome else 0
            entry = {
                'generation': generation,
                'total_frames': total_frames,
                'avg_frames_per_genome': avg_frames,
                'per_genome': per_genome
            }
            print(f"[MEASURE] Gen {generation}: total_frames={total_frames}, avg_per_genome={avg_frames:.1f}")
            with open(MEASURE_FILE, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


class CustomParallelEvaluator:
    """
    Custom ParallelEvaluator that uses a persistent Pool with an initializer 
    to create ONE GameEngine per worker, rather than per genome.
    Also supports optional frame measurement when MEASURE_FRAMES is enabled.
    """
    def __init__(self, num_workers, tracks, measure=False):
        self.num_workers = num_workers
        self.tracks = tracks
        self.generation = 0
        self.measure = bool(measure)
        self.last_measurements = {}  # genome_id -> frames used in last eval
        self.measure_history = []
        self.pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=init_worker_helper,
            initargs=(tracks,)
        )

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

    def evaluate(self, genomes, config):
        self.generation += 1
        gen_seed = self.generation * 1000
        genome_list = list(genomes)
        args = [(genome, config, gen_seed) for genome_id, genome in genome_list]
        results = self.pool.starmap(eval_genome_worker, args)
        self.last_measurements = {}
        for (genome_id, genome), res in zip(genome_list, results):
            if isinstance(res, tuple):
                fitness, frames = res
            else:
                fitness, frames = res, 0
            genome.fitness = fitness if fitness is not None else 0.0
            if self.measure:
                self.last_measurements[genome_id] = frames
        
        if self.measure:
            total_frames = sum(self.last_measurements.values())
            avg_frames = total_frames / len(self.last_measurements) if self.last_measurements else 0
            entry = {
                'generation': self.generation,
                'total_frames': total_frames,
                'avg_frames_per_genome': avg_frames,
                'per_genome': self.last_measurements.copy()
            }
            self.measure_history.append(entry)
            try:
                with open(MEASURE_FILE, 'a') as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
        return None

def save_best_so_far(genomes):
    global best_fitness_ever
    
    current_best = None
    current_max_fit = -float('inf')
    for item in genomes:
        if isinstance(item, tuple):
            genome_id, genome = item
        else:
            genome = item
            
        if genome.fitness is not None and genome.fitness > current_max_fit:
            current_max_fit = genome.fitness
            current_best = genome
            
    if current_best and current_max_fit > best_fitness_ever:
        best_fitness_ever = current_max_fit
        print(f"New Best Fitness: {current_max_fit:.2f} - Saving genome...")
        os.makedirs('genomes', exist_ok=True)
        with open('genomes/best_genome.pkl', 'wb') as f:
            pickle.dump(current_best, f)


class TrainingController:
    """
    A simple thread-safe controller with a Tkinter window to manage training.
    """
    def __init__(self):
        self.stop_requested = False
        self.force_quit = False
        self.root = None
        
    def start_gui(self):
        if tk is None:
            print("[!] Tkinter not available, control window disabled.")
            return

        self.root = tk.Tk()
        self.root.title("NEAT Control")
        self.root.geometry("300x150")
        
        lbl = tk.Label(self.root, text="NEAT Training Control", font=("Arial", 14))
        lbl.pack(pady=10)
        
        status_lbl = tk.Label(self.root, text="Running...", fg="green")
        status_lbl.pack(pady=5)
        
        def on_stop():
            self.stop_requested = True
            status_lbl.config(text="Stopping after generation...", fg="orange")
            print("\n[Control] Stop requested. Training will end after current generation.")
            
        def on_force():
            self.force_quit = True
            print("\n[Control] Force Quit requested!")
            self.root.destroy()
            os.kill(os.getpid(), signal.SIGKILL)

        btn_stop = tk.Button(self.root, text="Graceful Stop (Finish Gen)", command=on_stop, bg="#dddddd")
        btn_stop.pack(pady=5, fill=tk.X, padx=20)
        
        btn_force = tk.Button(self.root, text="Violent Force Quit", command=on_force, bg="#ffcccc")
        btn_force.pack(pady=5, fill=tk.X, padx=20)
        
        self.root.protocol("WM_DELETE_WINDOW", on_stop)
        self.root.mainloop()

class GracefulExit(Exception):
    pass

TRAINING_CTRL = TrainingController()




class DynamicReporter(neat.reporting.BaseReporter):
    """
    A custom reporter that updates a single-line status in the terminal.
    """
    def __init__(self, logger=None):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.max_fitness = -float('inf')
        self.logger = logger

    def start_generation(self, generation):
        self.generation = generation
        self.generation_start_time = time.time()
        sys.stdout.write(f"\rGen: {self.generation:<4} | Status: Running...                               ")
        sys.stdout.flush()

    def post_evaluate(self, config, population, species, best_genome):
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        avg_time = sum(self.generation_times) / len(self.generation_times)
        
        fitnesses = [c.fitness for c in population.values() if c.fitness is not None]
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        best_fitness = best_genome.fitness if best_genome else 0.0
        
        if best_fitness > self.max_fitness:
            self.max_fitness = best_fitness
        status = (
            f"\rGen: {self.generation:<4} | "
            f"Pop: {len(population):<3} | "
            f"Best: {best_fitness:<8.2f} (All-time: {self.max_fitness:<8.2f}) | "
            f"Avg: {avg_fitness:<8.2f} | "
            f"Time: {elapsed:<5.2f}s "
        )
        sys.stdout.write(status.ljust(100))
        sys.stdout.flush()
        if self.logger:
            self.logger.log('neat', {
                'generation': self.generation,
                'avg_fitness': avg_fitness,
                'best_fitness': best_fitness,
                'max_fitness': self.max_fitness
            })
            
        if TRAINING_CTRL.stop_requested:
            raise GracefulExit("User requested stop via Control Panel")

def run(config_path, gens, checkpoint_path=None, jobs=1, save_checkpoints=False):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    if checkpoint_path:
        if not checkpoint_path.startswith('checkpoints/') and not os.path.isabs(checkpoint_path):
            organized_path = os.path.join('checkpoints', checkpoint_path)
            if os.path.exists(organized_path):
                checkpoint_path = organized_path
        
        if os.path.exists(checkpoint_path):
            print(f"[*] Restoring Population from checkpoint: {checkpoint_path}")
            p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        else:
            print(f"[!] Checkpoint not found: {checkpoint_path}")
            print("[*] Creating new random population...")
            p = neat.Population(config)
    else:
        print("[*] Creating new random population...")
        p = neat.Population(config)
    logger = MetricsLogger()
    p.add_reporter(DynamicReporter(logger=logger)) 
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    if jobs > 1 and tk:
        gui_thread = threading.Thread(target=TRAINING_CTRL.start_gui, daemon=True)
        gui_thread.start()
        print("[*] Control Window Launched - Check your taskbar")

    def signal_handler(sig, frame):
        if TRAINING_CTRL.stop_requested:
            print("\n[!] Force Quit detected (2nd Ctrl+C). Exiting immediately...")
            sys.exit(1)
        else:
            TRAINING_CTRL.stop_requested = True
            print("\n[!] Ctrl+C detected. Finishing current generation then stopping...")
            print("    Press Ctrl+C again to Force Quit immediately.")
    
    signal.signal(signal.SIGINT, signal_handler)

    def save_neat_checkpoint():
        if save_checkpoints and hasattr(p, 'population'):
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                os.makedirs('checkpoints', exist_ok=True)
                cp = neat.Checkpointer(filename_prefix='checkpoints/neat-checkpoint-')
                orig_species_reporters = None
                if hasattr(p, 'species') and hasattr(p.species, 'reporters'):
                    orig_species_reporters = p.species.reporters
                    p.species.reporters = None
                try:
                    cp.save_checkpoint(p.config, p.population, p.species, p.generation)
                finally:
                    if orig_species_reporters is not None:
                        p.species.reporters = orig_species_reporters
                print(f"\n[*] Checkpoint saved at generation {p.generation}")
                signal.signal(signal.SIGINT, signal_handler)
            except Exception as e:
                print(f"\n[!] Error saving checkpoint: {e}")
                signal.signal(signal.SIGINT, signal_handler)
    try:
        if jobs > 1:
            total_cpus = os.cpu_count() or 1
            threads = max(1, total_cpus // max(1, jobs))
            os.environ['NEON_THREADS_PER_WORKER'] = str(threads)
            os.environ['OMP_NUM_THREADS'] = str(threads)
            os.environ['MKL_NUM_THREADS'] = str(threads)
            os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
            try:
                torch.set_num_threads(threads)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            print(f"Configuring {threads} threads per worker (jobs={jobs}, cpus={total_cpus})")
            print(f"[*] Running Parallel Training on {jobs} CPUs (Optimized Persistent Workers)...")
            pe = CustomParallelEvaluator(jobs, target_tracks, measure=MEASURE_FRAMES)
            winner = p.run(pe.evaluate, gens)
            if MEASURE_FRAMES:
                try:
                    latest = pe.measure_history[-1] if pe.measure_history else None
                    if latest:
                        print(f"[MEASURE] Gen {latest['generation']}: total_frames={latest['total_frames']}, avg_per_genome={latest['avg_frames_per_genome']:.1f}")
                except Exception:
                    pass
        else:
            print(f"[*] Running Visual Training on Single Process...")
            winner = p.run(eval_genomes, gens)
        print(f'\nFinal winner:\n{winner}')
        os.makedirs('genomes', exist_ok=True)
        with open('genomes/best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        save_neat_checkpoint()
        print("")
    
    except GracefulExit:
        print("\n[*] Graceful exit requested. Saving progress...")
        if stats and stats.best_genome():
             os.makedirs('genomes', exist_ok=True)
             with open('genomes/best_genome.pkl', 'wb') as f:
                pickle.dump(stats.best_genome(), f)
        save_neat_checkpoint()
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n[!] Training Interrupted by User (KeyboardInterrupt). FORCE SAVING...")
        if stats and stats.best_genome():
            print(f"[*] Saving Best Genome so far (Fitness: {stats.best_genome().fitness:.2f})...")
            os.makedirs('genomes', exist_ok=True)
            with open('genomes/best_genome.pkl', 'wb') as f:
                pickle.dump(stats.best_genome(), f)
        save_neat_checkpoint()

    except neat.population.CompleteExtinctionException:
        print("\n[!] Population Extinct! Saving best so far and exiting...")
        save_neat_checkpoint()
        
    except Exception as e:
        print(f"\n[!] Unexpected Error during training: {e}")
        save_best_so_far(p.population.values() if hasattr(p, 'population') else [])
        save_neat_checkpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NEAT AI")
    parser.add_argument('--gens', type=int, default=20, help="Number of generations to train")
    parser.add_argument('--track', type=str, default='default', help="Track name(s) to train on (comma-separated)")
    parser.add_argument('--speed', type=float, default=1.0, help="Initial simulation speed")
    parser.add_argument('--checkpoint', type=str, default=None, help="Checkoint file to resume from")
    parser.add_argument('--jobs', type=int, default=1, help="Number of parallel CPUs (1=Visual, >1=Headless)")
    parser.add_argument('--cpenable', action='store_true', help="Enable saving of NEAT checkpoints")
    parser.add_argument('--dashboard', action='store_true', help="Launch training dashboard")
    parser.add_argument('--no-dashboard', dest='dashboard', action='store_false')
    parser.add_argument('--no-sound', action='store_true', help="Disable sound effects")
    parser.add_argument('--measure', action='store_true', help='Measure frame usage and emit a per-generation report (appends JSONL to disk)')
    parser.add_argument('--measure-file', type=str, default='frame_measurements.jsonl', help='File to append frame measurement JSON lines to')
    parser.set_defaults(dashboard=True)
    args = parser.parse_args()
    MEASURE_FRAMES = bool(args.measure)
    MEASURE_FILE = args.measure_file if args.measure_file else MEASURE_FILE
    if MEASURE_FRAMES:
        print(f"[MEASURE] Frame measurement enabled; writing to {MEASURE_FILE}")
    if args.track:
        target_tracks = [t.strip() for t in args.track.split(',')]

    config_path = os.path.join(os.getcwd(), "config-feedforward.txt")
    if args.jobs == 1:
        print(f"Initializing NEAT Training on tracks: {target_tracks} at {args.speed}x speed")
        engine = GameEngine(mode='train_neat', track_name=target_tracks[0], enable_sound=(not args.no_sound))
        engine.sim_speed = args.speed
    else:
        print(f"Initializing Headless NEAT Training on tracks: {target_tracks}")
    
    dash_process = None
    if args.dashboard:
        if is_process_running(['dashboard.py', 'neon_racer.dashboard']):
            print("[*] Dashboard already running. Skipping launch.")
            dash_process = None
        else:
            print("[*] Launching Training Dashboard...")
            dash_path = os.path.join(os.getcwd(), 'neon_racer', 'dashboard.py')
            dash_process = subprocess.Popen([sys.executable, dash_path], start_new_session=True)

    try:
        run(config_path, args.gens, args.checkpoint, args.jobs, args.cpenable)
    finally:
        if dash_process:
            print("[*] Dashboard running in background.")

