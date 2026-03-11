import argparse
import signal
import threading
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
import pygame
import numpy as np
import sys
import subprocess
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from neon_racer.student.rl_wrapper import NeonRacerEnv
from neon_racer.metrics import MetricsLogger
from neon_racer.utils import is_process_running

def make_env_ignore_sigint(**kwargs):
    """Wrapper to create env with SIGINT ignored (prevents child process traceback spam)."""
    def _init():
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
        return Monitor(NeonRacerEnv(**kwargs))
    return _init

def close_env_with_timeout(env, timeout=2.0):
    """Close environment with timeout, force-killing workers if they hang."""
    def _close():
        try:
            env.close()
        except:
            pass
    
    close_thread = threading.Thread(target=_close, daemon=True)
    close_thread.start()
    close_thread.join(timeout=timeout)
    
    if close_thread.is_alive():
        if hasattr(env, 'processes') and env.processes:
            for p in env.processes:
                if p.is_alive():
                    p.terminate()
            for p in env.processes:
                p.join(timeout=0.5)
            for p in env.processes:
                if p.is_alive():
                    p.kill()

class HumanInteractiveCallback(BaseCallback):
    """
    Custom callback to handle Pygame events (rendering, speed control)
    during the RL training loop.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.visuals_enabled = True
        self.metrics = MetricsLogger()
        
    def _on_step(self) -> bool:
        
        is_parallel = isinstance(self.training_env, (SubprocVecEnv,)) if 'SubprocVecEnv' in globals() else False
        
        engine = None
        if hasattr(self.training_env, 'envs'):
             engine = self.training_env.envs[0].unwrapped.engine
        
        if engine:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        self.visuals_enabled = not self.visuals_enabled
                        if engine and not engine.user_muted:
                             engine.set_sound(self.visuals_enabled and engine.sim_speed < 2.0)
                        print(f"Visuals: {'ENABLED' if self.visuals_enabled else 'DISABLED (High Speed)'}")
                    elif event.key == pygame.K_p:
                        engine.paused = not engine.paused
                    elif event.key == pygame.K_d:
                        engine.show_debug = not engine.show_debug
                    elif event.key == pygame.K_h:
                        engine.show_hud = not engine.show_hud
                    elif event.key == pygame.K_m:
                        engine.set_sound(not engine.enable_sound, manual=True)
                        print(f"Sound: {'ENABLED' if engine.enable_sound else 'DISABLED'}")
                    elif event.key == pygame.K_LEFTBRACKET:
                        if engine.sim_speed > 0.15:
                            engine.sim_speed = max(0.1, round(engine.sim_speed - 0.1, 1))
                        else:
                            engine.sim_speed = max(0.01, round(engine.sim_speed - 0.01, 2))
                    elif event.key == pygame.K_RIGHTBRACKET:
                        if engine.sim_speed < 0.1:
                            engine.sim_speed = min(0.1, round(engine.sim_speed + 0.01, 2))
                        else:
                            engine.sim_speed = min(50.0, round(engine.sim_speed + 0.5, 1))
                    
                    if engine.sim_speed > 2.0:
                        engine.set_sound(False)
                    elif self.visuals_enabled and not engine.user_muted:
                         engine.set_sound(True)
            fps = self.logger.name_to_value.get("time/fps", 0) if self.logger else 0
            loss = self.logger.name_to_value.get("train/loss", 0.0) if self.logger else 0.0
            val_loss = self.logger.name_to_value.get("train/value_loss", 0.0) if self.logger else 0.0
            entropy = self.logger.name_to_value.get("train/entropy_loss", 0.0) if self.logger else 0.0
            pass_ep_rew_mean = 0.0
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                pass_ep_rew_mean = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            else:
                pass_ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", 0.0) if self.logger else 0.0

            engine.external_stats = {
                "Step": self.num_timesteps,
                "Loss": loss,
                "Val Loss": val_loss,
                "Entropy": entropy,
                "Reward": pass_ep_rew_mean,
                "FPS": int(fps)
            }
            if self.visuals_enabled:
                env = self.training_env.envs[0]
                env.render()
                if engine.sim_speed < 10.0:
                     env.unwrapped.clock.tick(60 * engine.sim_speed)
                     
        if self.logger and self.num_timesteps % 1000 == 0:
             fps = self.logger.name_to_value.get("time/fps", 0)
             loss = self.logger.name_to_value.get("train/loss", 0.0)
             pass_ep_rew_mean = 0.0
             if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                 pass_ep_rew_mean = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
             else:
                 pass_ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)
                 
             step = self.num_timesteps
             
             self.metrics.log('rl', {
                 'step': step,
                 'fps': fps,
                 'loss': loss,
                 'reward': pass_ep_rew_mean
             })

             print(f"\rStep: {step} | FPS: {int(fps)} | Loss: {loss:.4f} | Reward: {pass_ep_rew_mean:.1f} | Visuals: {'ON' if self.visuals_enabled else 'OFF'}      ", end="", flush=True)
                 
        return True

def main():
    parser = argparse.ArgumentParser(description='Neon-Racer RL Training (PPO)')
    parser.add_argument('--track', type=str, default='default', help='Track name to train on')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps to train')
    parser.add_argument('--load', type=str, default=None, help='Path to load existing model')
    parser.add_argument('--save', type=str, default='rl_model', help='Path to save model')
    
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel environments (CPU cores)')
    parser.add_argument('--dashboard', action='store_true', help="Launch training dashboard")
    parser.add_argument('--no-dashboard', dest='dashboard', action='store_false')
    parser.add_argument('--no-sound', action='store_true', help="Disable sound effects")
    parser.set_defaults(dashboard=True)
    
    args = parser.parse_args()
    total_cpus = os.cpu_count() or 1
    per_worker_threads = max(1, total_cpus // max(1, args.jobs))
    os.environ['NEON_THREADS_PER_WORKER'] = str(per_worker_threads)
    os.environ['OMP_NUM_THREADS'] = str(per_worker_threads)
    os.environ['MKL_NUM_THREADS'] = str(per_worker_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(per_worker_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(per_worker_threads)
    try:
        torch.set_num_threads(per_worker_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    print(f"Thread configuration: {per_worker_threads} threads per worker (jobs={args.jobs}, cpus={total_cpus})")
    
    os.makedirs('models', exist_ok=True)
    
    if args.save and not args.save.startswith('models/'):
        args.save = os.path.join('models', args.save)
    if args.load and not args.load.startswith('models/'):
        args.load = os.path.join('models', args.load)
    
    env_kwargs = {'render_mode': None, 'track_name': args.track, 'enable_sound': (not args.no_sound)}
        
    if args.jobs > 1:
        print(f"Parallelizing training across {args.jobs} CPUs...")
        env = SubprocVecEnv([make_env_ignore_sigint(**env_kwargs) for _ in range(args.jobs)])
    else:
        print("Running in single-process mode...")
        env = DummyVecEnv([lambda: Monitor(NeonRacerEnv(render_mode="human", track_name=args.track, enable_sound=(not args.no_sound), mode='train_rl'))])
    
    vec_path = args.save + "_vecnormalize.pkl"
    if args.load:
        load_vec_path = args.load.replace('.zip', '') + "_vecnormalize.pkl"
        if os.path.exists(load_vec_path):
            print(f"Loading normalization stats from {load_vec_path}...")
            env = VecNormalize.load(load_vec_path, env)
            env.training = True
        else:
            print("Warning: No normalization stats found. Starting fresh normalization (might be unstable).")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    if args.load:
        print(f"Loading model from {args.load}...")
        model = PPO.load(args.load, env=env)
    else:
        print("Creating new PPO model...")
        model = PPO("MlpPolicy", env, verbose=0, 
                    learning_rate=3e-4, 
                    ent_coef=0.005, 
                    clip_range=0.2,
                    target_kl=0.03,
                    n_steps=4096, 
                    batch_size=128, 
                    n_epochs=6,
                    policy_kwargs=dict(net_arch=[64, 64]))
    
    print(f"Starting training on track '{args.track}' for {args.timesteps} timesteps...")
    print("Controls: '[' / ']' Speed, 'V' Toggle Visuals, 'P' Pause, 'D' Debug, 'H' HUD")
    
    try:
        dash_process = None
        if args.dashboard:
            if is_process_running(['dashboard.py', 'neon_racer.dashboard']):
                print("[*] Dashboard already running. Skipping launch.")
                dash_process = None
            else:
                print("[*] Launching Training Dashboard...")
                dash_path = os.path.join(os.getcwd(), 'neon_racer', 'dashboard.py')
                dash_process = subprocess.Popen([sys.executable, dash_path], start_new_session=True)

        callback = HumanInteractiveCallback()
        reset_steps = args.load is None
        model.learn(total_timesteps=args.timesteps, callback=callback, reset_num_timesteps=reset_steps)
        
        print(f"Saving model to {args.save}...")
        model.save(args.save)
        print(f"Saving normalization stats to {vec_path}...")
        env.save(vec_path)
        
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving progress...")
        model.save(args.save)
        print(f"✓ Model saved to {args.save}.zip")
        env.save(vec_path)
        print(f"✓ Normalization stats saved to {vec_path}")
        print("\nShutting down workers...", end="", flush=True)
        
    finally:
        close_env_with_timeout(env, timeout=2.0)
        print(" Done.")
        if args.dashboard and 'dash_process' in locals() and dash_process:
             print("[*] Dashboard running in background.")
        print("\nTraining session ended.\n")

if __name__ == "__main__":
    main()
