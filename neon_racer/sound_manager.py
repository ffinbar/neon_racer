import pygame
import os
import numpy as np
import time
import random
from .student.settings import FIXED_DT
from .utils import quantize, clamp

class SoundManager:
    def __init__(self, assets_dir, max_channels=64, enabled=True):
        self.assets_dir = assets_dir
        self.max_channels = max_channels
        self._enabled = enabled
        self.is_loaded = False
        self.sounds = {}
        self.pitch_cache = {} 
        self.active_sounds = {} 
        
        if self._enabled:
            self._ensure_loaded()
            
    @property
    def enabled(self):
        return self._enabled
        
    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        if self._enabled:
            self._ensure_loaded()
        else:
            self.stop_all()

    def _ensure_loaded(self):
        if self.is_loaded:
            return

        if not pygame.mixer.get_init():
             pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        pygame.mixer.set_num_channels(self.max_channels)
        self.load_sounds()
        self.is_loaded = True
        
    def load_sounds(self):
        engine_path = os.path.join(self.assets_dir, "sound", "engine.wav")
        tires_path = os.path.join(self.assets_dir, "sound", "tires.wav")
        
        self.engine_array = None
        self.engine_len = 0.0
        self.tires_sound = None
        
        if os.path.exists(engine_path):
            try:
                temp_snd = pygame.mixer.Sound(engine_path)
                self.engine_array = pygame.sndarray.array(temp_snd)
                if self.engine_array.ndim == 1:
                     self.engine_array = np.column_stack((self.engine_array, self.engine_array))
                     
                self.engine_len = temp_snd.get_length()
                print(f"Loaded Engine Sound: {self.engine_array.shape}, {self.engine_len:.2f}s")
                self.precompute_pitch_cache()
            except Exception as e:
                print(f"Error loading {engine_path}: {e}")

        if os.path.exists(tires_path):
            try:
                self.tires_sound = pygame.mixer.Sound(tires_path)
                self.tires_sound.set_volume(0.1)
            except Exception as e:
                 print(f"Error loading {tires_path}: {e}")

    def precompute_pitch_cache(self):
        """Pre-generates pitched sounds to avoid runtime stutter."""
        if self.engine_array is None:
            return
            
        print("Pre-computing engine sounds...")
        start_t = time.time()
        
        min_p = 0.5
        max_p = 1.5
        step = 0.01
        
        current = min_p
        while current <= max_p + 0.001:
            self.get_pitched_data(current)
            current += step
            
        print(f"Pre-computed {len(self.pitch_cache)} pitch variants in {time.time() - start_t:.3f}s")

    def get_pitched_data(self, pitch):
        """
        Returns (Sound, resampled_numpy_array) for the given pitch.
        Quantizes pitch to reduce unique sound generation.
        """
        step = 0.01
        q_pitch = quantize(pitch, step, decimals=2)
        
        if q_pitch in self.pitch_cache:
            return self.pitch_cache[q_pitch], q_pitch
            
        if self.engine_array is None:
            return None, 0.0

        indices = np.arange(0, len(self.engine_array), q_pitch)
        indices = np.minimum(indices, len(self.engine_array) - 1).astype(int)
        
        resampled_data = self.engine_array[indices]
        
        try:
            snd = pygame.sndarray.make_sound(resampled_data)
            self.pitch_cache[q_pitch] = (snd, resampled_data)
            return (snd, resampled_data), q_pitch
        except Exception as e:
            print(f"Error creating pitched sound: {e}")
            return None, 0.0

    def get_slice_sound(self, pitch_data, start_phase):
        """
        Create a one-off sound object starting from phase (0.0-1.0).
        """
        snd, data = pitch_data
        start_idx = int(start_phase * len(data))
        if start_idx >= len(data) - 100:
             return None
             
        sliced_data = data[start_idx:]
        return pygame.sndarray.make_sound(sliced_data)

    def update_racer(self, racer_id, speed, slip, is_alive, is_crashed=False, dt=FIXED_DT):
        """
        Updates the sound for a specific racer.
        """
        if not self.enabled:
             return

        if racer_id not in self.active_sounds:
            self.active_sounds[racer_id] = {
                'engine': None,
                'tires': None,
                'current_pitch': -1.0,
                'phase': 0.0,
                'engine_vol': 0.0,
                'tires_vol': 0.0,
                'pitch_offset': random.uniform(-0.2, 0.5)
            }
        
        data = self.active_sounds[racer_id]
        if not is_alive or is_crashed:
            if data['engine']:
                data['engine_vol'] = max(0.0, data['engine_vol'] - (dt * 3.0))
                data['engine'].set_volume(data['engine_vol'])
                if data['engine_vol'] <= 0.01:
                    data['engine'].stop()
                    data['engine'] = None
                    
            if data['tires']:
                data['tires_vol'] = max(0.0, data['tires_vol'] - (dt * 3.0)) 
                data['tires'].set_volume(data['tires_vol'])
                if data['tires_vol'] <= 0.01:
                    data['tires'].stop()
                    data['tires'] = None
            data['current_pitch'] = -1.0
            data['phase'] = 0.0
            return
        clamped_speed = min(speed, 600.0)
        base_pitch = 0.6 + (clamped_speed / 600.0) * 0.6
        
        target_pitch = clamp(base_pitch + data['pitch_offset'], 0.5, 1.5)
        target_engine_vol = 0.3 + min(speed/600.0, 0.7) * 0.4
        if self.engine_len > 0:
            data['phase'] += (dt * target_pitch) / self.engine_len
            data['phase'] %= 1.0
        
        if self.engine_array is not None:
            pitch_obj, q_pitch = self.get_pitched_data(target_pitch)
            
            if pitch_obj:
                full_snd, full_data = pitch_obj
                
                needs_update = False
                if not data['engine'] or not data['engine'].get_busy():
                    needs_update = True
                elif q_pitch != data['current_pitch']:
                    needs_update = True
                
                if needs_update:
                    offset_sound = self.get_slice_sound(pitch_obj, data['phase'])
                    
                    try:
                        if data['engine']: data['engine'].stop()
                        
                        new_channel = None
                        
                        if offset_sound:
                            new_channel = offset_sound.play()
                            if new_channel:
                                new_channel.set_volume(data['engine_vol'])
                                new_channel.queue(full_snd) 
                        else:
                            new_channel = full_snd.play(loops=-1)
                            if new_channel:
                                new_channel.set_volume(data['engine_vol'])

                        if new_channel:
                            data['engine'] = new_channel
                            data['current_pitch'] = q_pitch
                    except Exception as e:
                        pass
                diff = target_engine_vol - data['engine_vol']
                if abs(diff) > 0.01:
                    fade_speed = 2.0 if diff > 0 else 2.0
                    change = diff * fade_speed * dt
                    if abs(change) > abs(diff): change = diff
                    
                    data['engine_vol'] += change
                    
                    if data['engine']:
                        data['engine'].set_volume(data['engine_vol'])
                else:
                    data['engine_vol'] = target_engine_vol
                    if data['engine']: data['engine'].set_volume(data['engine_vol'])
        target_tires_vol = 0.0
        if self.tires_sound and slip > 0.1:
            raw_vol = min(1.0, (slip - 0.1) * 2.0)
            target_tires_vol = raw_vol * 0.6
            if not data['tires'] or not data['tires'].get_busy():
                data['tires'] = self.tires_sound.play(loops=-1, fade_ms=50)
                data['tires_vol'] = 0.0
        if data['tires']:
            diff = target_tires_vol - data['tires_vol']
            if abs(diff) > 0.01:
                fade_speed = 5.0 
                change = diff * fade_speed * dt
                if abs(change) > abs(diff): change = diff
                data['tires_vol'] += change
                data['tires'].set_volume(data['tires_vol'])
            else:
                 data['tires_vol'] = target_tires_vol
                 data['tires'].set_volume(data['tires_vol'])
            if target_tires_vol <= 0.01 and data['tires_vol'] <= 0.01:
                 data['tires'].stop()
                 data['tires'] = None
    
    def stop_all(self):
        """Stops all currently playing sounds instantly."""
        for data in self.active_sounds.values():
            if data['engine']: 
                data['engine'].stop()
                data['engine'] = None
            if data['tires']: 
                data['tires'].stop()
                data['tires'] = None
                
    def cleanup(self):
        self.stop_all()
        pygame.mixer.stop()
