import json
import os

class HighscoreManager:
    def __init__(self, tracks_dir):
        self.tracks_dir = tracks_dir
        self.highscores_dir = os.path.join(tracks_dir, 'highscores')
        os.makedirs(self.highscores_dir, exist_ok=True)
        self.cache = {}

    def get_highscore_file(self, track_name):
        return os.path.join(self.highscores_dir, f"{track_name}_highscores.json")

    def load_highscores(self, track_name):
        if track_name in self.cache:
            return self.cache[track_name]

        file_path = self.get_highscore_file(track_name)
        data = {
            "mode": "time_trial",
            "global_best": None,
            "agent_bests": {}
        }

        if os.path.exists(file_path):
            load_path = file_path
        else:
            legacy_path = os.path.join(self.tracks_dir, f"{track_name}_highscores.json")
            load_path = legacy_path if os.path.exists(legacy_path) else None

        if load_path:
            try:
                with open(load_path, 'r') as f:
                    loaded = json.load(f)
                    data.update(loaded)
                if load_path != file_path:
                    self.cache[track_name] = data
                    self.save_highscores(track_name)
                    return data
            except Exception as e:
                print(f"Error loading highscores for {track_name}: {e}")

        self.cache[track_name] = data
        return data

    def save_highscores(self, track_name):
        if track_name not in self.cache:
            return
            
        file_path = self.get_highscore_file(track_name)
        try:
            with open(file_path, 'w') as f:
                json.dump(self.cache[track_name], f, indent=4)
        except Exception as e:
            print(f"Error saving highscores for {track_name}: {e}")

    def update_highscore(self, track_name, agent_name, score, metrics=None, mode='time_trial'):
        """
        Updates the highscore if improved.
        mode='time_trial': Lower score is better.
        mode='survival': Higher score is better.
        
        Returns: (is_new_global, is_new_personal)
        """
        data = self.load_highscores(track_name)
        data['mode'] = mode
        
        is_new_global = False
        is_new_personal = False
        current_pb = data['agent_bests'].get(agent_name)
        
        better_than_pb = False
        if current_pb is None:
            better_than_pb = True
        else:
            if mode == 'time_trial':
                if score < current_pb['score']:
                    better_than_pb = True
            else:
                if score > current_pb['score']:
                    better_than_pb = True
                elif score == current_pb['score']:
                    prev_time = current_pb.get('metrics', {}).get('time', 0)
                    curr_time = metrics.get('time', 0) if metrics else 0
                    if curr_time > prev_time:
                        better_than_pb = True

        if better_than_pb:
            data['agent_bests'][agent_name] = {
                'score': score,
                'metrics': metrics
            }
            is_new_personal = True
        current_gb = data.get('global_best')
        better_than_gb = False
        
        if current_gb is None:
            better_than_gb = True
        else:
            if mode == 'time_trial':
                if score < current_gb['score']:
                    better_than_gb = True
            else:
                if score > current_gb['score']:
                    better_than_gb = True
                elif score == current_gb['score']:
                     prev_time = current_gb.get('metrics', {}).get('time', 0)
                     curr_time = metrics.get('time', 0) if metrics else 0
                     if curr_time > prev_time:
                         better_than_gb = True
                         
        if better_than_gb:
            data['global_best'] = {
                'score': score,
                'agent': agent_name,
                'metrics': metrics
            }
            is_new_global = True
            
        if is_new_personal or is_new_global:
            self.save_highscores(track_name)
            
        return is_new_global, is_new_personal
