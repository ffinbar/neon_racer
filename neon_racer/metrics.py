
import json
import time
import os

class MetricsLogger:
    def __init__(self, filename='training_log.jsonl'):
        self.filename = filename
        self.file = open(filename, 'w')
        self.last_flush_time = time.time()
        self.flush_interval = 0.5  # seconds 
        print(f"[*] Logging metrics to {os.path.abspath(filename)}")

    def log(self, metrics_type, data):
        """
        Log a data point.
        metrics_type: 'neat' or 'rl'
        data: dict containing the metrics
        """
        entry = {
            'timestamp': time.time(),
            'type': metrics_type,
            'data': data
        }
        self.file.write(json.dumps(entry) + '\n')
        
        current_time = time.time()
        if current_time - self.last_flush_time >= self.flush_interval:
            self.file.flush()
            self.last_flush_time = current_time

    def close(self):
        self.file.flush()
        self.file.close()
