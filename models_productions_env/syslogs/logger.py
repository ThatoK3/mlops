import os
import sys
from datetime import datetime

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def setup_logging(logs_dir="syslogs/logs"):
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    log_path = os.path.join(logs_dir, log_filename)
    log_file = open(log_path, "w")

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"[Logger initialized] Logging to: {log_path}")


#from syslogs.logger import setup_logging

#setup_logging()

