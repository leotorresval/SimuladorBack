import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "..", "storage")
LAST_FILE = os.path.join(STORAGE_DIR, "last_simulation.json")

LAST_SIMULATION = None

def save_simulation(data):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(LAST_FILE, "w") as f:
        json.dump(data, f)

def load_simulation():
    if not os.path.exists(LAST_FILE):
        return None
    with open(LAST_FILE) as f:
        return json.load(f)
