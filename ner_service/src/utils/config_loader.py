import yaml
from pathlib import Path

def load_config(config_file):
    # Make sure config_file is a Path object
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
        
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config