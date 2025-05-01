import logging
import os
from pathlib import Path

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)  # Set the level here
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    try:
        # Try to create logs directory in /app/logs first (Docker)
        logs_dir = Path('/app/logs')
        if not logs_dir.exists():
            # Fall back to local logs directory
            logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(logs_dir / f'{name}.log')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to setup file logging: {str(e)}")
    
    # Ensure logs propagate to root logger
    logger.propagate = True
    
    return logger