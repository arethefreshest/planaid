"""
Centralized logging configuration for the Python service.

This module provides a consistent logging configuration across all components
of the Python service, with appropriate log levels for different environments.
"""

import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress Pydantic V2 warning about fields
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")

def setup_logger(
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a centralized logger with consistent configuration.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for logs
        log_file: Optional file to write logs to
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Default to INFO level if not specified
    if level is None:
        level = logging.INFO
        
    # Default format if not specified
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
    
    # Set specific log levels for different components
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("instructor").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Suppress specific loggers
    logging.getLogger("litellm").propagate = False
    logging.getLogger("litellm.utils").propagate = False
    logging.getLogger("litellm.proxy").propagate = False
    
    # Create and return a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    return logger

# Create a default logger instance
logger = setup_logger()

def setup_logger_for_name(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting and file output
    
    Args:
        name: The name of the logger (typically __name__ from the calling module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create file handler
    log_file = log_dir / f"python_service_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 