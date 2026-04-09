"""
Logging configuration for federated learning experiments.
Provides a standardized logger for all modules.
"""

import logging
import sys
from pathlib import Path


def base_logger(name: str, log_level=logging.INFO):
    """
    Create and configure a logger for the given module name.
    
    Args:
        name: Module name (typically __name__)
        log_level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if this logger doesn't have them yet
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


# Module-level logger
logger = base_logger(__name__)
