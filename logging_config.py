import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler for general logs
            logging.FileHandler(
                os.path.join(logs_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')
            ),
            # File handler for ML-specific logs
            logging.FileHandler(
                os.path.join(logs_dir, f'ml_models_{datetime.now().strftime("%Y%m%d")}.log')
            ),
            # Console handler
            logging.StreamHandler()
        ]
    )

    # Create specific loggers
    ml_logger = logging.getLogger('ml_models')
    ml_logger.setLevel(logging.DEBUG)

    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)

    return ml_logger, app_logger
