import logging
import time
from functools import wraps

def setup_logging(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'):
    logging.basicConfig(level=level, format=format)

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Starting {func.__name__} with parameters: {args}, {kwargs}")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            raise
        end_time = time.time()
        logging.info(f"Completed {func.__name__} in {end_time - start_time:.2f} seconds with result: {result}")
        return result
    return wrapper
