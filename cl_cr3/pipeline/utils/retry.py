import time
import logging
from functools import wraps

def retry(retries=3, delay=1, backoff=2):
    def decorator_retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _retries, _delay = retries, delay
            while _retries > 0:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Retrying {func.__name__} due to {e}, {_retries} retries left...")
                    _retries -= 1
                    time.sleep(_delay)
                    _delay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator_retry
