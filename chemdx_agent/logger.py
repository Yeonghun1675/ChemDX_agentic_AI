import logging
import os


def get_logger(name, stream=True, file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if stream:
        has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        if not has_stream:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)

    if file:
        file_name = f"log.txt"
        if os.path.exists(file_name):
            os.remove(file_name)
        handler = logging.FileHandler(file_name)
        # handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    return logger


logger = get_logger(__name__, stream=True, file=True)