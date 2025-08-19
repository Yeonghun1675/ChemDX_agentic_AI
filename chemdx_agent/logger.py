import logging


def get_logger(name, stream=True, file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if stream:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)

    if file:
        handler = logging.FileHandler(f"{name}.log")
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)

    return logger


logger = get_logger(__name__, stream=True, file=False)
error_logger = get_logger('error', stream=False, file=True)