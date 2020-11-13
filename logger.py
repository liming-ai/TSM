import logging

def get_logger(name):
    """Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.

    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)