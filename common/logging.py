import logging

def create_logger():
    logger = logging.getLogger()
    log_format = "%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d  %(message)s"
    logging.basicConfig(level="INFO", format=log_format)
    return logger