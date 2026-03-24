import logging
import colorlog


def base_logger(name, level=logging.DEBUG):
    handler = logging.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        reset=True,
        style='%'
    ))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger