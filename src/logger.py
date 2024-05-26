# refer to https://docs.python.org/zh-cn/3/library/logging.html for further improvements
# TODO
# - [ ] multiple logger
# - [ ] synchronize logger across processes
import logging
from typing import Optional
from termcolor import colored
import sys
import os

logger_initialized = {}


class _ColorfulFormatter(logging.Formatter):

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "magenta")
        elif record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(
    name: Optional[str] = None,
    output_dir: Optional[str] = '../log',
    rank: int = 0,
    log_level: int = logging.WARN,
    log_name: Optional[str] = 'Template',
    color: bool = True,
) -> logging.Logger:
    """Initialize the logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, only the logger of the master
    process is added console handler. If ``output_dir`` is specified, all loggers
    will be added file handler.

    Args:
        name (str): Logger name. Defaults to None to setup root logger.
        output_dir (str): The directory to save log.
        rank (int): Process rank in the distributed training. Defaults to 0.
        log_level (int): Verbosity level of the logger. Defaults to ``logging.INFO``.
        log_name (str): log file name. Defaults to 'Template'.
        color (bool): If True, color the output. Defaults to True.

    Returns:
        logging.Logger: A initialized logger.

    Examples:
        setup_logger("cpu", output_dir=self.work_dir, rank=get_rank())
    """
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger will not be propagated to its parent
    logger.propagate = False

    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s]: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )

    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter if color else formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, log_name + f"{rank}.log")
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger_initialized[name] = logger
    return logger
