import logging
import os

from rich.logging import RichHandler
from licenseplate_ocr.utils.config import LOGFILE

class LicensePlateLogger:
    def __init__(self, log_name: str = "LicensePlate") -> None:
        """
        """
        self.__log_name = log_name

    @staticmethod
    def __make_console_logger(logger) -> None:
        rich_handler = RichHandler(rich_tracebacks=True, markup=True)
        rich_fmt = "[%(name)s] - %(message)s"
        rich_handler.setFormatter(logging.Formatter(rich_fmt))
        rich_handler.setLevel(logging.DEBUG)

        logger.addHandler(rich_handler)

    @staticmethod
    def __make_file_logger(logger) -> None:
        log_file = LOGFILE["path"]
        if not os.path.isfile(log_file) and os.path.dirname(log_file) != "":
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_fmt = "[%(asctime)s] [%(levelname).1s] %(message)s"
        file_handler.setFormatter(logging.Formatter(file_fmt))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    def get_instance(self, gen_log: bool = False) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(self.__log_name)
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            logger.handlers.clear()

        self.__make_console_logger(logger)

        if gen_log:
            self.__make_file_logger(logger)

        return logger