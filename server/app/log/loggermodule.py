import logging
import os
from typing import Optional


class LoggerFactory:
    """
    Factory tạo logger cho từng module
    """

    LOG_DIR = "logs"

    @staticmethod
    def get_logger(log_file: Optional[str] = None) -> logging.Logger:
        logger = logging.getLogger("default_logger")
        logger.setLevel(logging.DEBUG)

        # Nếu logger đã có handler riêng thì không add nữa
        if logger.handlers:
            return logger

        # Nếu có log file -> ghi vào thư mục logs/
        if log_file:
            os.makedirs(LoggerFactory.LOG_DIR, exist_ok=True)

            log_path = os.path.join(LoggerFactory.LOG_DIR, log_file)

            file_handler = logging.FileHandler(log_path, encoding="utf-8")

            logger.addHandler(file_handler)

            # Không cho propagate để tránh duplicate log
            logger.propagate = False

        return logger