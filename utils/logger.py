import logging
import sys
from datetime import datetime
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


class Logger:
    def __init__(self, name, tqdm_handler=True):
        # Get logger and remove any existing handlers to prevent duplicates
        self.logger = logging.getLogger(name)
        self.logger.handlers = []

        # Prevent propagation to root logger to avoid duplicate logs
        self.logger.propagate = False

        # Set level
        self.logger.setLevel(logging.INFO)

        # Create formatter
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Add tqdm handler
        if tqdm_handler:
            self.tqdm_handler = TqdmLoggingHandler()
            self.tqdm_handler.setFormatter(log_format)
            self.logger.addHandler(self.tqdm_handler)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def critical(self, msg):
        self.logger.critical(msg)
