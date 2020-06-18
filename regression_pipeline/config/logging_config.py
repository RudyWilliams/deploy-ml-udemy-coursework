import logging
import sys

# Multiple calls to logging.getLogger('someLogger') return a
# reference to the same logger obj. This is true within the same
# module and also across modules as long as the same python
# interpreter process is being used

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler
