import logging
import sys

# ── Format ─────────────────────────────────────────────
LOG_FORMAT = (
    "%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level=logging.DEBUG):
    # Root handler → stderr, coloured by level
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        ColouredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    )
    logging.basicConfig(level=level, handlers=[handler])

    # Silence noisy third-party loggers
    for noisy in ["vosk", "alsa", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── ANSI colour wrapper ──────────────────────────────────
COLOURS = {
    "DEBUG":    "\033[90m",   # dark grey
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # amber
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[1;31m", # bold red
}
RESET = "\033[0m"

class ColouredFormatter(logging.Formatter):
    def format(self, record):
        colour = COLOURS.get(record.levelname, "")
        record.levelname = (
            f"{colour}{record.levelname}{RESET}"
        )
        return super().format(record)