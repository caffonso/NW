import logging
from pathlib import Path

def setup_logging(config):
    level = getattr(logging, str(config.get("level", "INFO")).upper(), logging.INFO)
    log_file = Path(config.get("file", "logs/neurowood.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("neurowood")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
