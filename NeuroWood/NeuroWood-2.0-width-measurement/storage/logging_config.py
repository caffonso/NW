import logging
from pathlib import Path

def setup_logging(cfg):
    p = Path(cfg.get("file", "logs/neurowood.log"))
    p.parent.mkdir(exist_ok=True)

    log = logging.getLogger("neurowood")
    log.setLevel(logging.INFO)

    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for h in [
            logging.FileHandler(p, encoding="utf-8"),
            logging.StreamHandler(),
        ]:
            h.setFormatter(fmt)
            log.addHandler(h)

    return log
