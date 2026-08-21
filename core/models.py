from dataclasses import dataclass,field
from datetime import datetime
import numpy as np

@dataclass
class PieceFrame:
    image: np.ndarray
    timestamp: float
    frame_id: int

@dataclass
class Defect:
    patch_id: int
    area_px: int
    bbox: tuple|None=None

@dataclass
class BoardResult:
    board_id: str
    timestamp: datetime
    classification: str
    defects: list[Defect]=field(default_factory=list)
    processing_time_ms: float=0.0
