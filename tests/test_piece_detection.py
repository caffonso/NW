import numpy as np
from vision.piece_detection import detect_piece

def test_piece_present():
    present, _ = detect_piece(np.full((100,100), 120, dtype=np.uint8), {})
    assert present is True

def test_piece_absent():
    present, _ = detect_piece(np.full((100,100), 240, dtype=np.uint8), {})
    assert present is False
