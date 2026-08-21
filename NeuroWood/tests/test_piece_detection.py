import numpy as np

from vision.piece_detection import detect_piece, measure_piece_width


def test_piece_present():
    frame = np.full((100, 100), 40, dtype=np.uint8)
    frame[30:70, :] = 160

    present, info = detect_piece(
        frame,
        min_contrast=25,
        min_height_frac=0.10,
        sample_step=4,
    )

    assert present is True
    assert info["height_px"] > 0


def test_piece_absent():
    frame = np.full((100, 100), 120, dtype=np.uint8)

    present, _ = detect_piece(
        frame,
        min_contrast=25,
        min_height_frac=0.10,
        sample_step=4,
    )

    assert present is False


def test_width_measurement_uses_five_points_and_two_pixels_per_mm():
    frame = np.full((200, 300), 210, dtype=np.uint8)
    frame[60:140, :] = 90

    result = measure_piece_width(
        frame,
        pixels_per_mm=2.0,
        n_points=5,
        min_contrast=25,
    )

    assert result["width_valid"] is True
    assert result["measurement_points"] == 5
    assert len(result["width_measurements"]) == 5
    assert len(result["width_samples_mm"]) == 5
    assert 78 <= result["width_px"] <= 82
    assert 39 <= result["width_mm"] <= 41


def test_width_measurement_rejects_uniform_background():
    frame = np.full((200, 300), 210, dtype=np.uint8)

    result = measure_piece_width(
        frame,
        pixels_per_mm=2.0,
        n_points=5,
        min_contrast=25,
    )

    assert result["width_valid"] is False
    assert result["width_mm"] is None
    assert result["width_samples_mm"] == []
