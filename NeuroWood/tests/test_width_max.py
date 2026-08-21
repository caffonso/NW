import importlib.util
import sys
import types
from pathlib import Path


def _load_pipeline_module():
    core_models = types.ModuleType("core.models")
    core_models.BoardResult = object
    core_models.Defect = object
    sys.modules["core.models"] = core_models

    acceptance = types.ModuleType("core.acceptance")
    acceptance.evaluate_acceptance = lambda *a, **k: ("ACCEPT", {})
    acceptance.load_acceptance_config = lambda *a, **k: {}
    sys.modules["core.acceptance"] = acceptance

    capture = types.ModuleType("vision.capture")
    capture.capture_piece_frames = lambda *a, **k: []
    sys.modules["vision.capture"] = capture

    defects = types.ModuleType("vision.defects")
    defects.detect_texture_defects = lambda *a, **k: ([], {})
    sys.modules["vision.defects"] = defects

    reconstruction = types.ModuleType("vision.reconstruction")
    reconstruction.reconstruct_board = lambda *a, **k: None
    sys.modules["vision.reconstruction"] = reconstruction

    piece_detection = types.ModuleType("vision.piece_detection")
    piece_detection.detect_piece = lambda *a, **k: (False, {})
    piece_detection.measure_piece_width = lambda *a, **k: {}
    sys.modules["vision.piece_detection"] = piece_detection

    path = Path(__file__).parents[1] / "core" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_board_width_uses_maximum_patch_measurement():
    module = _load_pipeline_module()

    values = iter([100.0, 104.0, 98.5, 112.0, 109.0])

    def fake_measure(frame, pixels_per_mm, n_points, min_contrast):
        value = next(values)
        return {
            "width_valid": True,
            "width_mm": value,
            "width_px": value * pixels_per_mm,
            "width_top_px": 100,
            "width_bottom_px": int(100 + value * pixels_per_mm - 1),
        }

    module.measure_piece_width = fake_measure

    frames = [{"frame_id": i, "frame": object()} for i in range(5)]

    result = module._measure_board_width(
        frames,
        min_contrast=25,
        pixels_per_mm=2.0,
    )

    assert result["width_mm"] == 112.0
    assert result["width_px"] == 224.0
    assert result["width_measurement_frame_id"] == 3
    assert result["width_samples_mm"] == [100.0, 104.0, 98.5, 112.0, 109.0]
    assert result["measurement_points"] == 1
    assert result["measurement_patches"] == 5
    assert result["width_aggregation"] == "max"


def test_invalid_patch_measurements_are_ignored():
    module = _load_pipeline_module()

    results = iter([
        {
            "width_valid": False,
            "width_mm": None,
            "width_px": None,
            "width_top_px": None,
            "width_bottom_px": None,
        },
        {
            "width_valid": True,
            "width_mm": 120.0,
            "width_px": 240.0,
            "width_top_px": 100,
            "width_bottom_px": 339,
        },
        {
            "width_valid": True,
            "width_mm": 118.0,
            "width_px": 236.0,
            "width_top_px": 102,
            "width_bottom_px": 337,
        },
    ])

    module.measure_piece_width = lambda *a, **k: next(results)

    frames = [{"frame_id": i, "frame": object()} for i in range(3)]

    result = module._measure_board_width(
        frames,
        min_contrast=25,
        pixels_per_mm=2.0,
    )

    assert result["width_mm"] == 120.0
    assert result["width_measurement_frame_id"] == 1
    assert result["width_samples_mm"] == [120.0, 118.0]
