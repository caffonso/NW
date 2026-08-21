import inspect
import logging
import time
import uuid
from datetime import datetime

from core.acceptance import evaluate_acceptance, load_acceptance_config
from core.models import BoardResult, Defect
from vision.capture import capture_piece_frames
from vision.defects import detect_texture_defects
from vision.piece_detection import detect_piece, measure_piece_width
from vision.reconstruction import reconstruct_board

logger = logging.getLogger(__name__)


def _cfg(config, path, default):
    """Lê configuração tanto de dicts quanto de objetos com atributos."""
    current = config

    for part in path.split("."):
        if current is None:
            return default

        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)

    return current


def _make_board_result(**kwargs):
    """Mantém compatibilidade com BoardResult com/sem model_version."""
    try:
        params = inspect.signature(BoardResult).parameters
    except (TypeError, ValueError):
        params = {}

    if "model_version" not in params:
        kwargs.pop("model_version", None)

    return BoardResult(**kwargs)


def _measure_board_width(
    piece_frames,
    min_contrast,
    pixels_per_mm=10.0,
):
    """
    Mede a largura uma vez em cada patch/frame e usa a maior medida válida.

    Cada medição é feita no centro horizontal do frame original. Isso preserva
    a calibração física da câmera (2 px = 1 mm) e evita medir sobre o crop
    redimensionado usado pelo detector de textura.
    """
    measurements = []

    for patch in piece_frames:
        frame_id = patch.get("frame_id")
        width_info = measure_piece_width(
            patch["frame"],
            pixels_per_mm=pixels_per_mm,
            n_points=1,
            min_contrast=min_contrast,
        )

        measurement = {
            "frame_id": frame_id,
            "valid": bool(width_info.get("width_valid", False)),
            "width_mm": width_info.get("width_mm"),
            "width_px": width_info.get("width_px"),
            "top": width_info.get("width_top_px"),
            "bottom": width_info.get("width_bottom_px"),
        }
        measurements.append(measurement)

    valid = [m for m in measurements if m["valid"]]

    if not valid:
        return {
            "width_valid": False,
            "width_mm": None,
            "width_px": None,
            "width_samples_mm": [],
            "width_samples_px": [],
            "width_measurements": measurements,
            "width_top_px": None,
            "width_bottom_px": None,
            "pixels_per_mm": float(pixels_per_mm),
            "measurement_points": 1,
            "measurement_patches": len(piece_frames),
            "width_measurement_frame_id": None,
            "width_aggregation": "max",
        }

    largest = max(valid, key=lambda m: float(m["width_mm"]))

    return {
        "width_valid": True,
        "width_mm": float(largest["width_mm"]),
        "width_px": float(largest["width_px"]),
        "width_samples_mm": [
            float(m["width_mm"])
            for m in valid
        ],
        "width_samples_px": [
            float(m["width_px"])
            for m in valid
        ],
        "width_measurements": measurements,
        "width_top_px": largest["top"],
        "width_bottom_px": largest["bottom"],
        "pixels_per_mm": float(pixels_per_mm),
        "measurement_points": 1,
        "measurement_patches": len(piece_frames),
        "width_measurement_frame_id": largest["frame_id"],
        "width_aggregation": "max",
    }


class NeuroWoodPipeline:
    """Orquestra uma inspeção da peça."""

    def __init__(self, camera, plc, config):
        self.camera = camera
        self.plc = plc
        self.config = config

        self.last_first_frame = None
        self.last_piece_info = None
        self.last_piece_frames = None
        self.last_detection_results = None
        self.last_summary = None
        self.last_board = None
        self.last_board_marked = None
        self.last_result = None
        self.last_acceptance = None

    def run_once(self):
        t0 = time.perf_counter()

        first_frame = self.camera.capture()
        self.last_first_frame = first_frame

        piece_min_contrast = _cfg(
            self.config,
            "vision.piece_min_contrast",
            25,
        )

        present, piece_info = detect_piece(
            first_frame,
            min_contrast=piece_min_contrast,
            min_height_frac=_cfg(
                self.config,
                "vision.piece_min_height_frac",
                0.10,
            ),
            sample_step=_cfg(
                self.config,
                "vision.piece_sample_step",
                8,
            ),
        )
        self.last_piece_info = piece_info

        if not present:
            return None

        logger.info("PIECE_DETECTED")

        piece_frames = capture_piece_frames(
            self.camera,
            n_frames=_cfg(
                self.config,
                "capture.max_frames",
                _cfg(self.config, "capture.n_frames", 13),
            ),
            belt_speed_mm_s=_cfg(
                self.config,
                "conveyor.nominal_speed_mm_s",
                600,
            ),
            capture_step_mm=_cfg(
                self.config,
                "capture.frame_spacing_mm",
                _cfg(self.config, "capture.capture_step_mm", 225),
            ),
            speed_correction=_cfg(
                self.config,
                "conveyor.speed_factor",
                _cfg(self.config, "conveyor.speed_correction", 1.0),
            ),
        )
        self.last_piece_frames = piece_frames
        logger.info("CAPTURE_COMPLETE")

        # Largura: uma medição central em cada patch/frame.
        # A largura final da tábua é a MAIOR medida válida dos 13 patches.
        # A calibração permanece 2 px = 1 mm.
        if piece_frames:
            width_info = _measure_board_width(
                piece_frames,
                min_contrast=piece_min_contrast,
                pixels_per_mm=2.0,
            )
            piece_info.update(width_info)
            self.last_piece_info = piece_info

            if width_info["width_valid"]:
                logger.info(
                    "BOARD_WIDTH_MAX width_mm=%.2f width_px=%.1f "
                    "samples_mm=%s max_frame_id=%s",
                    width_info["width_mm"],
                    width_info["width_px"],
                    [
                        round(v, 2)
                        for v in width_info["width_samples_mm"]
                    ],
                    width_info["width_measurement_frame_id"],
                )
            else:
                logger.warning(
                    "BOARD_WIDTH_NOT_MEASURED patches=%d",
                    len(piece_frames),
                )

        detection_results, summary = detect_texture_defects(
            piece_frames,
            min_area_frac=_cfg(
                self.config,
                "vision.min_area_frac",
                0.003,
            ),
            min_width_frac=_cfg(
                self.config,
                "vision.min_width_frac",
                0.08,
            ),
            rigor=_cfg(
                self.config,
                "vision.rigor",
                1.5,
            ),
        )
        self.last_detection_results = detection_results

        # Critério de aceitação por quantidade e tamanho dos defeitos.
        acceptance_cfg = load_acceptance_config(self.config)
        classification, acceptance_details = evaluate_acceptance(
            summary,
            acceptance_cfg,
        )
        self.last_acceptance = acceptance_details
        self.last_summary = summary

        board_marked = reconstruct_board(
            detection_results,
            use_marked=True,
        )
        board = reconstruct_board(
            detection_results,
            use_marked=False,
        )
        self.last_board_marked = board_marked
        self.last_board = board
        logger.info("BOARD_RECONSTRUCTED")

        defects = []
        for patch in summary["patches"]:
            patch_id = patch["frame_id"]
            for defect in patch["defects"]:
                defects.append(
                    Defect(
                        patch_id=patch_id,
                        area_px=defect["area_pixels"],
                        bbox=defect["bbox"],
                    )
                )

        processing_time_ms = (time.perf_counter() - t0) * 1000.0

        result = _make_board_result(
            board_id=f"NW-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(),
            classification=classification,
            defects=defects,
            processing_time_ms=processing_time_ms,
            model_version="texture-notebook",
        )

        self.last_result = result

        counts = acceptance_details["counts"]
        limits = acceptance_details["limits"]
        logger.info(
            "BOARD_PROCESSED board_id=%s class=%s defects=%d "
            "small=%d/%d medium=%d/%d large=%d/%d processing_ms=%.2f",
            result.board_id,
            result.classification,
            len(result.defects),
            counts["small"],
            limits["small"],
            counts["medium"],
            limits["medium"],
            counts["large"],
            limits["large"],
            result.processing_time_ms,
        )

        if self.plc is not None:
            self.plc.send_result(result)
            logger.info("RESULT_SENT")

        return result
