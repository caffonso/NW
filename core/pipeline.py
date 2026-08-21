import inspect
import logging
import time
import uuid
from datetime import datetime

from core.models import BoardResult, Defect
from vision.capture import capture_piece_frames
from vision.defects import detect_texture_defects
from vision.piece_detection import detect_piece
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


class NeuroWoodPipeline:
    """
    Orquestra uma inspeção usando as funções validadas no notebook.

    A interface externa permanece simples: run_once() retorna BoardResult ou None.
    As estruturas ricas do notebook ficam disponíveis em atributos last_*.
    """

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

    def run_once(self):
        t0 = time.perf_counter()

        first_frame = self.camera.capture()
        self.last_first_frame = first_frame

        present, piece_info = detect_piece(
            first_frame,
            min_contrast=_cfg(
                self.config,
                "vision.piece_min_contrast",
                25,
            ),
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
                        bbox=defect["bbox"],)
                    
                )

        classification = "REJECT" if defects else "ACCEPT"
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

        logger.info(
            "BOARD_PROCESSED board_id=%s class=%s defects=%d processing_ms=%.2f",
            result.board_id,
            result.classification,
            len(result.defects),
            result.processing_time_ms,
        )

        if self.plc is not None:
            self.plc.send_result(result)
            logger.info("RESULT_SENT")

        return result
