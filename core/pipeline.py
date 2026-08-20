from datetime import datetime
from time import perf_counter
import uuid
from core.models import BoardResult
from vision.piece_detection import detect_piece
from vision.capture import capture_piece_frames
from vision.reconstruction import reconstruct_board
from vision.defects import detect_texture_defects

class NeuroWoodPipeline:
    def __init__(self, camera, plc, config, logger):
        self.camera = camera
        self.plc = plc
        self.config = config
        self.logger = logger

    def run_once(self):
        start = perf_counter()
        frame = self.camera.capture()
        present, _ = detect_piece(frame, self.config["vision"])
        if not present:
            self.logger.info("NO_PIECE_DETECTED")
            return None

        self.logger.info("PIECE_DETECTED")
        piece_frames = capture_piece_frames(
            self.camera,
            max_frames=self.config["capture"]["max_frames"],
        )
        _board = reconstruct_board(
            piece_frames,
            overlap=self.config["vision"]["overlap"],
        )
        defects = detect_texture_defects(
            piece_frames,
            sensitivity=self.config["vision"]["sensitivity"],
            min_area_px=self.config["vision"]["min_defect_area_px"],
        )
        result = BoardResult(
            board_id=f"NW-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(),
            classification="REJECT" if defects else "OK",
            defects=defects,
            processing_time_ms=(perf_counter() - start) * 1000,
        )
        self.plc.send_result(result)
        self.logger.info(
            "BOARD_PROCESSED board_id=%s class=%s defects=%d processing_ms=%.2f",
            result.board_id, result.classification,
            len(result.defects), result.processing_time_ms
        )
        return result
