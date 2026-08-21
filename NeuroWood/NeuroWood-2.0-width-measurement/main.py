from core.pipeline import NeuroWoodPipeline
from vision.visualization import show_pipeline

from hardware.file_camera import FileCamera
from hardware.simulated_plc import SimulatedPLC
from storage.config import load_config
from storage.logging_config import setup_logging

config = load_config("config.yaml")
logger = setup_logging(config["logging"])
logger.info("APP_STARTED")

camera = FileCamera(config["camera"])
plc = SimulatedPLC(config["plc"])

camera.open()
plc.connect()

try:
    pipeline = NeuroWoodPipeline(camera, plc, config)
    result = pipeline.run_once()

    if pipeline.last_piece_info:
        width = pipeline.last_piece_info.get("width_mm")
        samples = pipeline.last_piece_info.get("width_samples_mm", [])
        if width is not None:
            print(f"Largura medida: {width:.2f} mm")
            print("5 medições:", [round(v, 2) for v in samples])

    if result is not None:
        show_pipeline(
            first=pipeline.last_first_frame,
            piece_frames=pipeline.last_piece_frames,
            board=pipeline.last_board_marked,
            detection_results=pipeline.last_detection_results,
            summary=pipeline.last_summary,
            result=result,
        )
finally:
    camera.close()
    plc.disconnect()
    logger.info("APP_STOPPED")
