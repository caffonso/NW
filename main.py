from core.pipeline import NeuroWoodPipeline
from hardware.simulated_camera import SimulatedCamera
from hardware.simulated_plc import SimulatedPLC
from storage.config import load_config
from storage.logging_config import setup_logging

def main():
    config = load_config("config.yaml")
    logger = setup_logging(config["logging"])
    logger.info("APP_STARTED")

    camera = SimulatedCamera(config["camera"])
    plc = SimulatedPLC(config["plc"])
    pipeline = NeuroWoodPipeline(camera, plc, config, logger)

    camera.open()
    plc.connect()
    try:
        result = pipeline.run_once()
        print(result)
    finally:
        camera.close()
        plc.disconnect()
        logger.info("APP_STOPPED")

if __name__ == "__main__":
    main()
