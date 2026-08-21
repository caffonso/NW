from pathlib import Path

import cv2
import numpy as np

from hardware.camera import Camera


class FileCamera(Camera):

    def __init__(self, config):
        self.image_path = Path(config["image_path"])
        self.connected = False
        self.image = None

    def open(self):
        image = cv2.imread(
            str(self.image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not open image: {self.image_path}"
            )

        image = cv2.resize(
            image,
            (1280, 1024),
            interpolation=cv2.INTER_AREA
        )

        self.image = image
        self.connected = True

    def capture(self) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("FileCamera is not open")

        return self.image.copy()

    def close(self):
        self.connected = False

    def is_connected(self):
        return self.connected
