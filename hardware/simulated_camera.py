import time
import numpy as np
from hardware.camera import Camera

class SimulatedCamera(Camera):
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.frame_counter = 0

    def open(self):
        self.connected = True

    def capture(self):
        if not self.connected:
            raise RuntimeError("Simulated camera is not open")
        h = int(self.config.get("height", 720))
        w = int(self.config.get("width", 1280))
        image = np.full((h, w), 220, dtype=np.uint8)
        top, bottom = int(h*0.35), int(h*0.65)
        image[top:bottom, :] = 125
        if self.frame_counter % 4 == 0:
            image[top+40:top+75, int(w*0.55):int(w*0.55)+55] = 55
        self.frame_counter += 1
        time.sleep(0.01)
        return image

    def close(self):
        self.connected = False

    def is_connected(self):
        return self.connected
