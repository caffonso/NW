import time
import cv2
import numpy as np
from hardware.camera import Camera

class SimulatedCamera(Camera):
    def __init__(self, cfg):
        self.cfg = cfg
        self.connected = False
        self.i = 0

    def open(self):
        self.connected = True

    def close(self):
        self.connected = False

    def capture(self):
        if not self.connected:
            raise RuntimeError("Camera not open")

        h = int(self.cfg.get("height", 1024))
        w = int(self.cfg.get("width", 1280))
        im = np.full((h, w), 225, np.uint8)
        top, bottom = int(h * .32), int(h * .68)

        yy = np.arange(bottom-top)[:, None]
        xx = np.arange(w)[None, :]
        wood = 135 + 12*np.sin(xx/35) + 5*np.sin((xx+yy)/17)
        im[top:bottom] = np.clip(wood, 80, 190).astype(np.uint8)

        if self.i % 4 == 0:
            cv2.ellipse(
                im,
                (int(w*.35), (top+bottom)//2),
                (55, 35),
                0, 0, 360, 45, -1
            )

        if self.i % 6 == 0:
            cv2.line(
                im,
                (int(w*.68), top+25),
                (int(w*.75), bottom-25),
                40, 8
            )

        self.i += 1
        time.sleep(.01)
        return im
