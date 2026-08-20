import cv2
import numpy as np
from core.models import Defect

def detect_texture_defects(piece_frames, sensitivity=0.65, min_area_px=40):
    defects = []
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    threshold = int(120 - 70*sensitivity)

    for pf in piece_frames:
        gray = pf.image if pf.image.ndim == 2 else cv2.cvtColor(pf.image, cv2.COLOR_BGR2GRAY)
        mask = (gray < threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area >= min_area_px:
                defects.append(Defect(patch_id=pf.frame_id, area_px=area))
    return defects
