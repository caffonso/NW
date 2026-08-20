import time
from core.models import PieceFrame

def capture_piece_frames(camera, max_frames=13):
    frames = []
    for frame_id in range(max_frames):
        frames.append(PieceFrame(
            image=camera.capture(),
            timestamp=time.time(),
            frame_id=frame_id,
        ))
    return frames
