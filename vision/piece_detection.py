def detect_piece(frame, config):
    if frame is None or frame.size == 0:
        return False, {}
    mean_value = float(frame.mean())
    return mean_value < 210.0, {"mean_intensity": mean_value}
