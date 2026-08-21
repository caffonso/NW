import time

from .piece_detection import crop_wood


def capture_piece_frames(
    source,
    n_frames=13,
    belt_speed_mm_s=600,
    capture_step_mm=225,
    speed_correction=1.0,
):
    """
    Captura patches ao longo da peça sincronizando o intervalo com a esteira.

    Retorna a mesma estrutura usada no notebook:
    frame_id, frame original, crop e geometry.
    """
    if n_frames <= 0:
        return []

    effective_speed = belt_speed_mm_s * speed_correction
    if effective_speed <= 0:
        raise ValueError("A velocidade efetiva da esteira deve ser > 0.")

    if capture_step_mm <= 0:
        raise ValueError("capture_step_mm deve ser > 0.")

    interval_s = capture_step_mm / effective_speed

    frames = []
    start_time = time.perf_counter()

    for i in range(n_frames):
        target_time = start_time + i * interval_s
        wait = target_time - time.perf_counter()

        if wait > 0:
            time.sleep(wait)

        frame = source.capture()
        crop, geometry = crop_wood(frame)

        frames.append({
            "frame_id": i,
            "frame": frame,
            "crop": crop,
            "geometry": geometry,
        })


    return frames
