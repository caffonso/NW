import cv2
import numpy as np


def detect_piece(
    frame,
    min_contrast=25,
    min_height_frac=0.10,
    sample_step=8,
):
    """Detecta a presença da madeira usando o perfil vertical da imagem."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    h, w = gray.shape

    x1 = int(w * 0.20)
    x2 = int(w * 0.80)
    roi = gray[:, x1:x2:sample_step]

    profile = np.median(roi, axis=1).astype(np.float32)
    profile = cv2.GaussianBlur(profile[:, None], (1, 21), 0).ravel()

    border = max(10, int(h * 0.10))
    background = np.median(
        np.concatenate([
            profile[:border],
            profile[-border:],
        ])
    )

    threshold = background + min_contrast
    wood_mask = profile > threshold
    indices = np.where(wood_mask)[0]

    if len(indices) == 0:
        return False, {
            "top": None,
            "bottom": None,
            "contrast": 0.0,
            "background_intensity": float(background),
        }

    groups = np.split(
        indices,
        np.where(np.diff(indices) > 1)[0] + 1,
    )
    largest = max(groups, key=len)

    top = int(largest[0])
    bottom = int(largest[-1])
    height_px = bottom - top + 1

    wood_intensity = float(np.median(profile[top:bottom + 1]))
    contrast = wood_intensity - background
    min_height = h * min_height_frac

    present = (
        height_px >= min_height
        and contrast >= min_contrast
    )

    return present, {
        "top": top,
        "bottom": bottom,
        "height_px": height_px,
        "background_intensity": float(background),
        "wood_intensity": wood_intensity,
        "contrast": float(contrast),
    }


def _largest_true_segment(mask):
    """Retorna o maior segmento True contíguo de uma máscara 1D."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None

    groups = np.split(
        indices,
        np.where(np.diff(indices) > 1)[0] + 1,
    )
    largest = max(groups, key=len)
    return int(largest[0]), int(largest[-1])


def measure_piece_width(
    frame,
    pixels_per_mm=2.0,
    n_points=5,
    min_contrast=25,
    band=5,
    background_band_frac=0.10,
    min_width_frac=0.10,
):
    """
    Mede a largura da tábua na imagem original usando diferença contra fundo fixo.

    A medição é feita em n_points posições horizontais:
    1. usa faixas fixas no topo e na base como referência da esteira;
    2. calcula diferença absoluta para o fundo;
    3. encontra o maior segmento diferente do fundo;
    4. usa início/fim como bordas superior/inferior.

    A largura final é a mediana das medidas válidas.
    Calibração padrão: 2 pixels = 1 mm.
    """
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm deve ser > 0.")
    if n_points <= 0:
        raise ValueError("n_points deve ser > 0.")

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    h, w = gray.shape
    if h < 20 or w < 20:
        raise ValueError(f"Frame muito pequeno para medição: {gray.shape}")

    background_band = max(5, int(round(h * background_band_frac)))
    background_band = min(background_band, max(5, h // 4))

    x_start = int(round(w * 0.20))
    x_end = int(round(w * 0.80))

    # Para uma única medição, usa exatamente o centro horizontal do frame.
    # Isso é usado pelo pipeline para obter uma medida por patch.
    if n_points == 1:
        x_positions = np.asarray([w // 2], dtype=int)
    else:
        x_positions = np.linspace(
            x_start,
            x_end,
            n_points,
        ).round().astype(int)

    measurements = []

    for x in x_positions:
        x = int(np.clip(x, 0, w - 1))
        x1 = max(0, x - band)
        x2 = min(w, x + band + 1)

        profile = np.median(
            gray[:, x1:x2],
            axis=1,
        ).astype(np.float32)

        background_samples = np.concatenate([
            profile[:background_band],
            profile[-background_band:],
        ])
        background = float(np.median(background_samples))

        difference = np.abs(profile - background)
        wood_mask = difference >= float(min_contrast)

        mask_image = (wood_mask.astype(np.uint8) * 255)[:, None]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
        wood_mask = mask_image[:, 0] > 0

        segment = _largest_true_segment(wood_mask)
        if segment is None:
            measurements.append({
                "x": x,
                "valid": False,
                "top": None,
                "bottom": None,
                "width_px": None,
                "width_mm": None,
                "background_intensity": background,
                "contrast": 0.0,
            })
            continue

        top, bottom = segment
        width_px = bottom - top + 1
        contrast = float(np.median(difference[top:bottom + 1]))
        valid = (
            width_px >= h * min_width_frac
            and contrast >= min_contrast
        )

        measurements.append({
            "x": x,
            "valid": bool(valid),
            "top": top if valid else None,
            "bottom": bottom if valid else None,
            "width_px": float(width_px) if valid else None,
            "width_mm": float(width_px / pixels_per_mm) if valid else None,
            "background_intensity": background,
            "contrast": contrast,
        })

    valid_measurements = [m for m in measurements if m["valid"]]

    if not valid_measurements:
        return {
            "width_valid": False,
            "width_mm": None,
            "width_px": None,
            "width_samples_mm": [],
            "width_samples_px": [],
            "width_measurements": measurements,
            "width_top_px": None,
            "width_bottom_px": None,
            "pixels_per_mm": float(pixels_per_mm),
            "measurement_points": int(n_points),
        }

    widths_px = np.asarray(
        [m["width_px"] for m in valid_measurements],
        dtype=np.float32,
    )
    widths_mm = widths_px / float(pixels_per_mm)
    tops = np.asarray([m["top"] for m in valid_measurements], dtype=np.float32)
    bottoms = np.asarray([m["bottom"] for m in valid_measurements], dtype=np.float32)

    return {
        "width_valid": True,
        "width_mm": float(np.median(widths_mm)),
        "width_px": float(np.median(widths_px)),
        "width_samples_mm": [float(v) for v in widths_mm],
        "width_samples_px": [float(v) for v in widths_px],
        "width_measurements": measurements,
        "width_top_px": int(round(float(np.median(tops)))),
        "width_bottom_px": int(round(float(np.median(bottoms)))),
        "pixels_per_mm": float(pixels_per_mm),
        "measurement_points": int(n_points),
    }


def crop_wood(frame, band=5, target_size=(256 * 2, 112 * 2)):
    """Detecta bordas superior/inferior da madeira e retorna crop 512x224."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    h, w = gray.shape

    def detect_edges(x):
        x1 = max(0, x - band)
        x2 = min(w, x + band + 1)

        profile = np.median(
            gray[:, x1:x2],
            axis=1,
        ).astype(np.float32)

        profile = cv2.GaussianBlur(
            profile[:, None],
            (1, 21),
            0,
        ).ravel()

        gradient = np.gradient(profile)

        top = np.argmax(
            gradient[:int(h * 0.7)]
        )

        bottom = (
            top + 20
            + np.argmin(gradient[top + 20:])
        )

        return top, bottom

    top1, bottom1 = detect_edges(int(w * 0.25))
    top2, bottom2 = detect_edges(int(w * 0.75))

    top = int((top1 + top2) / 2)
    bottom = int((bottom1 + bottom2) / 2)

    if bottom <= top:
        raise ValueError(
            f"Crop inválido: top={top}, bottom={bottom}. "
            "Verifique iluminação/contraste da imagem."
        )

    crop = frame[top:bottom, :]
    crop = cv2.resize(
        crop,
        target_size,
        interpolation=cv2.INTER_AREA,
    )

    return crop, {
        "top": top,
        "bottom": bottom,
        "width_px": bottom - top,
    }


def ensure_bgr(frame):
    """Garante que a imagem tenha 3 canais BGR."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame

    raise ValueError(f"Formato de imagem não suportado: {frame.shape}")


def check_image_quality(
    frame,
    min_brightness=50,
    max_brightness=220,
    min_contrast=15,
    min_sharpness=100,
):
    """Validação básica de brilho, contraste e nitidez do frame."""
    frame = ensure_bgr(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()
    contrast = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    ok = (
        min_brightness <= brightness <= max_brightness
        and contrast >= min_contrast
        and sharpness >= min_sharpness
    )

    return {
        "ok": bool(ok),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "sharpness": float(sharpness),
    }
