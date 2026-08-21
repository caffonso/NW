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


def crop_wood(frame, band=5, target_size=(256 * 2, 112 * 2)):
    """
    Detecta as bordas superior e inferior da madeira e retorna o crop.

    O target_size mantém exatamente o valor usado no notebook atual:
    largura=512 px e altura=224 px.
    """
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
