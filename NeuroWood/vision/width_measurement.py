from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Formato de imagem não suportado: {image.shape}")


def _load_background(
    frame_gray: np.ndarray,
    background: np.ndarray | str | Path | None,
    background_intensity: float | None,
) -> np.ndarray:
    """Retorna uma imagem de fundo em escala de cinza com o mesmo tamanho do frame."""
    if background is not None:
        if isinstance(background, (str, Path)):
            bg = cv2.imread(str(background), cv2.IMREAD_GRAYSCALE)
            if bg is None:
                raise ValueError(f"Não foi possível abrir o fundo: {background}")
        else:
            bg = _to_gray(np.asarray(background))

        if bg.shape != frame_gray.shape:
            bg = cv2.resize(
                bg,
                (frame_gray.shape[1], frame_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return bg.astype(np.uint8)

    if background_intensity is None:
        h = frame_gray.shape[0]
        border = max(10, int(round(h * 0.08)))
        background_intensity = float(
            np.median(
                np.concatenate(
                    [
                        frame_gray[:border, :].ravel(),
                        frame_gray[-border:, :].ravel(),
                    ]
                )
            )
        )

    return np.full(
        frame_gray.shape,
        int(round(background_intensity)),
        dtype=np.uint8,
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if n_labels <= 1:
        return np.zeros_like(mask)

    # Ignora o fundo (label 0).
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))

    component = np.zeros_like(mask)
    component[labels == label] = 255
    return component


def _robust_filter(values: np.ndarray) -> np.ndarray:
    """Remove outliers com MAD; preserva pequenas variações reais de borda."""
    if values.size < 5:
        return values

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if mad < 1e-6:
        tolerance = 3.0
    else:
        sigma_robust = 1.4826 * mad
        tolerance = max(3.0, 3.5 * sigma_robust)

    keep = np.abs(values - median) <= tolerance
    filtered = values[keep]

    # Nunca descarta a maior parte das amostras por um limiar excessivamente rígido.
    if filtered.size < max(5, values.size // 2):
        return values

    return filtered


def measure_board_width(
    frame: np.ndarray,
    pixels_per_mm: float,
    *,
    background: np.ndarray | str | Path | None = None,
    background_intensity: float | None = None,
    diff_threshold: float = 25.0,
    sample_count: int = 50,
    x_margin_frac: float = 0.08,
    min_width_px: int = 20,
) -> dict[str, Any]:
    """
    Mede a largura da tábua diretamente no FRAME ORIGINAL.

    Estratégia:
      1. diferença contra fundo fixo (ou fundo estimado pelas bordas);
      2. threshold + morfologia;
      3. maior componente conectado;
      4. mede borda superior/inferior em dezenas de colunas;
      5. remove outliers por MAD;
      6. largura nominal = MEDIANA;
      7. "máxima robusta" = percentil 95, não o máximo bruto.

    A função NÃO usa crop redimensionado nem a imagem reconstruída.
    """
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm deve ser > 0.")

    gray = _to_gray(np.asarray(frame))
    h, w = gray.shape

    bg = _load_background(gray, background, background_intensity)

    diff = cv2.absdiff(gray, bg)

    # O threshold é aplicado no diff ORIGINAL para não deslocar as bordas.
    # A morfologia é usada apenas para localizar de forma robusta a tábua.
    _, raw_mask = cv2.threshold(
        diff,
        float(diff_threshold),
        255,
        cv2.THRESH_BINARY,
    )
    mask = raw_mask.copy()

    # Preenche pequenas falhas de textura e remove pontos isolados.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    board_mask = _largest_component(mask)

    if not np.any(board_mask):
        return {
            "width_mm": None,
            "width_px": None,
            "width_min_mm": None,
            "width_max_mm": None,
            "width_std_mm": None,
            "width_samples_px": [],
            "width_samples_mm": [],
            "width_valid_samples": 0,
            "width_method": "background-difference-median",
        }

    x0 = int(round(w * x_margin_frac))
    x1 = int(round(w * (1.0 - x_margin_frac)))
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))

    n = max(5, int(sample_count))
    xs = np.linspace(x0, x1 - 1, n, dtype=int)

    widths = []
    tops = []
    bottoms = []

    # Mede no threshold original para evitar a expansão/contração geométrica
    # que a morfologia pode introduzir nas bordas.
    measurement_mask = cv2.bitwise_and(raw_mask, board_mask)

    for x in xs:
        ys = np.flatnonzero(measurement_mask[:, x] > 0)
        if ys.size < 2:
            continue

        top = int(ys[0])
        bottom = int(ys[-1])
        width_px = bottom - top + 1

        if width_px >= int(min_width_px):
            widths.append(float(width_px))
            tops.append(top)
            bottoms.append(bottom)

    if not widths:
        return {
            "width_mm": None,
            "width_px": None,
            "width_min_mm": None,
            "width_max_mm": None,
            "width_std_mm": None,
            "width_samples_px": [],
            "width_samples_mm": [],
            "width_valid_samples": 0,
            "width_method": "background-difference-median",
        }

    widths_arr = _robust_filter(np.asarray(widths, dtype=np.float32))

    width_px = float(np.median(widths_arr))
    width_mm = width_px / float(pixels_per_mm)

    samples_mm = widths_arr / float(pixels_per_mm)
    width_min_mm = float(np.percentile(samples_mm, 5))
    width_max_mm = float(np.percentile(samples_mm, 95))
    width_std_mm = float(np.std(samples_mm, ddof=0))

    return {
        "width_mm": float(width_mm),
        "width_px": float(width_px),
        "width_min_mm": width_min_mm,
        "width_max_mm": width_max_mm,
        "width_std_mm": width_std_mm,
        "width_samples_px": [float(v) for v in widths_arr.tolist()],
        "width_samples_mm": [float(v) for v in samples_mm.tolist()],
        "width_valid_samples": int(widths_arr.size),
        "width_method": "background-difference-median",
        "width_diff_threshold": float(diff_threshold),
        "width_pixels_per_mm": float(pixels_per_mm),
    }
