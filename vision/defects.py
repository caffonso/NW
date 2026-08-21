import cv2
import numpy as np


def detect_texture_defects(
    piece_frames,
    min_area_frac=0.003,
    min_width_frac=0.08,
    rigor=1.5,
):
    """
    Detecta defeitos de textura em todos os patches de uma peça.

    rigor:
        < 1.0 -> mais sensível
        = 1.0 -> padrão
        > 1.0 -> mais conservador

    Retorna
    -------
    results : list[dict]
        Informações detalhadas de cada patch, incluindo máscara e imagem marcada.
    summary : dict
        Resumo geral da peça e resumo por patch.
    """
    if rigor <= 0:
        raise ValueError("rigor deve ser > 0.")

    results = []
    total_defect_pixels_piece = 0
    total_pixels_piece = 0

    for item in piece_frames:
        patch = item["crop"]

        if patch.ndim == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            marked = patch.copy()
        else:
            gray = patch.copy()
            marked = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

        # Guarda o patch original em BGR para permitir reconstrução sem marcações.
        original_image = marked.copy()

        gray = gray.astype(np.float32)
        h, w = gray.shape

        bg1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=9)
        bg2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
        dark_response = np.maximum(bg1 - gray, bg2 - gray)

        median = np.median(dark_response)
        mad = np.median(np.abs(dark_response - median)) + 1e-6

        threshold = max(
            15 * rigor,
            median + (3 * rigor) * 1.4826 * mad,
        )

        mask = (dark_response > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 7),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        defects = []
        defect_mask = np.zeros((h, w), dtype=np.uint8)

        min_area = h * w * min_area_frac * rigor
        min_width = w * min_width_frac * np.sqrt(rigor)

        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)

            if contour_area >= min_area and bw >= min_width:
                single_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(
                    single_mask,
                    [cnt],
                    -1,
                    255,
                    thickness=-1,
                )

                area_pixels = int(np.count_nonzero(single_mask))
                area_pct = 100.0 * area_pixels / (h * w)

                defects.append({
                    "bbox": (x, y, bw, bh),
                    "area_pixels": area_pixels,
                    "area_pct": float(area_pct),
                })

                defect_mask = cv2.bitwise_or(
                    defect_mask,
                    single_mask,
                )

                cv2.rectangle(
                    marked,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    marked,
                    f"{area_pixels}px",
                    (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        total_defect_pixels = int(np.count_nonzero(defect_mask))
        total_pixels = h * w
        defect_area_pct = 100.0 * total_defect_pixels / total_pixels

        results.append({
            "frame_id": item.get("frame_id"),
            "defect": len(defects) > 0,
            "n_defects": len(defects),
            "defects": defects,
            "total_defect_pixels": total_defect_pixels,
            "defect_area_pct": float(defect_area_pct),
            "total_pixels": total_pixels,
            "threshold": float(threshold),
            "rigor": float(rigor),
            "mask": defect_mask,
            "image": original_image,
            "marked_image": marked,
            "geometry": item.get("geometry"),
        })

        total_defect_pixels_piece += total_defect_pixels
        total_pixels_piece += total_pixels

    patch_summary = []
    for r in results:
        patch_summary.append({
            "frame_id": r["frame_id"],
            "defect": r["defect"],
            "n_defects": r["n_defects"],
            "defects": r["defects"],
            "total_defect_pixels": r["total_defect_pixels"],
            "defect_area_pct": r["defect_area_pct"],
        })

    if total_pixels_piece > 0:
        total_defect_area_pct = (
            100.0 * total_defect_pixels_piece / total_pixels_piece
        )
    else:
        total_defect_area_pct = 0.0

    summary = {
        "rigor": float(rigor),
        "frames_analyzed": len(results),
        "frames_with_defect": sum(r["defect"] for r in results),
        "total_defects": sum(r["n_defects"] for r in results),
        "total_defect_pixels": total_defect_pixels_piece,
        "defect_area_pct": float(total_defect_area_pct),
        "patches": patch_summary,
    }

    return results, summary
