import matplotlib.pyplot as plt
import numpy as np


def show(ax, img, title):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img[..., ::-1])
    ax.set_title(title)
    ax.axis("off")


def show_pipeline(
    first,
    piece_frames,
    board,
    detection_results,
    summary,
    result=None,
):
    """Visualiza as principais etapas do pipeline atualizado."""
    fig, ax = plt.subplots(figsize=(10, 6))
    show(ax, first, "1. Frame inicial")

    fig2, axs = plt.subplots(2, 3, figsize=(15, 8))
    ids = np.linspace(
        0,
        len(piece_frames) - 1,
        min(6, len(piece_frames)),
        dtype=int,
    ) if piece_frames else []

    for a in axs.ravel():
        a.axis("off")

    for a, i in zip(axs.ravel(), ids):
        show(a, piece_frames[i]["crop"], f"Patch {i}")

    fig2.suptitle(f"2. Patches capturados ({len(piece_frames)})")

    fig3, ax3 = plt.subplots(figsize=(18, 5))
    show(ax3, board, "3. Peça reconstruída com defeitos")

    fig4, axs4 = plt.subplots(2, 3, figsize=(15, 8))
    for a in axs4.ravel():
        a.axis("off")

    defect_ids = [
        i for i, r in enumerate(detection_results)
        if r["n_defects"] > 0
    ]
    ids4 = defect_ids[:6] or list(range(min(6, len(detection_results))))

    for a, i in zip(axs4.ravel(), ids4):
        r = detection_results[i]
        show(
            a,
            r["marked_image"],
            f"Patch {r['frame_id']} - defeitos: {r['n_defects']}",
        )

    classification = (
        getattr(result, "classification", None)
        if result is not None
        else ("REJECT" if summary["total_defects"] else "ACCEPT")
    )

    fig4.suptitle(
        "4. Defeitos marcados | "
        f"{classification} | total={summary['total_defects']}"
    )

    plt.show()
