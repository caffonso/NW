import numpy as np


def reconstruct_board(results, use_marked=True):
    """Reconstrói a tábua concatenando os patches na ordem."""
    if not results:
        raise ValueError("results está vazio; não é possível reconstruir a peça.")

    key = "marked_image" if use_marked else "image"
    images = [r[key] for r in results]

    return np.concatenate(images, axis=1)
