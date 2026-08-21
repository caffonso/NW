import numpy as np


def reconstruct_board(results, use_marked=True):
    """
    Reconstrói a tábua concatenando os patches na ordem.

    Parameters
    ----------
    results : list
        Saída de detect_texture_defects().
    use_marked : bool
        True -> usa imagens com defeitos marcados.
        False -> usa imagens originais.
    """
    if not results:
        raise ValueError("results está vazio; não é possível reconstruir a peça.")

    key = "marked_image" if use_marked else "image"
    images = [r[key] for r in results]

    return np.concatenate(images, axis=1)
