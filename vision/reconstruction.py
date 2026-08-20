import numpy as np

def reconstruct_board(piece_frames, overlap=0.25):
    if not piece_frames:
        raise ValueError("piece_frames is empty")
    images = [pf.image for pf in piece_frames]
    width = images[0].shape[1]
    step = max(1, int(width*(1-overlap)))
    total_width = step*(len(images)-1) + width
    board = np.zeros((images[0].shape[0], total_width), dtype=images[0].dtype)
    for i, image in enumerate(images):
        x = i*step
        board[:, x:x+width] = image
    return board
