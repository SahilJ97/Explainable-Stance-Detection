import torch
import numpy as np


def get_pad_mask(inputs):
    """Used to zero embeddings corresponding to [PAD] tokens before pooling BERT embeddings"""
    inputs = inputs.tolist()
    mask = np.ones_like(inputs)
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j] == 0:
                mask[i][j] = 0
    return torch.tensor(mask, dtype=torch.float)
