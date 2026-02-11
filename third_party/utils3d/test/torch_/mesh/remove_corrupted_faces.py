import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import torch

def run():
    for i in range(100):
        if i == 0:
            faces = np.array([[0, 1, 2], [0, 2, 2], [0, 2, 3]])
            expected = np.array([[0, 1, 2], [0, 2, 3]])
        else:
            L = np.random.randint(1, 1000)
            N = np.random.randint(100, 1000)
            faces = np.random.randint(0, N, size=(L, 3))
            faces[..., 1] = (faces[..., 0] + 1) % N
            faces[..., 2] = (faces[..., 0] + 2) % N

        expected = utils3d.numpy.remove_corrupted_faces(faces)

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        faces = torch.tensor(faces, device=device)

        actual = utils3d.torch.remove_corrupted_faces(faces).cpu().numpy()
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'{faces}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
