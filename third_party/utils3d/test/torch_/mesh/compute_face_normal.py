import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import torch

def run():
    for i in range(100):
        if i == 0:
            spatial = []
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            faces = np.array([[0, 1, 2]])
            expected = np.array([[0, 0, 1]])
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
            N = np.random.randint(100, 1000)
            vertices = np.random.rand(*spatial, N, 3)
            L = np.random.randint(1, 1000)
            faces = np.random.randint(0, N, size=(*spatial, L, 3))
            faces[..., 1] = (faces[..., 0] + 1) % N
            faces[..., 2] = (faces[..., 0] + 2) % N

        expected = utils3d.numpy.compute_face_normal(vertices, faces)

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        vertices = torch.tensor(vertices, device=device)
        faces = torch.tensor(faces, device=device)

        actual = utils3d.torch.compute_face_normal(vertices, faces).cpu().numpy()
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'{faces}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
