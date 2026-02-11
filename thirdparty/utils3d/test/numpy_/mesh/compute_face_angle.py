import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np

def run():
    for i in range(100):
        if i == 0:
            spatial = []
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            faces = np.array([[0, 1, 2]])
            expected = np.array([[np.pi/2, np.pi/4, np.pi/4]])
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
            N = np.random.randint(100, 1000)
            vertices = np.random.rand(*spatial, N, 3)
            L = np.random.randint(1, 1000)
            faces = np.random.randint(0, N, size=(*spatial, L, 3))
            faces[..., 1] = (faces[..., 0] + 1) % N
            faces[..., 2] = (faces[..., 0] + 2) % N

            faces_ = faces.reshape(-1, L, 3)
            vertices_ = vertices.reshape(-1, N, 3)
            N = vertices_.shape[0]
            expected = np.zeros((N, L, 3), dtype=float)
            for i in range(3):
                edge0 = vertices_[np.arange(N)[:, None], faces_[..., (i+1)%3]] - vertices_[np.arange(N)[:, None], faces_[..., i]]
                edge1 = vertices_[np.arange(N)[:, None], faces_[..., (i+2)%3]] - vertices_[np.arange(N)[:, None], faces_[..., i]]
                expected[..., i] = np.arccos(np.sum(
                    edge0 / np.linalg.norm(edge0, axis=-1, keepdims=True) * \
                    edge1 / np.linalg.norm(edge1, axis=-1, keepdims=True),
                    axis=-1
                ))
            expected = expected.reshape(*spatial, L, 3)

        actual = utils3d.numpy.compute_face_angle(vertices, faces)
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'{faces}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
