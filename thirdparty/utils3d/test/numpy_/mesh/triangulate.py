import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np

def run():
    for i in range(100):
        if i == 0:
            spatial = []
            L = 1
            N = 5
            faces = np.array([[0, 1, 2, 3, 4]])
            expected = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
            L = np.random.randint(1, 1000)
            N = np.random.randint(3, 10)
            faces = np.random.randint(0, 10000, size=(*spatial, L, N))

            loop_indices = [[0, i, i + 1] for i in range(1, N - 1)]
            expected = faces[..., loop_indices].reshape((*spatial, L * (N - 2), 3))

        actual = utils3d.numpy.triangulate(faces)
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'{faces}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
