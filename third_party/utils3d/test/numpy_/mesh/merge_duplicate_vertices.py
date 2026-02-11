import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np

def run():
    for i in range(100):
        if i == 0:
            spatial = []
            vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
            faces = np.array([[0, 1, 2]])
            expected_vertices = np.array([[0, 0, 0], [1, 0, 0]])
            expected_faces = np.array([[0, 1, 1]])
            expected = expected_vertices[expected_faces]
        else:
            N = np.random.randint(100, 1000)
            vertices = np.random.rand(N, 3)
            L = np.random.randint(1, 1000)
            faces = np.random.randint(0, N, size=(L, 3))
            faces[..., 1] = (faces[..., 0] + 1) % N
            faces[..., 2] = (faces[..., 0] + 2) % N
            vertices[-(N//2):] = vertices[:N//2]
            
            expected_vertices = vertices[:-(N//2)].copy()
            expected_faces = faces.copy()
            expected_faces[expected_faces >= N - N//2] -= N - N//2
            expected = expected_vertices[expected_faces]

        actual_vertices, actual_faces = utils3d.numpy.merge_duplicate_vertices(vertices, faces)
        actual = actual_vertices[actual_faces]
                    
        assert expected_vertices.shape == actual_vertices.shape and np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'{faces}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
