import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
from trimesh import geometry

def run():
    for i in range(100):
        if i == 0:
            spatial = []
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            faces = np.array([[0, 1, 2]])
            expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
            N = np.random.randint(100, 1000)
            vertices = np.random.rand(*spatial, N, 3)
            L = np.random.randint(1, 1000)
            faces = np.random.randint(0, N, size=(*spatial, L, 3))
            faces[..., 1] = (faces[..., 0] + 1) % N
            faces[..., 2] = (faces[..., 0] + 2) % N
            face_normals = utils3d.numpy.compute_face_normal(vertices, faces)

            faces_ = faces.reshape(-1, L, 3)
            face_normals = face_normals.reshape(-1, L, 3)
            vertices_normals = []
            for face, face_normal in zip(faces_, face_normals):
                vertices_normals.append(
                    geometry.mean_vertex_normals(N, face, face_normal)
                )
            expected = np.array(vertices_normals).reshape(*spatial, N, 3)

        actual = utils3d.numpy.compute_vertex_normal(vertices, faces)
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'{faces}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
