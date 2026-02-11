import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import utils3d


def run():
    for i in range(10):
        if i == 0:
            spatial = []
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
        angle = np.random.uniform(-np.pi, np.pi, spatial)
        axis = np.random.uniform(-1, 1, spatial + [3])
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        axis_angle = angle[..., None] * axis
        matrix = R.from_rotvec(axis_angle.reshape((-1, 3))).as_matrix().reshape(spatial + [3, 3])
        # matrix = np.array([
        #     [1, 0, 0],
        #     [0, 0, -1],
        #     [0, 1, 0]
        # ]).astype(np.float32)
        # dim = 0
        # spatial = []
        expected = R.from_matrix(matrix.reshape(-1, 3, 3)).as_quat().reshape(spatial + [4])[..., [3, 0, 1, 2]]
        actual = utils3d.torch.matrix_to_quaternion(
            torch.from_numpy(matrix)
        ).cpu().numpy()
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tangle: {angle}\n' + \
            f'\taxis: {axis}\n' + \
            f'\tmatrix: {matrix}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'

if __name__ == '__main__':
    run()