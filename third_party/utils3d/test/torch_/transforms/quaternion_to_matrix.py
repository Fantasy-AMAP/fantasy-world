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
        angle = np.random.uniform(-np.pi, np.pi, spatial).astype(np.float32)
        axis = np.random.uniform(-1, 1, spatial + [3]).astype(np.float32)
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        axis_angle = angle[..., None] * axis
        quat = R.from_rotvec(axis_angle.reshape((-1, 3))).as_quat().reshape(spatial + [4])
        expected = R.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(spatial + [3, 3])
        actual = utils3d.torch.quaternion_to_matrix(
            torch.from_numpy(quat[..., [3, 0, 1, 2]])
        ).cpu().numpy()
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tangle: {angle}\n' + \
            f'\taxis: {axis}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'

if __name__ == '__main__':
    run()