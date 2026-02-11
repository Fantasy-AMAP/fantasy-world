from scipy.spatial.transform import Rotation, Slerp
import numpy as np
import torch
import utils3d


def run():
    for i in range(100):
        quat_1 = np.random.rand(4)  # [w, x, y, z]
        quat_2 = np.random.rand(4)
        t = np.array(0)
        expected = Slerp([0, 1], Rotation.from_quat([quat_1[[1, 2, 3, 0]], quat_2[[1, 2, 3, 0]]]))(t).as_matrix()
        matrix_1 = Rotation.from_quat(quat_1[[1, 2, 3, 0]]).as_matrix()
        matrix_2 = Rotation.from_quat(quat_2[[1, 2, 3, 0]]).as_matrix()
        actual = utils3d.torch.slerp(
            torch.from_numpy(matrix_1), 
            torch.from_numpy(matrix_2), 
            torch.from_numpy(t)
        ).numpy()
        assert np.allclose(actual, expected)

if __name__ == '__main__':
    run()