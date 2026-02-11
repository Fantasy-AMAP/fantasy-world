import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np

def run():
    for i in range(100):
        if i == 0:
            spatial = []
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
        fov_x = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        fov_y = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)

        focal_x = 0.5 / np.tan(fov_x / 2)
        focal_y = 0.5 / np.tan(fov_y / 2)
        cx = cy = 0.5
        expected = np.zeros((*spatial, 3, 3))
        expected[..., 0, 0] = focal_x
        expected[..., 1, 1] = focal_y
        expected[..., 0, 2] = cx
        expected[..., 1, 2] = cy
        expected[..., 2, 2] = 1

        actual = utils3d.numpy.intrinsics_from_fov_xy(fov_x, fov_y)

        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfov_x: {fov_x}\n' + \
            f'\tfov_y: {fov_y}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
