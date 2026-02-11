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
        focal_x = np.random.uniform(1, 10000, spatial)
        focal_y = np.random.uniform(1, 10000, spatial)
        center_x = np.random.uniform(1, 10000, spatial)
        center_y = np.random.uniform(1, 10000, spatial)

        expected = np.zeros((*spatial, 3, 3))
        expected[..., 0, 0] = focal_x
        expected[..., 1, 1] = focal_y
        expected[..., 0, 2] = center_x
        expected[..., 1, 2] = center_y
        expected[..., 2, 2] = 1
  
        actual = utils3d.numpy.intrinsics(focal_x, focal_y, center_x, center_y)
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfocal_x: {focal_x}\n' + \
            f'\tfocal_y: {focal_y}\n' + \
            f'\tcenter_x: {center_x}\n' + \
            f'\tcenter_y: {center_y}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
