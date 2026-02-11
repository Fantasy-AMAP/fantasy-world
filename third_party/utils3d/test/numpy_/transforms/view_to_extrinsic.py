import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import glm

def run():
    for i in range(100):
        if i == 0:
            spatial = []
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
        eye = np.random.uniform(-10, 10, [*spatial, 3]).astype(np.float32)
        lookat = np.random.uniform(-10, 10, [*spatial, 3]).astype(np.float32)
        up = np.random.uniform(-10, 10, [*spatial, 3]).astype(np.float32)

        expected = utils3d.numpy.extrinsics_look_at(eye, lookat, up)

        actual = utils3d.numpy.view_to_extrinsics(utils3d.numpy.view_look_at(eye, lookat, up))
        
        assert np.allclose(expected, actual, 1e-5, 1e-5), '\n' + \
            'Input:\n' + \
            f'eye: {eye}\n' + \
            f'lookat: {lookat}\n' + \
            f'up: {up}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
        