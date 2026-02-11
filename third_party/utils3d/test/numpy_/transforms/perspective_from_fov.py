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
        fov = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        width = np.random.uniform(1, 10000, spatial)
        height = np.random.uniform(1, 10000, spatial)
        near = np.random.uniform(0.1, 100, spatial)
        far = np.random.uniform(near*2, 1000, spatial)

        fov_y = 2 * np.arctan(np.tan(fov / 2) * height / np.maximum(width, height))
        expected = []
        for i in range(np.prod(spatial) if len(spatial) > 0 else 1):
            expected.append(glm.perspective(fov_y.flat[i], width.flat[i] / height.flat[i], near.flat[i], far.flat[i]))
        expected = np.concatenate(expected, axis=0).reshape(*spatial, 4, 4)

        actual = utils3d.numpy.perspective_from_fov(fov, width, height, near, far)
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfov: {fov}\n' + \
            f'\twidth: {width}\n' + \
            f'\theight: {height}\n' + \
            f'\tnear: {near}\n' + \
            f'\tfar: {far}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
        