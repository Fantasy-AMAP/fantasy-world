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
        fovy = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        aspect = np.random.uniform(0.01, 100, spatial)
        near = np.random.uniform(0.1, 100, spatial)
        far = np.random.uniform(near*2, 1000, spatial)
        
        expected = []
        for i in range(np.prod(spatial) if len(spatial) > 0 else 1):
            expected.append(glm.perspective(fovy.flat[i], aspect.flat[i], near.flat[i], far.flat[i]))
        expected = np.concatenate(expected, axis=0).reshape(*spatial, 4, 4)
        
        actual = utils3d.numpy.perspective(fovy, aspect, near, far)
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfovy: {fovy}\n' + \
            f'\taspect: {aspect}\n' + \
            f'\tnear: {near}\n' + \
            f'\tfar: {far}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
