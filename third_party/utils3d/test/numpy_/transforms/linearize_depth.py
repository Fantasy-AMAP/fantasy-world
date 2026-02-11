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
        near = np.random.uniform(0.1, 100, spatial)
        far = np.random.uniform(near*2, 1000, spatial)
        depth = np.random.uniform(near, far, spatial)
        
        expected = depth
        
        actual = utils3d.numpy.depth_buffer_to_linear(
            utils3d.numpy.project_depth(depth, near, far),
            near, far
        )
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tdepth: {depth}\n' + \
            f'\tnear: {near}\n' + \
            f'\tfar: {far}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
