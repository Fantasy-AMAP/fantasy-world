import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import torch

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

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        near = torch.tensor(near, device=device)
        far = torch.tensor(far, device=device)
        depth = torch.tensor(depth, device=device)
        
        actual = utils3d.torch.depth_buffer_to_linear(
            utils3d.torch.project_depth(depth, near, far),
            near, far
        ).cpu().numpy()
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tdepth: {depth}\n' + \
            f'\tnear: {near}\n' + \
            f'\tfar: {far}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
