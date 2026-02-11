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
        fov_x = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        fov_y = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        near = np.random.uniform(0.1, 100, spatial)
        far = np.random.uniform(near*2, 1000, spatial)

        expected = utils3d.numpy.perspective_from_fov_xy(fov_x, fov_y, near, far)

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        fov_x = torch.tensor(fov_x, device=device)
        fov_y = torch.tensor(fov_y, device=device)
        near = torch.tensor(near, device=device)
        far = torch.tensor(far, device=device)

        actual = utils3d.torch.perspective_from_fov_xy(fov_x, fov_y, near, far).cpu().numpy()
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfov_x: {fov_x}\n' + \
            f'\tfov_y: {fov_y}\n' + \
            f'\tnear: {near}\n' + \
            f'\tfar: {far}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
