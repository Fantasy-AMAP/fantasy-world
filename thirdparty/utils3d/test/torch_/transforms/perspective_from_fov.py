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
        fov = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        width = np.random.uniform(1, 10000, spatial)
        height = np.random.uniform(1, 10000, spatial)
        near = np.random.uniform(0.1, 100, spatial)
        far = np.random.uniform(near*2, 1000, spatial)

        expected = utils3d.numpy.perspective_from_fov(fov, width, height, near, far)

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        fov = torch.tensor(fov, device=device)
        width = torch.tensor(width, device=device)
        height = torch.tensor(height, device=device)
        near = torch.tensor(near, device=device)
        far = torch.tensor(far, device=device)

        actual = utils3d.torch.perspective_from_fov(fov, width, height, near, far).cpu().numpy()
        
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
        