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
        eye = np.random.uniform(-10, 10, [*spatial, 3]).astype(np.float32)
        lookat = np.random.uniform(-10, 10, [*spatial, 3]).astype(np.float32)
        up = np.random.uniform(-10, 10, [*spatial, 3]).astype(np.float32)

        expected = utils3d.numpy.extrinsics_look_at(eye, lookat, up)

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        eye = torch.tensor(eye, device=device)
        lookat = torch.tensor(lookat, device=device)
        up = torch.tensor(up, device=device)

        actual = utils3d.torch.extrinsics_look_at(eye, lookat, up).cpu().numpy()
        
        assert np.allclose(expected, actual, 1e-5, 1e-5), '\n' + \
            'Input:\n' + \
            f'eye: {eye}\n' + \
            f'lookat: {lookat}\n' + \
            f'up: {up}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
        