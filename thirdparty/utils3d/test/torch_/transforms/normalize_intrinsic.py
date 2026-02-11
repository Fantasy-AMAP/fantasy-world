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
        fov_x = np.where(width >= height, fov, 2 * np.arctan(np.tan(fov / 2) * width / height))
        fov_y = np.where(width >= height, 2 * np.arctan(np.tan(fov / 2) * height / width), fov)

        expected = utils3d.numpy.normalize_intrinsics(utils3d.numpy.intrinsics_from_fov(fov, width, height), width, height)

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        fov = torch.tensor(fov, device=device)
        width = torch.tensor(width, device=device)
        height = torch.tensor(height, device=device)

        actual = utils3d.torch.normalize_intrinsics(utils3d.torch.intrinsics_from_fov(fov, width, height), width, height).cpu().numpy()

        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfov: {fov}\n' + \
            f'\twidth: {width}\n' + \
            f'\theight: {height}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
