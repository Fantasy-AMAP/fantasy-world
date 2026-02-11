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
        left = np.random.uniform(0, width, spatial)
        top = np.random.uniform(0, height, spatial)
        crop_width = np.random.uniform(0, width - left, spatial)
        crop_height = np.random.uniform(0, height - top, spatial)

        expected = utils3d.numpy.crop_intrinsics(
            utils3d.numpy.normalize_intrinsics(
                utils3d.numpy.intrinsics_from_fov(fov, width, height),
                width, height
            ),
            width, height, left, top, crop_width, crop_height
        )

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        fov = torch.tensor(fov, device=device)
        width = torch.tensor(width, device=device)
        height = torch.tensor(height, device=device)
        left = torch.tensor(left, device=device)
        top = torch.tensor(top, device=device)
        crop_width = torch.tensor(crop_width, device=device)
        crop_height = torch.tensor(crop_height, device=device)

        actual = utils3d.torch.crop_intrinsics(
            utils3d.torch.normalize_intrinsics(
                utils3d.torch.intrinsics_from_fov(fov, width, height),
                width, height
            ),
            width, height, left, top, crop_width, crop_height
        ).cpu().numpy()

        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfov: {fov}\n' + \
            f'\twidth: {width}\n' + \
            f'\theight: {height}\n' + \
            f'\tleft: {left}\n' + \
            f'\ttop: {top}\n' + \
            f'\tcrop_width: {crop_width}\n' + \
            f'\tcrop_height: {crop_height}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
