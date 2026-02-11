import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np

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

        focal = np.maximum(width, height) / (2 * np.tan(fov / 2))
        cx = width / 2 - left
        cy = height / 2 - top
        expected = np.zeros((*spatial, 3, 3))
        expected[..., 0, 0] = focal
        expected[..., 1, 1] = focal
        expected[..., 0, 2] = cx
        expected[..., 1, 2] = cy
        expected[..., 2, 2] = 1
        expected = utils3d.numpy.normalize_intrinsics(expected, crop_width, crop_height)

        actual = utils3d.numpy.crop_intrinsics(
            utils3d.numpy.normalize_intrinsics(
                utils3d.numpy.intrinsics_from_fov(fov, width, height),
                width, height
            ),
            width, height, left, top, crop_width, crop_height
        )

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
