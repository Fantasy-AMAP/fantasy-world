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
            N = 1
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
            N = np.random.randint(1, 10)
        focal_x = np.random.uniform(0, 10, spatial)
        focal_y = np.random.uniform(0, 10, spatial)
        center_x = np.random.uniform(0, 1, spatial)
        center_y = np.random.uniform(0, 1, spatial)
        eye = np.random.uniform(-10, 10, [*spatial, 3])
        lookat = np.random.uniform(-10, 10, [*spatial, 3])
        up = np.random.uniform(-10, 10, [*spatial, 3])
        points = np.random.uniform(-10, 10, [*spatial, N, 3])

        expected = points
        
        actual = utils3d.numpy.transforms.unproject_cv(
            *utils3d.numpy.transforms.project_cv(points,
                                     utils3d.numpy.transforms.extrinsics_look_at(eye, lookat, up),
                                     utils3d.numpy.transforms.intrinsics(focal_x, focal_y, center_x, center_y)),
            utils3d.numpy.transforms.extrinsics_look_at(eye, lookat, up),
            utils3d.numpy.transforms.intrinsics(focal_x, focal_y, center_x, center_y)
        )
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfocal_x: {focal_x}\n' + \
            f'\tfocal_y: {focal_y}\n' + \
            f'\tcenter_x: {center_x}\n' + \
            f'\tcenter_y: {center_y}\n' + \
            f'\teye: {eye}\n' + \
            f'\tlookat: {lookat}\n' + \
            f'\tup: {up}\n' + \
            f'\tpoints: {points}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
