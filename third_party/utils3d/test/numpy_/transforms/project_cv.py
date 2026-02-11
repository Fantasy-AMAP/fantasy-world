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

        pts = points - eye[..., None, :]
        z_axis = lookat - eye
        x_axis = np.cross(-up, z_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
        y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
        z_axis = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)
        z = (pts * z_axis[..., None, :]).sum(axis=-1)
        x = (pts * x_axis[..., None, :]).sum(axis=-1)
        y = (pts * y_axis[..., None, :]).sum(axis=-1)
        x = (x / z * focal_x[..., None] + center_x[..., None])
        y = (y / z * focal_y[..., None] + center_y[..., None])
        expected = np.stack([x, y], axis=-1)
        
        actual, _ = utils3d.numpy.transforms.project_cv(points,
                                            utils3d.numpy.extrinsics_look_at(eye, lookat, up),
                                            utils3d.numpy.intrinsics(focal_x, focal_y, center_x, center_y))
        
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
