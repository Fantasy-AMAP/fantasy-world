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
        fovy = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        aspect = np.random.uniform(0.01, 100, spatial)
        near = np.random.uniform(0.1, 100, spatial)
        far = np.random.uniform(near*2, 1000, spatial)
        eye = np.random.uniform(-10, 10, [*spatial, 3])
        lookat = np.random.uniform(-10, 10, [*spatial, 3])
        up = np.random.uniform(-10, 10, [*spatial, 3])
        points = np.random.uniform(-10, 10, [*spatial, N, 3])

        pts = points - eye[..., None, :]
        z_axis = (eye - lookat)
        x_axis = np.cross(up, z_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
        y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
        z_axis = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)
        z = (pts * z_axis[..., None, :]).sum(axis=-1)
        x = (pts * x_axis[..., None, :]).sum(axis=-1)
        y = (pts * y_axis[..., None, :]).sum(axis=-1)
        x = (x / -z / np.tan(fovy[..., None] / 2) / aspect[..., None]) * 0.5 + 0.5
        y = (y / -z / np.tan(fovy[..., None] / 2)) * 0.5 + 0.5
        z = utils3d.numpy.project_depth(-z, near[..., None], far[..., None])
        expected = np.stack([x, y, z], axis=-1)
        
        actual, _ = utils3d.numpy.transforms.project_gl(points, None,
                                          utils3d.numpy.view_look_at(eye, lookat, up),
                                          utils3d.numpy.perspective(fovy, aspect, near, far))
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tfovy: {fovy}\n' + \
            f'\taspect: {aspect}\n' + \
            f'\tnear: {near}\n' + \
            f'\tfar: {far}\n' + \
            f'\teye: {eye}\n' + \
            f'\tlookat: {lookat}\n' + \
            f'\tup: {up}\n' + \
            f'\tpoints: {points}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
