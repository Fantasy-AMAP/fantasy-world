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
            N = 1
        else:
            dim = np.random.randint(4)
            spatial = [np.random.randint(1, 10) for _ in range(dim)]
            N = np.random.randint(1, 10)
        fovy = np.random.uniform(5 / 180 * np.pi, 175 / 180 * np.pi, spatial)
        aspect = np.random.uniform(0.01, 100, spatial)
        focal_x = 0.5 / (np.tan(fovy / 2) * aspect)
        focal_y = 0.5 / np.tan(fovy / 2)
        near = np.random.uniform(0.1, 100, spatial)
        far = np.random.uniform(near*2, 1000, spatial)
        eye = np.random.uniform(-10, 10, [*spatial, 3])
        lookat = np.random.uniform(-10, 10, [*spatial, 3])
        up = np.random.uniform(-10, 10, [*spatial, 3])
        points = np.random.uniform(-10, 10, [*spatial, N, 3])

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        fovy = torch.tensor(fovy, device=device)
        aspect = torch.tensor(aspect, device=device)
        focal_x = torch.tensor(focal_x, device=device)
        focal_y = torch.tensor(focal_y, device=device)
        near = torch.tensor(near, device=device)
        far = torch.tensor(far, device=device)
        eye = torch.tensor(eye, device=device)
        lookat = torch.tensor(lookat, device=device)
        up = torch.tensor(up, device=device)
        points = torch.tensor(points, device=device)

        gl = utils3d.torch.project_gl(points, None,
                                    utils3d.torch.view_look_at(eye, lookat, up),
                                    utils3d.torch.perspective(fovy, aspect, near, far))
        gl_uv = gl[0][..., :2].cpu().numpy()
        gl_uv[..., 1] = 1 - gl_uv[..., 1]
        gl_depth = gl[1].cpu().numpy()
        
        cv = utils3d.torch.project_cv(points,
                                    utils3d.torch.extrinsics_look_at(eye, lookat, up),
                                    utils3d.torch.intrinsics(focal_x, focal_y, 0.5, 0.5))
        cv_uv = cv[0][..., :2].cpu().numpy()
        cv_depth = cv[1].cpu().numpy()
        
        assert np.allclose(gl_uv, cv_uv) and np.allclose(gl_depth, cv_depth), '\n' + \
            'Input:\n' + \
            f'\tfovy: {fovy}\n' + \
            f'\taspect: {aspect}\n' + \
            f'\teye: {eye}\n' + \
            f'\tlookat: {lookat}\n' + \
            f'\tup: {up}\n' + \
            f'\tpoints: {points}\n' + \
            'GL:\n' + \
            f'{gl}\n' + \
            'CV:\n' + \
            f'{cv}'
