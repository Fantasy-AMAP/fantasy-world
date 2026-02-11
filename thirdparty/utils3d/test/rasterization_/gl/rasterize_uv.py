import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import imageio

def run():
    image_uv, image_mesh = utils3d.numpy.utils.image_mesh(128, 128)
    image_mesh = image_mesh.reshape(-1, 4)
    depth = np.ones((128, 128), dtype=np.float32) * 2
    depth[32:96, 32:96] = 1
    depth = depth.reshape(-1)
    intrinsics = utils3d.numpy.transforms.intrinsics_from_fov(1.0, 128, 128).astype(np.float32)
    intrinsics = utils3d.numpy.transforms.normalize_intrinsics(intrinsics, 128, 128)
    extrinsics = utils3d.numpy.transforms.extrinsics_look_at([0, 0, 1], [0, 0, 0], [0, 1, 0]).astype(np.float32)
    pts = utils3d.numpy.transforms.unproject_cv(image_uv, depth, extrinsics, intrinsics)
    pts = pts.reshape(-1, 3)
    image_mesh = utils3d.numpy.mesh.triangulate(image_mesh, vertices=pts)
    
    perspective = utils3d.numpy.transforms.perspective(1.0, 1.0, 0.1, 10)
    view = utils3d.numpy.transforms.view_look_at([1, 0, 1], [0, 0, 0], [0, 1, 0])
    mvp = np.matmul(perspective, view)
    ctx = utils3d.numpy.rasterization.RastContext(
        standalone=True,
        backend='egl',
        device_index=0,
    )
    uv = utils3d.numpy.rasterization.rasterize_triangle_faces(
        ctx,
        pts,
        image_mesh,
        image_uv,
        width=128,
        height=128,
        mvp=mvp,
    )[0]
    uv = (np.concatenate([uv, np.zeros((128, 128, 1), dtype=np.float32)], axis=-1) * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(os.path.dirname(__file__), '..', '..', 'results_to_check', 'rasterize_uv.png'), uv)

if __name__ == '__main__':
    run()
