import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import utils3d
import numpy as np

def run():
    image_uv, image_mesh = utils3d.numpy.utils.image_mesh(128, 128)
    image_mesh = image_mesh.reshape(-1, 4)
    depth = np.ones((128, 128), dtype=float) * 2
    depth[32:96, 32:96] = 1
    depth = depth.reshape(-1)
    intrinsics = utils3d.numpy.transforms.intrinsics_from_fov(1.0, 128, 128)
    intrinsics = utils3d.numpy.transforms.normalize_intrinsics(intrinsics, 128, 128)
    extrinsics = utils3d.numpy.transforms.extrinsics_look_at([0, 0, 1], [0, 0, 0], [0, 1, 0])
    pts = utils3d.numpy.transforms.unproject_cv(image_uv, depth, extrinsics, intrinsics)
    pts = pts.reshape(-1, 3)
    image_mesh = utils3d.numpy.mesh.triangulate(image_mesh, vertices=pts)
    utils3d.io.write_ply(os.path.join(os.path.dirname(__file__), '..', 'results_to_check', 'write_ply.ply'), pts, image_mesh)

if __name__ == '__main__':
    run()
