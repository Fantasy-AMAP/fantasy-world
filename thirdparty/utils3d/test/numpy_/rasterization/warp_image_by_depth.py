import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import imageio

def run():
    depth = np.ones((128, 128), dtype=np.float32) * 2
    depth[32:48, 32:48] = 1
    intrinsics = utils3d.numpy.transforms.intrinsics(1.0, 1.0, 0.5, 0.5).astype(np.float32)
    extrinsics_src = utils3d.numpy.transforms.extrinsics_look_at([0, 0, 1], [0, 0, 0], [0, 1, 0]).astype(np.float32)
    extrinsics_tgt = utils3d.numpy.transforms.extrinsics_look_at([1, 0, 1], [0, 0, 0], [0, 1, 0]).astype(np.float32)
    ctx = utils3d.numpy.rasterization.RastContext(
        standalone=True,
        backend='egl',
        device_index=0,
    )
    uv, _ = utils3d.numpy.rasterization.warp_image_by_depth(
        ctx,
        depth,
        extrinsics_src=extrinsics_src,
        extrinsics_tgt=extrinsics_tgt,
        intrinsics_src=intrinsics
    )
    uv = (np.concatenate([uv, np.zeros((128, 128, 1), dtype=np.float32)], axis=-1) * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(os.path.dirname(__file__), '..', '..', 'results_to_check', 'warp_image_uv.png'), uv)

if __name__ == '__main__':
    run()
