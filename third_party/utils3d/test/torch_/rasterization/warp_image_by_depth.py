import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import torch
import imageio

def run():
    depth = torch.ones((1, 128, 128), dtype=torch.float32, device='cuda') * 2
    depth[:, 32:48, 32:48] = 1
    intrinsics = utils3d.torch.transforms.intrinsics(1.0, 1.0, 0.5, 0.5).to(depth)
    extrinsics_src = utils3d.torch.transforms.extrinsics_look_at([0., 0., 1.], [0., 0., 0.], [0., 1., 0.]).to(depth)
    extrinsics_tgt = utils3d.torch.transforms.extrinsics_look_at([1., 0., 1.], [0., 0., 0.], [0., 1., 0.]).to(depth)
    ctx = utils3d.torch.rasterization.RastContext(backend='gl', device='cuda')
    uv, _ = utils3d.torch.rasterization.warp_image_by_depth(
        ctx,
        depth,
        extrinsics_src=extrinsics_src,
        extrinsics_tgt=extrinsics_tgt,
        intrinsics_src=intrinsics,
        antialiasing=False,
    )
    uv = torch.cat([uv, torch.zeros((1, 1, 128, 128)).to(uv)], dim=1) * 255
    uv = uv.permute(0, 2, 3, 1).squeeze().cpu().numpy().astype(np.uint8)

    imageio.imwrite(os.path.join(os.path.dirname(__file__), '..', '..', 'results_to_check', 'torch_warp_image_uv.png'), uv)

if __name__ == '__main__':
    run()
