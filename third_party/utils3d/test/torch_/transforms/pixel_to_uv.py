import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import torch

def run():
    for i in range(100):
        H = np.random.randint(1, 1000)
        W = np.random.randint(1, 1000)
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        pixel = np.stack([x, y], axis=-1)
  
        expected = utils3d.numpy.pixel_to_uv(pixel, W, H)

        device = [torch.device('cpu'), torch.device('cuda')][np.random.randint(2)]
        pixel = torch.tensor(pixel, device=device)
        W = torch.tensor(W, device=device)
        H = torch.tensor(H, device=device)

        actual = utils3d.torch.pixel_to_uv(pixel, W, H).cpu().numpy()
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tH: {H}\n' + \
            f'\tW: {W}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
