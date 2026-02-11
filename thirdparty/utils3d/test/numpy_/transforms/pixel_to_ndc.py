import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np

def run():
    for i in range(100):
        H = np.random.randint(1, 1000)
        W = np.random.randint(1, 1000)
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        pixel = np.stack([x, y], axis=-1)
        
        expected = np.stack(
            np.meshgrid(
                np.linspace(-1 + 1 / W, 1 - 1 / W, W),
                np.linspace(1 - 1 / H, -1 + 1 / H, H),
                indexing='xy'
            ),
            axis=-1
        )
  
        actual = utils3d.numpy.pixel_to_ndc(pixel, W, H)
        
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'\tH: {H}\n' + \
            f'\tW: {W}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
