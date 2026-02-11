import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import numpy as np
import torch

def run():
    args = [
        {'W':2, 'H':2, 'backslash': torch.tensor([False])},
        {'W':2, 'H':2, 'backslash': torch.tensor([True])},
        {'H':2, 'W':3, 'backslash': torch.tensor([True, False])},
    ]

    expected = [
        np.array([[0, 2, 1], [1, 2, 3]]),
        np.array([[0, 2, 3], [0, 3, 1]]),
        np.array([[0, 3, 4], [0, 4, 1], [1, 4, 2], [2, 4, 5]]),
    ]

    for args, expected in zip(args, expected):
        actual = utils3d.torch.triangulate(
            utils3d.torch.image_mesh(args['H'], args['W'])[1],
            backslash=args.get('backslash', None),
        ).cpu().numpy()
                    
        assert np.allclose(expected, actual), '\n' + \
            'Input:\n' + \
            f'{args}\n' + \
            'Actual:\n' + \
            f'{actual}\n' + \
            'Expected:\n' + \
            f'{expected}'
