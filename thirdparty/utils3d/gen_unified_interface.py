import inspect
import textwrap
import re
import itertools
import numbers
import importlib
import sys
import functools
from pathlib import Path

from utils3d._helpers import suppress_traceback


def _contains_tensor(obj):
    if isinstance(obj, (list, tuple)):
        return any(_contains_tensor(item) for item in obj)
    elif isinstance(obj, dict):
        return any(_contains_tensor(value) for value in obj.values())
    else:
        import torch
        return isinstance(obj, torch.Tensor)

@suppress_traceback
def _call_based_on_args(fname, args, kwargs):
    if 'torch' in sys.modules:
        if any(_contains_tensor(arg) for arg in args) or any(_contains_tensor(v) for v in kwargs.values()):
            fn = getattr(utils3d.torch, fname, None)
            if fn is None:
                raise NotImplementedError(f"Function {fname} has no torch implementation.")
            return fn(*args, **kwargs)
    fn = getattr(utils3d.numpy, fname, None)
    if fn is None:
        raise NotImplementedError(f"Function {fname} has no numpy implementation.") 
    return fn(*args, **kwargs)
    

def extract_signature(fn):
    signature = inspect.signature(fn)
    
    signature_str = str(signature)

    signature_str = re.sub(r"<class '.*'>", lambda m: m.group(0).split('\'')[1], signature_str)
    signature_str = re.sub(r"(?<!\.)numpy\.", "numpy_.", signature_str)
    signature_str = re.sub(r"(?<!\.)torch\.", "torch_.", signature_str)
    
    return signature_str



if __name__ == "__main__":
    import utils3d.numpy, utils3d.torch
    numpy_impl = utils3d.numpy
    torch_impl = utils3d.torch
    numpy_funcs = {name: getattr(numpy_impl, name) for name in numpy_impl.__all__}
    torch_funcs = {name: getattr(torch_impl, name) for name in torch_impl.__all__}

    all = {**numpy_funcs, **torch_funcs}

    Path("utils3d/_unified").mkdir(exist_ok=True)

    with open("utils3d/_unified/__init__.pyi", "w", encoding="utf-8") as f:
        f.write(inspect.cleandoc(
            f"""
            # Auto-generated interface file
            from typing import List, Tuple, Dict, Union, Optional, Any, overload, Literal, Callable
            import numpy as numpy_
            import torch as torch_
            import nvdiffrast.torch
            import numbers
            from . import numpy, torch
            import utils3d.numpy, utils3d.torch
            """
        ))
        f.write("\n\n")
        f.write(f"__all__ = [{', \n'.join('\"' + s + '\"' for s in all.keys())}]\n\n")

        for fname, fn in itertools.chain(numpy_funcs.items(), torch_funcs.items()):
            sig, doc = extract_signature(fn), inspect.getdoc(fn)

            f.write(f"@overload\n")
            f.write(f"def {fname}{sig}:\n")
            f.write(f"    \"\"\"{doc}\"\"\"\n" if doc else "")
            f.write(f"    {fn.__module__}.{fn.__qualname__}\n\n")

    with open("utils3d/_unified/__init__.py", "w", encoding="utf-8") as f:
        f.write(inspect.cleandoc(
            f"""
            # Auto-generated implementation redirecting to numpy/torch implementations
            import sys
            from typing import TYPE_CHECKING
            import utils3d
            from .._helpers import suppress_traceback
            """
        ))
        f.write("\n\n")
        f.write(f"__all__ = [{', \n'.join('\"' + s + '\"' for s in all.keys())}]\n\n")
        f.write(inspect.getsource(_contains_tensor) + "\n\n")
        f.write(inspect.getsource(_call_based_on_args) + "\n\n")

        for fname in {**numpy_funcs, **torch_funcs}:
            f.write(f'@suppress_traceback\n')
            f.write(f"def {fname}(*args, **kwargs):\n")
            f.write(f"    if TYPE_CHECKING:  # redirected to:\n")
            f.write(f"        {'utils3d.numpy.' + fname if fname in numpy_funcs else 'None'}, {'utils3d.torch.'+ fname if fname in torch_funcs else 'None'}\n")
            f.write(f"    return _call_based_on_args('{fname}', args, kwargs)\n\n")
            