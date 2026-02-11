# utils3d
Easy 3D python utilities for computer vision and graphics researchers.

Supports:
* Transformation between OpenCV and OpenGL coordinate systems, **no more confusion**
* Easy rasterization, **no worries about OpenGL objects and buffers**
* Some mesh processing utilities, **all vectorized for effciency; some differentiable**
* Projection, unprojection, depth-based image warping, flow-based image warping...
* Easy Reading and writing .obj, .ply files
* Reading and writing Colmap format camera parameters
* NeRF/MipNeRF utilities

For most functions, there are both numpy (indifferentiable) and pytorch implementations (differentiable).

Pytorch is not required for using this package, but if you want to use the differentiable functions, you will need to install pytorch (and nvdiffrast if you want to use the pytorch rasterization functions).

## Install

Install by git

```bash
pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d
```

or clone the repo and install with `-e` option for convenient updating and modifying.

```bash
git clone https://github.com/EasternJournalist/utils3d.git
pip install -e ./utils3d
```

## Topics (TODO)

### Camera

### Rotations

### Mesh

### Rendering

### Projection

### Image warping

### NeRF