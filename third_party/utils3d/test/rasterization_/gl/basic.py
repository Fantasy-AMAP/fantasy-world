import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import utils3d
import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

# -------------------
# CREATE CONTEXT HERE
# -------------------

import moderngl

def run():
    ctx = moderngl.create_context(
        standalone=True,
        backend='egl',
        # These are OPTIONAL if you want to load a specific version
        libgl='libGL.so.1',
        libegl='libEGL.so.1',
    )

    prog = ctx.program(vertex_shader="""
        #version 330
        uniform mat4 model;
        in vec2 in_vert;
        in vec3 in_color;
        out vec3 color;
        void main() {
            gl_Position = model * vec4(in_vert, 0.0, 1.0);
            color = in_color;
        }
        """,
        fragment_shader="""
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
    """)

    vertices = np.array([
        -0.6, -0.6,
        1.0, 0.0, 0.0,
        0.6, -0.6,
        0.0, 1.0, 0.0,
        0.0, 0.6,
        0.0, 0.0, 1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((512, 512), 4)])

    fbo.use()
    ctx.clear()
    prog['model'].write(Matrix44.from_eulers((0.0, 0.1, 0.0), dtype='f4'))
    vao.render(moderngl.TRIANGLES)

    data = fbo.read(components=3)
    image = Image.frombytes('RGB', fbo.size, data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(os.path.join(os.path.dirname(__file__), '..', '..', 'results_to_check', 'output.png'))


if __name__ == '__main__':
    run()
