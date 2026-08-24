"""Differentiable path tracer as a Tesseract.

Endpoint names come from the Tesseract Core endpoint reference:
  apply                      (required)
  vector_jacobian_product    (optional)
  jacobian_vector_product    (optional)
  jacobian                   (optional)
  abstract_eval              (optional)

They must be spelled exactly like this or the runtime will not register them.
`vjp` / `jvp` are NOT valid names.
"""

from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32

# --------------------------------------------------------------------------
# Mitsuba setup.  llvm_ad_rgb = CPU + autodiff.  Set before any mi.* call.
# --------------------------------------------------------------------------
mi.set_variant("llvm_ad_rgb")

_HERE = Path(__file__).parent
_SCENE = None
_PARAMS = None

# Verify these against `python pipeline/dump_params.py` output. If the BSDF is
# wrapped in <bsdf type="twosided"> each key gains a `brdf_0.` segment.
KEY_BASECOLOR = "object.bsdf.base_color.data"
KEY_ROUGHNESS = "object.bsdf.roughness.data"
KEY_METALLIC = "object.bsdf.metallic.data"


def _scene():
    """Load the scene once per container, not once per call."""
    global _SCENE, _PARAMS
    if _SCENE is None:
        _SCENE = mi.load_file(str(_HERE / "scene.xml"))
        _PARAMS = mi.traverse(_SCENE)
    return _SCENE, _PARAMS


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class InputSchema(BaseModel):
    basecolor: Differentiable[Array[(None, None, 3), Float32]]
    roughness: Differentiable[Array[(None, None, 1), Float32]]
    metallic: Differentiable[Array[(None, None, 1), Float32]]
    camera_index: int = Field(0, ge=0, le=5)
    spp: int = Field(16, ge=1)
    seed: int = Field(0)


class OutputSchema(BaseModel):
    image: Differentiable[Array[(None, None, 3), Float32]]


# --------------------------------------------------------------------------
# Shared forward path
# --------------------------------------------------------------------------
def _set_textures(params, basecolor, roughness, metallic):
    """Write incoming maps into the scene's bitmap texture buffers.

    We pass (H, W, C) arrays directly; Mitsuba's TensorXf handles conversion.
    Scene.xml loads placeholder PNGs to establish buffers. Resolution MUST
    match placeholders or update() will error.
    TODO: Verify actual shape with dump_params.py before first run.
    """
    params[KEY_BASECOLOR] = mi.TensorXf(np.ascontiguousarray(basecolor, dtype=np.float32))
    params[KEY_ROUGHNESS] = mi.TensorXf(np.ascontiguousarray(roughness, dtype=np.float32))
    params[KEY_METALLIC] = mi.TensorXf(np.ascontiguousarray(metallic, dtype=np.float32))
    params.update()


def _render(params, camera_index, spp, seed):
    scene, _ = _scene()
    sensor = scene.sensors()[camera_index]
    return mi.render(scene, params, sensor=sensor, spp=spp, seed=seed, seed_grad=seed + 1)


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------
def apply(inputs: InputSchema) -> OutputSchema:
    _, params = _scene()
    _set_textures(params, inputs.basecolor, inputs.roughness, inputs.metallic)
    img = _render(params, inputs.camera_index, inputs.spp, inputs.seed)
    return OutputSchema(image=np.array(img, dtype=np.float32))


# --------------------------------------------------------------------------
# vector_jacobian_product
# --------------------------------------------------------------------------
def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set,
    vjp_outputs: set,
    cotangent_vector: dict,
):
    """Backward pass.

    Contract (from the endpoint reference):
      cotangent_vector : {vjp_outputs: array}, shaped like the outputs
      returns          : {vjp_inputs:  array}, shaped like the inputs

    The three PARAMS.update() calls below are the classic failure point. Setting
    a gradient without update() gives zeros with no error message.
    """
    scene, params = _scene()

    key_map = {
        "basecolor": (KEY_BASECOLOR, inputs.basecolor),
        "roughness": (KEY_ROUGHNESS, inputs.roughness),
        "metallic": (KEY_METALLIC, inputs.metallic),
    }

    # 1. Load the primal point, then mark requested inputs differentiable.
    _set_textures(params, inputs.basecolor, inputs.roughness, inputs.metallic)
    for name in vjp_inputs:
        if name in key_map:
            dr.enable_grad(params[key_map[name][0]])
    params.update()

    # 2. Re-render WITH grad tracking enabled. Same seed as the forward pass so
    #    PRB replays identical paths.
    img = _render(params, inputs.camera_index, inputs.spp, inputs.seed)

    # 3. Seed the output with the incoming cotangent and propagate backward.
    ct = np.ascontiguousarray(cotangent_vector["image"], dtype=np.float32)
    dr.set_grad(img, mi.TensorXf(ct))
    dr.backward_from(img)

    # 4. Read gradients off the scene parameters.
    out = {}
    for name in vjp_inputs:
        if name not in key_map:
            continue
        key, ref = key_map[name]
        g = dr.grad(params[key])
        g = np.array(g, dtype=np.float32).reshape(np.asarray(ref).shape)
        out[name] = g
    return out


# --------------------------------------------------------------------------
# abstract_eval
# --------------------------------------------------------------------------
def abstract_eval(abstract_inputs):
    """Shape inference without running the renderer."""
    h, w, _ = abstract_inputs.basecolor.shape
    return {"image": {"shape": (h, w, 3), "dtype": "float32"}}
