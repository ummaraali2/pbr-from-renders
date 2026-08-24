"""PBRNet as a Tesseract.

The hackathon requires two or more composed Tesseracts, so the network gets its
own container. The boundary this creates is real in the sense criterion 01
asks for: this container differentiates with torch autograd, the renderer
container differentiates with Dr.Jit. Neither AD system can express the other.

Design choice: the weights are a DIFFERENTIABLE INPUT (a flat float32 vector),
not container-internal state. Tesseract's endpoints are stateless -- there is no
"update my weights" endpoint -- so the optimizer must live outside and pass
weights in. The VJP returns the gradient with respect to that vector, which the
outer torch optimizer then steps.

Cost: the weight vector crosses HTTP twice per iteration. At base_ch=16 that is
~480k floats (~1.9 MB) each way. Measure it -- that number is the substance of
a Best Engineering writeup, not a footnote.
"""

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from tesseract_core.runtime import Array, Differentiable, Float32

from model import PBRNet

BASE_CH = 16  # smaller than the in-process default: this vector is serialized

_NET = None


def _net():
    global _NET
    if _NET is None:
        _NET = PBRNet(base_ch=BASE_CH).float()
        _NET.eval()  # BatchNorm in train mode with batch size 1 is unstable
    return _NET


def n_params():
    return sum(p.numel() for p in _net().parameters())


def _load_weights(net, weights):
    vector_to_parameters(
        torch.as_tensor(np.asarray(weights, dtype=np.float32)).clone(),
        net.parameters(),
    )


class InputSchema(BaseModel):
    views: Differentiable[Array[(None, None, None, 3), Float32]]  # [6,H,W,3]
    weights: Differentiable[Array[(None,), Float32]]
    base_ch: int = Field(BASE_CH)


class OutputSchema(BaseModel):
    basecolor: Differentiable[Array[(None, None, 3), Float32]]
    roughness: Differentiable[Array[(None, None, 1), Float32]]
    metallic: Differentiable[Array[(None, None, 1), Float32]]


def _forward(views, net):
    v = torch.as_tensor(np.asarray(views, dtype=np.float32)).unsqueeze(0)
    out = net(v)
    return {
        "basecolor": out["basecolor"][0].permute(1, 2, 0),
        "roughness": out["roughness"][0].permute(1, 2, 0),
        "metallic": out["metallic"][0].permute(1, 2, 0),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    net = _net()
    _load_weights(net, inputs.weights)
    with torch.no_grad():
        out = _forward(inputs.views, net)
    return OutputSchema(**{k: v.numpy() for k, v in out.items()})


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set,
    vjp_outputs: set,
    cotangent_vector: dict,
):
    net = _net()
    _load_weights(net, inputs.weights)

    w = parameters_to_vector(net.parameters()).detach().clone().requires_grad_(True)
    vector_to_parameters(w, net.parameters())

    v = torch.as_tensor(np.asarray(inputs.views, dtype=np.float32)).unsqueeze(0)
    if "views" in vjp_inputs:
        v.requires_grad_(True)

    out = net(v)
    maps = {
        "basecolor": out["basecolor"][0].permute(1, 2, 0),
        "roughness": out["roughness"][0].permute(1, 2, 0),
        "metallic": out["metallic"][0].permute(1, 2, 0),
    }

    # Contract each requested output against its cotangent, then one backward.
    scalar = None
    for name in vjp_outputs:
        ct = torch.as_tensor(np.asarray(cotangent_vector[name], dtype=np.float32))
        term = (maps[name] * ct).sum()
        scalar = term if scalar is None else scalar + term
    scalar.backward()

    result = {}
    if "weights" in vjp_inputs:
        result["weights"] = w.grad.detach().numpy().astype(np.float32)
    if "views" in vjp_inputs:
        g = v.grad if v.grad is not None else torch.zeros_like(v)
        result["views"] = g[0].detach().numpy().astype(np.float32)
    return result


def abstract_eval(abstract_inputs):
    _, h, w, _ = abstract_inputs.views.shape
    return {
        "basecolor": {"shape": (h, w, 3), "dtype": "float32"},
        "roughness": {"shape": (h, w, 1), "dtype": "float32"},
        "metallic": {"shape": (h, w, 1), "dtype": "float32"},
    }
