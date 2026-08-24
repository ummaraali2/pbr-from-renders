"""Validate the renderer VJP before spending any money on GPU training.

    python pipeline/gradient_check.py

Two independent checks:
  1. Endpoint presence  -- is vector_jacobian_product actually registered?
  2. AD vs finite differences on a scalar loss.

Pass criterion: relative error under ~1%. Sign disagreement means the VJP is
wrong and no amount of training tuning will help.

Monte Carlo note: gradients are stochastic. Use a FIXED seed on both sides and
high spp here, or FD noise will swamp the comparison.
"""

import numpy as np
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

RES = 128
SPP = 128      # high on purpose: this is a correctness test, not training
SEED = 42
IMAGE = "pbr-renderer"


def const_maps(bc=0.5, ro=0.4, me=0.1, res=RES):
    return (
        torch.full((res, res, 3), bc, dtype=torch.float32),
        torch.full((res, res, 1), ro, dtype=torch.float32),
        torch.full((res, res, 1), me, dtype=torch.float32),
    )


def main():
    tess = Tesseract.from_image(IMAGE)
    tess.serve()
    try:
        print(f"Endpoints: {tess.available_endpoints}")
        if "vector_jacobian_product" not in tess.available_endpoints:
            print("\nFAIL: VJP endpoint missing.")
            print("The function in tesseract_api.py must be named exactly")
            print("'vector_jacobian_product' -- not 'vjp'.")
            return

        bc, ro, me = const_maps()
        bc.requires_grad_(True)

        out = apply_tesseract(
            tess,
            {"basecolor": bc, "roughness": ro, "metallic": me,
             "camera_index": 0, "spp": SPP, "seed": SEED},
        )
        loss = out["image"].sum()
        loss.backward()

        if bc.grad is None:
            print("\nFAIL: basecolor.grad is None -- graph is disconnected.")
            return
        gn = float(bc.grad.norm())
        print(f"\nloss = {float(loss):.6f}   |grad| = {gn:.6e}")
        if gn == 0.0:
            print("FAIL: zero gradient. Check PARAMS.update() in the VJP,")
            print("and confirm the KEY_* names via pipeline/dump_params.py.")
            return

        # --- finite differences on a single scalar knob ---
        # Perturb the whole basecolor map uniformly; AD equivalent is the sum
        # of all basecolor gradient entries.
        eps = 1e-3
        with torch.no_grad():
            def loss_at(delta):
                b, r, m = const_maps(bc=0.5 + delta)
                o = tess.apply({"basecolor": b.numpy(), "roughness": r.numpy(),
                                "metallic": m.numpy(), "camera_index": 0,
                                "spp": SPP, "seed": SEED})
                return float(np.asarray(o["image"]).sum())

            fd = (loss_at(eps) - loss_at(-eps)) / (2 * eps)

        ad = float(bc.grad.sum())
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        print(f"\n{'AD':>16s} {'FD':>16s} {'RelErr':>10s}")
        print(f"{ad:16.6e} {fd:16.6e} {rel:10.2%}")

        if rel < 0.01:
            print("\nPASS -- proceed to training.")
        elif np.sign(ad) != np.sign(fd):
            print("\nFAIL -- signs disagree. VJP is wrong. Do not train.")
        else:
            print("\nMARGINAL -- raise SPP and re-run before trusting it.")
    finally:
        tess.teardown()


if __name__ == "__main__":
    main()
