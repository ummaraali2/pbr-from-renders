"""Train PBRNet through the containerized path tracer.

    python pipeline/train.py --smoke      # 5 iterations, gate check
    python pipeline/train.py              # full run

Composition pattern follows Tesseract's learned-closure demo: the network is
plain in-process torch, the solver is a served container, and apply_tesseract
bridges them so loss.backward() dispatches a VJP over HTTP.

CRITICAL: use apply_tesseract() inside the loop, never tess.apply().
tess.apply() returns decoded NumPy -- no autograd connection -- and produces a
loss that prints fine and never decreases.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

from model import PBRNet, to_renderer_layout

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
IMAGE = "pbr-renderer"
RES = 128


def tonemap(x):
    """Reinhard. Smooth and monotonic, so gradients survive.
    np.clip would zero the gradient wherever a highlight saturates."""
    return x / (x + 1.0)


def load_targets():
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    views = []
    for i in range(6):
        bitmap = mi.Bitmap(str(DATA / "targets" / f"view_{i:02d}.exr"))
        img = np.array(bitmap, dtype=np.float32)
        views.append(tonemap(img))
    return torch.from_numpy(np.stack(views))          # [6,H,W,3]


def load_gt():
    return {k: np.load(DATA / f"gt_{k}.npy") for k in
            ("basecolor", "roughness", "metallic")}


def recovery_error(maps, gt):
    out = {}
    for k, v in gt.items():
        pred = maps[k].detach().cpu().numpy()
        out[k] = float(np.abs(pred - v).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lpips-weight", type=float, default=0.1)
    ap.add_argument("--network-mode", choices=["inprocess", "tesseract"],
                    default="inprocess",
                    help="inprocess = plain torch (debug first); "
                         "tesseract = two-container mode (required for submission)")
    args = ap.parse_args()

    n_iter = 5 if args.smoke else args.iters
    cams = [0] if args.smoke else list(range(6))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "checkpoints").mkdir(exist_ok=True)

    targets = load_targets()
    gt = load_gt()
    views_in = targets.unsqueeze(0)                    # [1,6,H,W,3]

    base_ch = 32 if args.network_mode == "inprocess" else 16
    net = PBRNet(base_ch=base_ch)
    print(f"PBRNet parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    perceptual = None
    if args.lpips_weight > 0 and not args.smoke:
        import lpips
        perceptual = lpips.LPIPS(net="vgg")

    def compute_loss(render, target):
        # Both already tonemapped to [0,1]. render is a torch tensor carrying
        # graph history from apply_tesseract -- do NOT re-wrap it.
        l1 = torch.mean(torch.abs(render - target))
        if perceptual is None:
            return l1
        r = render.permute(2, 0, 1).unsqueeze(0) * 2 - 1   # LPIPS wants [-1,1]
        t = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return l1 + args.lpips_weight * perceptual(r, t).squeeze()

    tess = Tesseract.from_image(IMAGE)
    tess.serve()
    history = []
    try:
        net_tess = None
        if args.network_mode == "tesseract":
            net_tess = Tesseract.from_image("pbr-network")
            net_tess.serve()
            print("Two-Tesseract mode: pbr-network -> pbr-renderer")

        for it in range(n_iter):
            opt.zero_grad()
            if net_tess is None:
                maps = to_renderer_layout(net(views_in))
            else:
                # Weights cross the boundary as a differentiable input, since
                # Tesseract endpoints are stateless. Gradient comes back in the
                # VJP and lands on `wvec`, which the optimizer then steps.
                from torch.nn.utils import parameters_to_vector
                wvec = parameters_to_vector(net.parameters())
                maps = apply_tesseract(
                    net_tess, {"views": targets, "weights": wvec})

            total = 0.0
            for c in cams:
                out = apply_tesseract(
                    tess,
                    {"basecolor": maps["basecolor"],
                     "roughness": maps["roughness"],
                     "metallic": maps["metallic"],
                     "camera_index": c,
                     "spp": args.spp,
                     "seed": it * 100 + c},   # vary seed: fixed seed turns MC
                )                             # noise into systematic bias
                total = total + compute_loss(tonemap(out["image"]), targets[c])

            total.backward()

            gn = float(torch.sqrt(sum(p.grad.pow(2).sum()
                                      for p in net.parameters()
                                      if p.grad is not None)))
            opt.step()

            err = recovery_error(maps, gt)
            history.append({"iter": it, "loss": float(total), "grad_norm": gn, **err})

            if it % (1 if args.smoke else 25) == 0:
                print(f"[{it:4d}] loss={float(total):.5f} |g|={gn:.4e} "
                      f"bc={err['basecolor']:.4f} ro={err['roughness']:.4f} "
                      f"me={err['metallic']:.4f}")

            if gn == 0.0:
                print("\nSTOP: grad_norm is exactly zero. The chain is broken.")
                print("Check that apply_tesseract (not tess.apply) is used above,")
                print("and re-run pipeline/gradient_check.py.")
                return

            if not args.smoke and it % 199 == 0:
                torch.save({"iter": it, "model": net.state_dict(),
                            "opt": opt.state_dict()},
                           RESULTS / "checkpoints" / f"ckpt_{it:04d}.pt")

        (RESULTS / "history.json").write_text(json.dumps(history, indent=2))
        if args.smoke:
            print("\nSMOKE PASS: gradients flow. Proceed to the full run.")
    finally:
        tess.teardown()
        if 'net_tess' in dir() and net_tess is not None:
            net_tess.teardown()


if __name__ == "__main__":
    main()
