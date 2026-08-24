"""Create placeholder textures, ground-truth maps, and the six target views.

Run in two stages:

    python pipeline/generate_targets.py --init      # placeholders only (do first)
    python pipeline/generate_targets.py --render    # ground truth + targets

The --init stage must run BEFORE `tesseract build`, because scene.xml loads
these PNGs and they need to exist inside the image.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "renderer" / "assets"
DATA = ROOT / "data"
RES = 128  # texture resolution; must match training resolution

# Ground truth. Two regions: a metallic band and a dielectric body. Metal vs
# dielectric is a QUALITATIVE difference (metal has no diffuse lobe, coloured
# specular), so the network must find the boundary, not just a scalar.
GT_BODY = dict(basecolor=(0.70, 0.20, 0.10), roughness=0.45, metallic=0.05)
GT_BAND = dict(basecolor=(0.05, 0.05, 0.05), roughness=0.15, metallic=0.90)


def region_mask(res):
    """Horizontal band across the middle third of UV space.

    Replace with Spot's own spot pattern (threshold spot_texture.png) once the
    simple version works -- debug one thing at a time.
    """
    v = np.linspace(0, 1, res)[:, None] * np.ones((1, res))
    return ((v > 0.33) & (v < 0.66)).astype(np.float32)[..., None]


def build_gt(res):
    m = region_mask(res)
    bc = np.array(GT_BODY["basecolor"], np.float32) * (1 - m) + \
         np.array(GT_BAND["basecolor"], np.float32) * m
    ro = np.full((res, res, 1), GT_BODY["roughness"], np.float32) * (1 - m) + \
         np.full((res, res, 1), GT_BAND["roughness"], np.float32) * m
    me = np.full((res, res, 1), GT_BODY["metallic"], np.float32) * (1 - m) + \
         np.full((res, res, 1), GT_BAND["metallic"], np.float32) * m
    return bc.astype(np.float32), ro.astype(np.float32), me.astype(np.float32)


def save_png(arr, path):
    from PIL import Image

    a = np.clip(arr, 0, 1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def do_init():
    ASSETS.mkdir(parents=True, exist_ok=True)
    mid = np.full((RES, RES, 3), 0.5, np.float32)
    save_png(mid, ASSETS / "init_basecolor.png")
    save_png(mid, ASSETS / "init_roughness.png")
    save_png(mid, ASSETS / "init_metallic.png")
    print(f"Placeholders written at {RES}x{RES}. Now run: tesseract build renderer/")


def do_render(spp):
    import mitsuba as mi

    mi.set_variant("llvm_ad_rgb")
    from renderer.tesseract_api import (  # noqa: E402
        KEY_BASECOLOR, KEY_METALLIC, KEY_ROUGHNESS,
    )

    DATA.mkdir(exist_ok=True)
    (DATA / "targets").mkdir(exist_ok=True)

    bc, ro, me = build_gt(RES)
    np.save(DATA / "gt_basecolor.npy", bc)
    np.save(DATA / "gt_roughness.npy", ro)
    np.save(DATA / "gt_metallic.npy", me)

    scene = mi.load_file(str(ROOT / "renderer" / "scene.xml"))
    params = mi.traverse(scene)
    params[KEY_BASECOLOR] = mi.TensorXf(bc)
    params[KEY_ROUGHNESS] = mi.TensorXf(ro)
    params[KEY_METALLIC] = mi.TensorXf(me)
    params.update()

    # High spp for targets: Mitsuba's docs warn that a noisy reference image
    # perturbs the optimization. Training uses far fewer samples.
    for i in range(6):
        img = mi.render(scene, sensor=scene.sensors()[i], spp=spp, seed=1000 + i)
        mi.Bitmap(img).write(str(DATA / "targets" / f"view_{i:02d}.exr"))
        print(f"  view_{i:02d}.exr  mean={float(np.array(img).mean()):.4f}")

    print(f"\n6 targets at {spp} spp. If any mean is ~0.0 the object is not in frame.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--spp", type=int, default=256)
    a = p.parse_args()
    if a.init:
        do_init()
    elif a.render:
        do_render(a.spp)
    else:
        p.print_help()
