"""Print every Mitsuba scene parameter key.

RUN THIS FIRST, before anything else. The three KEY_* constants in
renderer/tesseract_api.py are guesses until this confirms them. Wrong key ->
silent zero gradients, not an error.

    python pipeline/dump_params.py
"""

from pathlib import Path

import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

SCENE = Path(__file__).parent.parent / "renderer" / "scene.xml"

scene = mi.load_file(str(SCENE))
params = mi.traverse(scene)

print(f"Scene: {SCENE}")
print(f"Sensors: {len(scene.sensors())} (need 6)\n")

print("--- keys containing 'bsdf' ---")
for k in params.keys():
    if "bsdf" in k:
        try:
            v = params[k]
            shape = getattr(v, "shape", "scalar")
        except Exception as exc:  # noqa: BLE001
            shape = f"<{exc}>"
        print(f"  {k:<55s} {shape}")

# Print shapes for the three keys we care about
print("\n--- SHAPE VERIFICATION (needed for TensorXf fix) ---")
KEY_BASECOLOR = "object.bsdf.base_color.data"
KEY_ROUGHNESS = "object.bsdf.roughness.data"
KEY_METALLIC = "object.bsdf.metallic.data"

for key in [KEY_BASECOLOR, KEY_ROUGHNESS, KEY_METALLIC]:
    if key in params:
        print(f"{key}: shape = {params[key].shape}")
    else:
        print(f"{key}: NOT FOUND")

print("\n--- all keys ---")
for k in params.keys():
    print(f"  {k}")

print("\nCopy the base_color / roughness / metallic keys into")
print("renderer/tesseract_api.py (KEY_BASECOLOR, KEY_ROUGHNESS, KEY_METALLIC).")
