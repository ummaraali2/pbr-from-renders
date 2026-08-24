# pbr-from-renders

Recovering PBR material maps (basecolor, roughness, metallic) from six
multi-view images, with a Monte Carlo path tracer as the only supervision —
no ground-truth material labels anywhere.

**Tesseract Hackathon 2026 — Track 05, differentiable graphics & rendering.**

## Architecture

    six target views
        └─> [Tesseract A: PBRNet U-Net, PyTorch autograd]
                └─> basecolor / roughness / metallic maps
                        └─> [Tesseract B: Mitsuba 3 PRB path tracer, Dr.Jit]
                                └─> rendered image
                                        └─> L1 + 0.1 * LPIPS vs target
                                                └─> loss.backward()

**Two composed Tesseracts with end-to-end gradients:**
- **Tesseract A (network/)**: PyTorch U-Net predicting material maps
- **Tesseract B (renderer/)**: Mitsuba 3 differentiable renderer using Dr.Jit

One `loss.backward()` crosses two independent autodiff systems and two
containers. The boundary is differentiation strategy: PyTorch autograd cannot
express Dr.Jit's path-replay backpropagation, and vice versa. The Tesseract
VJP protocol bridges them, enabling independent evolution of each component.

## Layout

    renderer/            Tesseract: scene.xml, tesseract_api.py, config, assets
    pipeline/            plain Python: model, target generation, checks, training
    data/                ground-truth maps + rendered targets
    results/             checkpoints, history, figures

## Status

Scaffold. Not yet executed — see SETUP.md for the ordered gates.

## License

Apache 2.0
