# pbr-from-renders

Recovering PBR material maps (basecolor, roughness, metallic) from six
multi-view images, with a Monte Carlo path tracer as the only supervision —
no ground-truth material labels anywhere.

**Tesseract Hackathon 2026 — Track 05, differentiable graphics & rendering.**

## Architecture

    six target views
        └─> PBRNet (plain torch nn.Module, in-process)
                └─> basecolor / roughness / metallic maps
                        └─> [Tesseract: Mitsuba 3 prb path tracer, container]
                                └─> rendered image
                                        └─> L1 + 0.1 * LPIPS vs target
                                                └─> loss.backward()

One `loss.backward()` crosses two independent autodiff systems: Dr.Jit inside
the renderer container, torch autograd in the network. Neither can express the
other; the VJP protocol is the interface between them.

Only the renderer is containerized. The network stays in-process, following
the pattern in Tesseract's learned-closure demo.

## Layout

    renderer/            Tesseract: scene.xml, tesseract_api.py, config, assets
    pipeline/            plain Python: model, target generation, checks, training
    data/                ground-truth maps + rendered targets
    results/             checkpoints, history, figures

## Status

Scaffold. Not yet executed — see SETUP.md for the ordered gates.

## License

Apache 2.0
