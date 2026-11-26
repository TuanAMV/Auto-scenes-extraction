# Third-Party Licenses

This file summarizes major third-party components bundled in this repository.
It is a practical inventory, not legal advice.

## Primary Upstream Projects Used

| Project | URL |
|---|---|
| CLIP | https://github.com/openai/CLIP |
| FG-CLIP | https://github.com/360CVGroup/FG-CLIP |
| VideoJev-4B | https://huggingface.co/MCG-NJU/VideoJev-4B |
| FFmpeg | https://github.com/FFmpeg/FFmpeg |

## Bundled Binaries And Model Artifacts

| Component | Path | Declared License | Evidence |
|---|---|---|---|
| FFmpeg shared build | `models/ffmpeg` | GPLv3 | `models/ffmpeg/LICENSE.txt` |
| FG-CLIP2 model files | `models/qihoo360-fg-clip2-base` | Apache-2.0 (model card declaration) | `models/qihoo360-fg-clip2-base/README.md` |
| openai/clip-vit-large-patch14 model files | `models/openai-clip-vit-large-patch14` | Upstream terms apply (not fully bundled here) | `models/openai-clip-vit-large-patch14/README.md` |
| VideoJev-4B model files | `models/MCG-NJU--VideoJev-4B/snapshots/master` | Apache-2.0 (model card declaration) | `models/MCG-NJU--VideoJev-4B/snapshots/master/LICENSE` |
| aslakey/shot-scale model files | `models/aslakey_shot_scale` | Apache-2.0 (model card declaration) | `models/aslakey_shot_scale/README.md` |

## Embedded Python Environment (Key Packages)

Observed in `python/Lib/site-packages` metadata:

| Package | Declared License |
|---|---|
| torch | BSD-3-Clause |
| torchvision | BSD |
| transformers | Apache-2.0 |
| open_clip_torch | MIT |
| decord | Apache |
| av | BSD-3-Clause |

## Redistribution Notes

1. Keep original license texts and notices for bundled third-party components.
2. If distributing with `models/ffmpeg` (GPLv3 build), follow GPLv3 distribution
   obligations for the redistributed package.
3. Model usage terms can differ from code licenses. Verify upstream model terms
   (including commercial-use restrictions, if any) before distribution.
4. If you add or update dependencies/models, update this file and `NOTICE`.
