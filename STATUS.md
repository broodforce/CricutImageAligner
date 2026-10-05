# Project Status - Cricut Mat Image Rectification & Grid Aligner

**Last updated:** 2026-10-05

## Current State: Grid-fit pipeline working, verified by QA metric

All 3 test images produce 3600×3600 px (300 DPI) PNGs whose re-detected grid
lines sit within ~2 px (≈0.007") RMS of the ideal k·300 px positions.

### Pipeline (`python main.py -i <input> -o <output>`)

```
Input photo → mat corners → rough warp (4000², init only)
→ line response (green-mat top-hat + long morph opening)
→ 13-line lattice fit → per-line sub-pixel fit → RANSAC homography (×3 passes)
→ Phase 2: 2-D polynomial residual field → single Lanczos render to 3600²
→ notch → top → QA (re-detect grid, source px/inch) → PNG
```

### Test Results (QA = deviation of output grid lines from k·300 px)
| Image | Lines | RMS | p95 | Max | Source res. | Notch |
|-------|-------|-----|-----|-----|-------------|-------|
| `10.36.03` (overhead, 1600×1200) | 26/26 | 1.5 px | 2.9 px | 4.0 px | ≥84 px/in | right → top |
| `210.36.03` (45°, 1200×1600) | 26/26 | 2.1 px | 3.6 px | 7.0 px | ≥32 px/in | bottom → top |
| `310.36.04` (moderate, 1200×1600) | 26/26 | 1.8 px | 2.9 px | 7.2 px | ≥39 px/in | bottom → top |

Before this change (old Phase 1), the same measurement showed one axis
squashed by 7–13% (grid 3130–3350 px instead of 3600, mat border included in
the crop), ~40 px offset on the other axis, and the notch rotated to the bottom.

`python -m pytest test_grid_fit.py` enforces RMS < 2.5 px, p95 < 4 px,
26/26 lines in the fit, and notch found.

## Architecture

```
main.py           CLI + pipeline; QA print-out; legacy phase1_perspective() fallback
grid_fit.py       line_responses, lattice search, line fits, fit_grid(),
                  fit_distortion() (Phase 2), render(), measure_output() (QA)
orientation.py    detect_mat_corners, detect_notch_side, rotate_notch_to_top
                  (+ legacy detect_triangle_edge / correct_orientation)
test_grid_fit.py  regression test on Test/*.jpeg
mesh_warper.py    legacy per-cell refiner (unused)
grid_detector.py  legacy (unused)
gui.py            legacy GUI (not integrated)
```

### Key details
- **Canvas space**: ideal grid at 300 px/inch with a 300 px margin (4200²),
  so the mat border (notch) stays visible for orientation detection.
- **Line response**: `tophat(gray, 21×21)` masked to dilated green, then
  `open` with 151×1 / 1×151 kernels → only long thin lines survive.
- **Lattice search**: brute-force `(offset, spacing)` maximising projection
  support of exactly 13 lines; rough pass spacing ∈ [R/15.5, R/12.2].
- **Line fit**: one sub-pixel centroid per column/row inside a band,
  robust LSQ (reject at 12/5/3 px).
- **Phase 2**: degree-4 bivariate polynomial for dx (from vertical lines) and
  dy (from horizontal lines); skipped if field fit RMS > 3 px.
- **Render**: output px → canvas (+ field) → photo via H⁻¹ → one `cv2.remap`
  with `INTER_LANCZOS4` (old pipeline resampled twice with bilinear).
- **Notch**: enclosed, non-white, non-green hole centred on one margin edge.

## Open Questions / TODO
- **Confirm orientation**: notch now goes to the top (pi.md spec). The artwork's
  handwriting then reads upside down; if Design Space expects the opposite,
  use `--notch bottom` (or change the default). Verify with a test cut.
- **Source resolution is the main quality limit**: all test photos are WhatsApp
  compressed (32–84 px/inch on the mat vs 300 px/inch output). Use full-res
  originals (send as document) and shoot overhead.
- GUI verification (pi.md module 4) not integrated.
- Batch processing (folder → folder).
- Optional white balance / colour normalisation from mat + paper.

## How to Resume

```bash
cd /Volumes/MacOSExt/Development/CIA
for f in "10.36.03" "210.36.03" "310.36.04"; do
  python main.py -i "Test/WhatsApp Image 2026-10-04 at ${f}.jpeg" -o "output_${f}.png"
done
python -m pytest test_grid_fit.py -v
```

## Key Design Decisions
1. **Grid lines, not mat corners, define the geometry** - mat corners are
   rounded and the mat margin is asymmetric; corners are only an initial guess.
2. **Fit exactly 13 lines as a lattice** - prevents off-by-one-cell crops.
3. **Only thin lines on green count** - paper, tape and white printing are
   rejected by the green mask + morphology, so occlusion is handled naturally.
4. **Smooth global field for Phase 2** - occluded nodes are filled from visible
   ones without per-cell instability.
5. **Measure the output** - QA re-detects the grid in the final image so
   alignment claims are backed by numbers.
