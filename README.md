# Cricut Mat Image Rectification & Precision Grid Aligner

Takes photos of artwork on a Cricut 12×12" mat and produces a perspective-corrected, 
grid-aligned 3600×3600 px (300 DPI) square PNG for Cricut Design Space.

## Pipeline

1. **Rough warp** - HSV mat mask + 4 mat corners → 4000×4000 (initialisation only)
2. **Line response** - white top-hat restricted to the green mat, opened with long
   horizontal/vertical kernels (drops paper, tape, text and ruler ticks)
3. **Lattice fit** - finds the 13 H + 13 V one-inch lines as `offset + k·spacing`
4. **Line fit** - sub-pixel, robust straight-line fit per grid line
5. **Homography** - 169 line intersections → RANSAC homography onto an ideal
   300 px/inch lattice; refined twice in the rectified canvas
6. **Phase 2** - smooth 2-D polynomial displacement field from the residuals
   (lens barrel/pincushion, mat bending); occluded areas filled smoothly
7. **Render** - one Lanczos resample straight from the photo to 3600×3600
8. **Orientation** - triangle notch (hanging slot) rotated to the top (`--notch bottom` to flip)
9. **QA** - re-detects the grid in the output and reports deviation from k·300 px,
   plus the effective source resolution (px/inch)

## Usage

```bash
# Basic usage
python main.py -i input.jpg -o output.png

# Homography only (skip residual distortion field)
python main.py -i input.jpg -o output.png --no-phase2

# Put the triangle notch at the bottom instead of the top
python main.py -i input.jpg -o output.png --notch bottom

# Regression test on Test/ images
python -m pytest test_grid_fit.py -v

# Custom DPI
python main.py -i input.jpg -o output.png --dpi 300
```

## Output
- **Size**: 3600 × 3600 pixels
- **DPI**: 300 (12 inches × 300 DPI)
- **Format**: PNG with embedded DPI metadata
- **Content**: Tightly cropped to the 12×12" grid area (no mat border)

## Files
| File | Description |
|------|-------------|
| `main.py` | CLI entry point & pipeline runner |
| `grid_fit.py` | Grid lattice fit, homography, Phase 2 distortion field, render, QA |
| `orientation.py` | Mat detection, corner finding, notch detection & rotation |
| `test_grid_fit.py` | Regression test (alignment thresholds on `Test/` images) |
| `mesh_warper.py` | Legacy per-cell refiner (no longer used) |
| `grid_detector.py` | Legacy grid detection (no longer used) |
| `gui.py` | Interactive GUI (optional) |
| `requirements.txt` | Dependencies |

## Dependencies
```
numpy
opencv-python
scipy
Pillow
scikit-image
```

## Test Images
See `Test/` folder for sample input photos and expected outputs.
