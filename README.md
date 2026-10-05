# Cricut Mat Image Rectification & Precision Grid Aligner

Takes photos of artwork on a Cricut 12×12" mat and produces a perspective-corrected, 
grid-aligned 3600×3600 px (300 DPI) square PNG for Cricut Design Space.

## Pipeline

### Phase 1: Global Perspective Correction
1. **Mat Detection** - HSV color thresholding finds the green mat region
2. **Corner Detection** - Contour + `approxPolyDP` finds 4 mat corners (works for any angle)
3. **Perspective Warp** - `cv2.getPerspectiveTransform` maps mat to 4000×4000
4. **Grid Line Detection** - Projection profiles find interior grid lines
5. **Grid Crop** - Extrapolates grid boundaries, crops to 12×12" grid area
6. **Resize** - Scales to 3600×3600 (12in × 300 DPI)
7. **Orientation Fix** - Detects triangle (hanging slot), rotates so it's at bottom

### Phase 2: Per-Cell Homography Refinement
1. Splits 3600×3600 into 12×12 cells of 300×300px
2. Detects actual grid line positions (using green mat areas only)
3. Computes per-cell perspective transform (actual → ideal positions)
4. **Safety**: Skips refinement if detection is unreliable (std > 20px)

## Usage

```bash
# Basic usage
python main.py -i input.jpg -o output.png

# Skip Phase 2 (faster, for heavily occluded mats)
python main.py -i input.jpg -o output.png --no-phase2

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
| `orientation.py` | Mat detection, corner finding, orientation fix |
| `mesh_warper.py` | Phase 2 per-cell homography refinement |
| `grid_detector.py` | Grid line detection utilities |
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
