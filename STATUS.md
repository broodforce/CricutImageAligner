# Project Status - Cricut Mat Image Rectification & Grid Aligner

**Last updated:** 2026-02-06

## Current State: Phase 1 Complete & Working

All 3 test images produce correct 3600×3600 px (300 DPI) PNG outputs.

### Working Pipeline (`python main.py -i <input> -o <output>`)

```
Input photo → Mat corner detection → Perspective warp (4000×4000) 
→ Grid line detection → Grid crop → Resize (3600×3600) 
→ Orientation fix (triangle at bottom) → Phase 2 (per-cell, currently skipped)
→ Output PNG
```

### Test Results
| Image | Input | Output | Size | Notes |
|-------|-------|--------|------|-------|
| 1 | `Test/WhatsApp Image 2026-10-04 at 10.36.03.jpeg` (1600×1200) | `output_10.36.03.png` | 6834 KB | Overhead shot, triangle rotated from right→bottom |
| 2 | `Test/WhatsApp Image 2026-10-04 at 210.36.03.jpeg` (1200×1600) | `output_210.36.03.png` | 6608 KB | 45° angle, perspective corrected |
| 3 | `Test/WhatsApp Image 2026-10-04 at 310.36.04.jpeg` (1200×1600) | `output_310.36.04.png` | 6815 KB | Moderate angle, perspective corrected |

## Architecture

### Files
```
/Volumes/MacOSExt/Development/CIA/
├── main.py              # CLI + pipeline runner (Phase 1 + Phase 2)
├── orientation.py       # Mat detection, corners, triangle detection, rotation
├── mesh_warper.py       # Phase 2: GridRefiner class (per-cell homography)
├── grid_detector.py     # Legacy grid detection (not used in main pipeline)
├── gui.py               # Legacy GUI (not integrated)
├── requirements.txt     # numpy, opencv-python, scipy, Pillow, scikit-image
├── README.md            # Project docs
├── pi.md                # Original spec
├── Test/                # 3 test JPEG images
└── STATUS.md            # This file
```

### Key Functions
```python
# orientation.py
detect_mat_corners(image) -> Optional[np.ndarray]  # 4x2 [TL,TR,BR,BL]
detect_triangle_edge(warped) -> Optional[str]      # 'top'|'bottom'|'left'|'right'
correct_orientation(warped, edge) -> np.ndarray    # Rotates triangle to bottom

# main.py
phase1_perspective(image) -> np.ndarray            # Full Phase 1
phase2_refinement(image) -> np.ndarray             # Phase 2 (safety-gated)
process_image(input, output, dpi=300) -> bool      # Full pipeline

# mesh_warper.py
GridRefiner(image, cell_size=300, grid_n=12)       # Phase 2 refiner
  .refine(verbose) -> np.ndarray
  ._detect_lines_mat_aware() -> (h_lines, v_lines) # Green-mat-aware detection
  ._apply_corrections(h_lines, v_lines) -> np.ndarray  # Per-cell homography
```

### Phase 1 Details
1. HSV threshold: `inRange([35,20,20], [85,255,255])` → green mat mask
2. Morphology: close(7×7, iter=4) + open(7×7, iter=2)
3. Contour → `approxPolyDP` (eps 0.005→0.05) → 4 corners
4. Corner ordering: min(x+y)=TL, max(x+y)=BR, min(x-y)=TR, max(x-y)=BL
5. Warp to 4000×4000 (WARP_SIZE constant in main.py)
6. Grid detection: brightness threshold (>180), projection profiles, `find_peaks(distance=250)`
7. Skip 300px border on each side for detection (avoids number markings)
8. Extrapolate: first_peak - spacing, last_peak + spacing → grid boundaries
9. Crop + resize to 3600×3600
10. Triangle detection: dark pixel count in center strips of each edge (threshold >200 px)
11. Rotation: right→90°CW, left→90°CCW, top→180°, bottom→none

### Phase 2 Details (Currently Safety-Gated)
- Detects grid lines using green-mat-aware brightness peaks
- Only searches ±50px around expected positions (i*300)
- Uses HSV green mask to ignore white artwork
- **Safety gate**: skips if std(offsets) > 20px (currently triggers for all 3 test images)
- Per-cell: 144 cells, each gets a 4-point perspective transform

## What's Working
- ✅ Perspective correction (overhead + angled shots)
- ✅ Grid crop (excludes mat border/numbers)
- ✅ Orientation fix (triangle detection + rotation)
- ✅ 3600×3600 output at 300 DPI with PNG metadata
- ✅ CLI with `--no-phase2` flag

## What's Not Working / TODO

### Phase 2 (Per-Cell Refinement) - BLOCKED
- **Problem**: Grid line detection unreliable when artwork covers most of the mat
- **Current status**: Safety gate skips refinement for all 3 test images (std > 20px)
- **Root cause**: White artwork paper has similar brightness to grid lines; 
  only ~30-40% of grid lines are visible on green mat
- **Possible fixes**:
  - Only detect lines in the border ring (outer 1-2 cells) where mat is visible
  - Use the known regular spacing to interpolate undetected lines
  - Require minimum N visible lines (e.g., 8+) before attempting refinement
  - Use edge detection (Canny) on green mat instead of brightness threshold
  - Weight detection by number of visible mat pixels in the search strip

### GUI - NOT INTEGRATED
- `gui.py` exists but is broken/not connected to pipeline
- Was supposed to allow interactive grid point verification
- Low priority - CLI works well

### Potential Improvements
- Lens distortion correction (radial/tangential) for wide-angle phone photos
- Multi-photo stitching if mat doesn't fit in one frame
- Batch processing (folder of images → folder of outputs)
- Confidence scoring / quality warning if mat detection is poor

## How to Resume

```bash
cd /Volumes/MacOSExt/Development/CIA

# Run on a single image
python main.py -i "Test/WhatsApp Image 2026-10-04 at 10.36.03.jpeg" -o output_test.png

# Run all 3 test images
for f in "10.36.03" "210.36.03" "310.36.04"; do
  python main.py -i "Test/WhatsApp Image 2026-10-04 at ${f}.jpeg" -o "output_${f}.png"
done

# Dependencies (already installed)
pip install numpy opencv-python scipy Pillow scikit-image
```

## Key Design Decisions
1. **Mat corners (not grid lines) for Phase 1** - More robust for angled shots where grid lines aren't axis-aligned
2. **4000×4000 intermediate warp** - Gives room for border detection before final crop
3. **Triangle for orientation** - Most reliable landmark; numbers/text are ambiguous
4. **Phase 2 safety gate** - Better to skip refinement than apply wrong corrections
5. **`approxPolyDP` over `minAreaRect`** - Handles non-rectangular quadrilaterals from perspective
