#!/usr/bin/env python3
"""
Cricut Mat Image Rectification & Precision Grid Aligner

1. Fit the 13x13 one-inch grid lattice -> homography onto ideal 300 px/inch grid
2. Phase 2: smooth residual distortion field (lens / mat bending)
3. Single-resample render, orientation (notch at top), QA measurement

Output: 3600x3600 px PNG at 300 DPI (12x12 inches)
"""

import argparse
import os
import sys
import numpy as np
import cv2
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orientation import (detect_mat_corners, detect_triangle_edge, correct_orientation,
                         detect_notch_side, rotate_notch_to_top)
from grid_fit import fit_grid, fit_distortion, render, measure_output, MARGIN, CELL

OUTPUT_SIZE = 3600   # 12 inches * 300 DPI
WARP_SIZE = 4000     # Intermediate warp size (includes mat border)


def load_image(path: str):
    if not os.path.exists(path):
        print(f"  Error: File not found: {path}")
        return None
    image = cv2.imread(path)
    if image is None:
        print(f"  Error: Could not read image: {path}")
    return image


def save_image(image: np.ndarray, output_path: str, dpi: int = 300) -> bool:
    try:
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(output_path, 'PNG', dpi=(dpi, dpi))
        return True
    except Exception as e:
        print(f"  Error saving: {e}")
        return False


def phase1_perspective(image: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Legacy fallback (used only if the grid fit fails).

    Phase 1: Global perspective correction + grid crop + orientation.
    
    1. Detect mat corners
    2. Warp to WARP_SIZE x WARP_SIZE (includes border)
    3. Detect interior grid lines
    4. Extrapolate grid boundaries
    5. Crop to grid area, resize to 3600x3600
    6. Fix orientation (triangle at bottom)
    """
    h, w = image.shape[:2]
    
    # Step 1: Detect mat corners
    corners = detect_mat_corners(image)
    if corners is None:
        if verbose:
            print("   ⚠️ Mat not detected, center-crop fallback")
        side = min(h, w)
        x_off = (w - side) // 2
        y_off = (h - side) // 2
        return cv2.resize(image[y_off:y_off+side, x_off:x_off+side],
                          (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    
    if verbose:
        names = ['TL', 'TR', 'BR', 'BL']
        print("   Mat corners:")
        for i, c in enumerate(corners):
            print(f"     {names[i]}: ({c[0]:.0f}, {c[1]:.0f})")
    
    # Step 2: Warp to WARP_SIZE
    dst_pts = np.array([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]],
                       dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(128, 128, 128))
    
    # Step 3: Detect interior grid lines
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    border = 300  # Skip mat border area
    h_profile = bright[border:WARP_SIZE-border, border:WARP_SIZE-border].sum(axis=1).astype(float)
    v_profile = bright[border:WARP_SIZE-border, border:WARP_SIZE-border].sum(axis=0).astype(float)
    
    h_peaks, _ = find_peaks(h_profile, height=2000, distance=250)
    v_peaks, _ = find_peaks(v_profile, height=2000, distance=250)
    h_peaks += border
    v_peaks += border
    
    # Step 4: Extrapolate grid boundaries
    if len(h_peaks) >= 3 and len(v_peaks) >= 3:
        h_spacing = np.median(np.diff(h_peaks))
        v_spacing = np.median(np.diff(v_peaks))
        
        if verbose:
            print(f"   Grid: {len(h_peaks)}H x {len(v_peaks)}V interior lines")
            print(f"   Spacing: {h_spacing:.0f}px x {v_spacing:.0f}px")
        
        h_first = max(0, h_peaks[0] - h_spacing)
        h_last = min(WARP_SIZE, h_peaks[-1] + h_spacing)
        v_first = max(0, v_peaks[0] - v_spacing)
        v_last = min(WARP_SIZE, v_peaks[-1] + v_spacing)
        
        if verbose:
            print(f"   Grid crop: ({v_first:.0f},{h_first:.0f}) to ({v_last:.0f},{h_last:.0f})")
        
        cropped = warped[int(h_first):int(h_last), int(v_first):int(v_last)]
        result = cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    else:
        if verbose:
            print("   ⚠️ Grid not detected, using full warp")
        result = cv2.resize(warped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Step 5: Fix orientation
    triangle_edge = detect_triangle_edge(result)
    if triangle_edge and triangle_edge != 'bottom':
        if verbose:
            print(f"   Orientation: triangle on '{triangle_edge}' → rotating")
        result = correct_orientation(result, triangle_edge)
    
    return result


def source_dpi(H: np.ndarray) -> float:
    """Lowest effective source resolution (photo px per inch) over the grid."""
    Hi = np.linalg.inv(H)
    worst = np.inf
    for gx in range(0, 13, 3):
        for gy in range(0, 13, 3):
            p = np.float32([[[MARGIN + gx * CELL, MARGIN + gy * CELL]],
                            [[MARGIN + gx * CELL + 1, MARGIN + gy * CELL]],
                            [[MARGIN + gx * CELL, MARGIN + gy * CELL + 1]]])
            q = cv2.perspectiveTransform(p, Hi).reshape(3, 2)
            sx = np.linalg.norm(q[1] - q[0])
            sy = np.linalg.norm(q[2] - q[0])
            worst = min(worst, min(sx, sy) * CELL)
    return float(worst)


def process_image(input_path: str, output_path: str, dpi: int = 300,
                  verbose: bool = True, enable_phase2: bool = True,
                  notch: str = 'top') -> bool:
    """Full pipeline."""

    if verbose:
        print(f"\n1. Loading: {os.path.basename(input_path)}")
    image = load_image(input_path)
    if image is None:
        return False
    h, w = image.shape[:2]
    if verbose:
        print(f"   {w}x{h}")

    if verbose:
        print(f"\n2. Grid fit (13x13 lattice + homography)")
    fit = fit_grid(image, verbose=verbose)

    if fit is None:
        if verbose:
            print("   ⚠️ Grid fit failed, falling back to mat-corner crop")
        result = phase1_perspective(image, verbose=verbose)
    else:
        distortion = None
        if enable_phase2:
            if verbose:
                print(f"\n3. Phase 2: residual distortion field")
            distortion = fit_distortion(fit['h_lines'], fit['v_lines'])
            if verbose:
                print(f"   Field fit rms={distortion['fit_rms']:.2f}px")
            if distortion['fit_rms'] > 3.0:
                if verbose:
                    print("   ⚠️ Field fit poor, using homography only")
                distortion = None

        if verbose:
            print(f"\n4. Rendering {OUTPUT_SIZE}x{OUTPUT_SIZE} (single Lanczos resample)")
        result = render(image, fit['H'], distortion, size=OUTPUT_SIZE)

        notch_target = notch
        notch = detect_notch_side(fit['canvas'], MARGIN, MARGIN + OUTPUT_SIZE)
        if verbose:
            print(f"   Notch: {notch or 'not found'} → placing at {notch_target}")
        result = rotate_notch_to_top(result, notch)
        if notch_target == 'bottom':
            result = cv2.rotate(result, cv2.ROTATE_180)

        qa = measure_output(result)
        src = source_dpi(fit['H'])
        if verbose:
            print(f"\n5. QA: {qa['lines']}/26 lines, deviation rms={qa['rms']:.1f}px "
                  f"p95={qa['p95']:.1f}px max={qa['max']:.1f}px "
                  f"(1px = 1/300\")")
            print(f"   Source resolution: ≥{src:.0f} px/inch")
            if qa['lines'] < 20 or qa['p95'] > 6:
                print("   ⚠️ Grid alignment is poor — check the output")
            if src < 150:
                print(f"   ⚠️ Low source resolution ({src:.0f} px/inch): output will be soft. "
                      "Shoot closer / overhead, or send the photo as a file, not compressed.")

    if verbose:
        print(f"\n6. Saving: {output_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if save_image(result, output_path, dpi):
        if verbose:
            size_kb = os.path.getsize(output_path) / 1024
            print(f"   ✓ {size_kb:.0f} KB, {OUTPUT_SIZE}x{OUTPUT_SIZE}, {dpi} DPI")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Cricut Mat Image Rectification & Precision Grid Aligner')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', required=True, help='Output image path')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI (default: 300)')
    parser.add_argument('--notch', choices=['top', 'bottom'], default='top',
                        help='Where the triangle notch ends up (default: top, per spec)')
    parser.add_argument('--no-phase2', action='store_true', help='Skip residual distortion correction (homography only)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Cricut Mat Image Rectification & Grid Aligner")
    print("=" * 60)
    
    success = process_image(args.input, args.output, dpi=args.dpi,
                            enable_phase2=not args.no_phase2, notch=args.notch)
    
    print("\n" + "=" * 60)
    print(f"  {'✓ Success!' if success else '✗ Failed'}")
    print("=" * 60)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
