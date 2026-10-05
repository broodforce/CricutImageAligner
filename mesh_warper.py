"""
Module 3: Phase 2 - Per-Cell Homography Refinement

After Phase 1 (global perspective + grid crop), this module refines
alignment by detecting visible grid lines on the green mat and
applying per-cell corrections.

Key insight: Grid lines are only detectable where the green mat is
visible (not covered by artwork). We detect lines in visible areas
and interpolate for occluded positions.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class GridRefiner:
    """
    Phase 2: Refine grid alignment using per-cell homography.
    
    Only detects grid lines in visible green mat areas to avoid
    interference from white artwork.
    """
    
    def __init__(self, image: np.ndarray, cell_size: int = 300, grid_n: int = 12):
        self.image = image
        self.cell_size = cell_size
        self.grid_n = grid_n
        self.size = cell_size * grid_n  # 3600
    
    def refine(self, verbose: bool = False) -> np.ndarray:
        """Run Phase 2 refinement."""
        # Detect line positions using mat-aware detection
        h_lines, v_lines = self._detect_lines_mat_aware()
        
        if h_lines is None or v_lines is None:
            if verbose:
                print("   No reliable grid lines detected, skipping")
            return self.image
        
        ideal = np.arange(13) * self.cell_size
        h_off = h_lines - ideal
        v_off = v_lines - ideal
        
        if verbose:
            print(f"   H offsets: mean={np.mean(h_off):.1f}, std={np.std(h_off):.1f}, max={np.max(np.abs(h_off)):.1f}px")
            print(f"   V offsets: mean={np.mean(v_off):.1f}, std={np.std(v_off):.1f}, max={np.max(np.abs(v_off)):.1f}px")
        
        # Safety: skip if offsets are tiny (< 5px)
        if np.max(np.abs(h_off)) < 5 and np.max(np.abs(v_off)) < 5:
            if verbose:
                print("   Offsets < 5px, no refinement needed")
            return self.image
        
        # Safety: skip if detection is unreliable (high variance = noise)
        # If std > 20px, the detected positions are too inconsistent to trust
        if np.std(h_off) > 20 or np.std(v_off) > 20:
            if verbose:
                print("   ⚠️ Detection unreliable (std > 20px), skipping refinement")
            return self.image
        
        # Apply per-cell corrections
        result = self._apply_corrections(h_lines, v_lines)
        return result
    
    def _detect_lines_mat_aware(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect grid line positions using only green mat areas.
        
        Grid lines on Cricut mats are lighter than the surrounding green.
        We look for local brightness peaks within the green mat,
        ignoring white artwork areas.
        """
        h, w = self.image.shape[:2]
        
        # Create a "green mat" mask - pixels that are greenish (not white artwork)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([90, 255, 255]))
        
        # Grid lines are brighter than surrounding mat
        # Use the green channel or value channel
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Only consider pixels that are on the green mat
        # For each expected line position, search in the mat area only
        
        cs = self.cell_size
        search_range = 50  # Search ±50px around expected position
        
        h_lines = []
        v_lines = []
        
        # Detect horizontal lines (13 lines)
        for i in range(13):
            expected_y = i * cs
            y_start = max(0, expected_y - search_range)
            y_end = min(h, expected_y + search_range)
            
            # Get the strip
            strip = gray[y_start:y_end, :]
            mat_strip = green_mask[y_start:y_end, :]
            
            # Only count bright pixels that are on green mat
            # A grid line pixel is: bright AND on green mat
            line_signal = np.zeros(y_end - y_start)
            for row in range(y_end - y_start):
                mat_pixels = mat_strip[row] > 0
                if mat_pixels.sum() > 100:  # Enough mat visible
                    # Average brightness of mat pixels in this row
                    line_signal[row] = gray[y_start + row][mat_pixels].mean()
            
            if line_signal.max() == 0:
                h_lines.append(float(expected_y))
                continue
            
            # Find peak
            peak = np.argmax(line_signal)
            h_lines.append(y_start + peak)
        
        # Detect vertical lines (13 lines)
        for j in range(13):
            expected_x = j * cs
            x_start = max(0, expected_x - search_range)
            x_end = min(w, expected_x + search_range)
            
            strip = gray[:, x_start:x_end]
            mat_strip = green_mask[:, x_start:x_end]
            
            line_signal = np.zeros(x_end - x_start)
            for col in range(x_end - x_start):
                mat_pixels = mat_strip[:, col] > 0
                if mat_pixels.sum() > 100:
                    line_signal[col] = gray[:, mat_pixels].mean() if False else \
                        np.mean(gray[np.where(mat_strip[:, col] > 0)[0], x_start + col])
            
            if line_signal.max() == 0:
                v_lines.append(float(expected_x))
                continue
            
            peak = np.argmax(line_signal)
            v_lines.append(x_start + peak)
        
        return np.array(h_lines, dtype=np.float64), np.array(v_lines, dtype=np.float64)
    
    def _apply_corrections(self, h_lines: np.ndarray, v_lines: np.ndarray) -> np.ndarray:
        """Apply per-cell homography corrections."""
        result = np.zeros_like(self.image)
        cs = self.cell_size
        n = self.grid_n
        
        for i in range(n):
            for j in range(n):
                # Actual corner positions from detected lines
                actual = np.array([
                    [v_lines[j], h_lines[i]],
                    [v_lines[j+1], h_lines[i]],
                    [v_lines[j+1], h_lines[i+1]],
                    [v_lines[j], h_lines[i+1]]
                ], dtype=np.float32)
                
                # Ideal positions
                ideal = np.array([
                    [j*cs, i*cs],
                    [(j+1)*cs, i*cs],
                    [(j+1)*cs, (i+1)*cs],
                    [j*cs, (i+1)*cs]
                ], dtype=np.float32)
                
                # Compute homography: actual → ideal
                H = cv2.getPerspectiveTransform(actual, ideal)
                
                # Extract source region with padding
                pad = 20
                x_min = max(0, int(min(actual[:,0])) - pad)
                y_min = max(0, int(min(actual[:,1])) - pad)
                x_max = min(self.size, int(max(actual[:,0])) + pad)
                y_max = min(self.size, int(max(actual[:,1])) + pad)
                
                if x_max - x_min < cs or y_max - y_min < cs:
                    # Fallback: direct copy
                    result[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = \
                        self.image[i*cs:(i+1)*cs, j*cs:(j+1)*cs]
                    continue
                
                src = self.image[y_min:y_max, x_min:x_max]
                
                # Adjust points relative to crop
                actual_adj = actual.copy()
                actual_adj[:, 0] -= x_min
                actual_adj[:, 1] -= y_min
                
                H_adj = cv2.getPerspectiveTransform(actual_adj, ideal)
                
                cell = cv2.warpPerspective(src, H_adj, (cs, cs),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REPLICATE)
                
                result[i*cs:(i+1)*cs, j*cs:(j+1)*cs] = cell
        
        return result


def refine_grid(image: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Phase 2 entry point."""
    refiner = GridRefiner(image, cell_size=300, grid_n=12)
    return refiner.refine(verbose=verbose)
