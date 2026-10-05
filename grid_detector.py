"""
Module 2: Line Detection & RANSAC Grid Initialization

Detects the 13x13 grid within the mat region for fine correction.
If grid lines are not visible (e.g., obscured by artwork), returns
a regular grid based on the detected mat boundaries.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class GridDetector:
    """Detects grid lines within the mat region."""
    
    GRID_SIZE = 13
    
    def __init__(self, image: np.ndarray, mat_corners: Optional[np.ndarray] = None):
        """
        Args:
            image: BGR numpy array (the full photo)
            mat_corners: 4x2 array of mat corners [TL, TR, BR, BL]
        """
        self.image = image
        self.h, self.w = image.shape[:2]
        self.mat_corners = mat_corners
        self.horizontal_lines: List[float] = []
        self.vertical_lines: List[float] = []
        self.grid_nodes: List[Tuple[float, float, bool]] = []
    
    def detect_grid_lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect grid lines within the mat region.
        
        Returns:
            Tuple of (grid_points, confidence_mask)
            - grid_points: 169x2 array of (x, y) in input image coordinates
            - confidence_mask: 169 boolean array (True = detected, False = interpolated)
        """
        # Work in the mat region
        if self.mat_corners is not None:
            # Crop to mat region for detection
            mat_region = self._get_mat_region()
        else:
            mat_region = self.image
            self.mat_corners = np.array([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]], dtype=np.float32)
        
        # Try to detect actual grid lines
        detected = self._try_detect_lines(mat_region)
        
        if detected and len(self.horizontal_lines) >= 5 and len(self.vertical_lines) >= 5:
            # Build grid from detected lines
            self._build_grid_from_lines()
            return self._get_grid_with_offsets()
        else:
            # No visible grid - create regular grid mapped to mat corners
            self._create_regular_grid()
            return self._get_grid_with_offsets()
    
    def _get_mat_region(self) -> np.ndarray:
        """Get the mat region cropped and straightened."""
        # For simplicity, use the bounding box of the mat corners
        x_min = int(self.mat_corners[:, 0].min())
        x_max = int(self.mat_corners[:, 0].max())
        y_min = int(self.mat_corners[:, 1].min())
        y_max = int(self.mat_corners[:, 1].max())
        
        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.w, x_max)
        y_max = min(self.h, y_max)
        
        return self.image[y_min:y_max, x_min:x_max]
    
    def _try_detect_lines(self, mat_region: np.ndarray) -> bool:
        """Try to detect grid lines. Returns True if lines found."""
        gray = cv2.cvtColor(mat_region, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold for white/light lines on green background
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )
        
        # Also try a simple threshold for very white lines
        _, white_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Combine
        combined = cv2.bitwise_and(thresh, white_thresh)
        
        # Morphological operations
        kernel_h = np.ones((1, 30), np.uint8)  # Horizontal
        kernel_v = np.ones((30, 1), np.uint8)  # Vertical
        
        # Detect horizontal lines
        h_lines_img = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_h)
        # Detect vertical lines  
        v_lines_img = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_v)
        
        # Find horizontal lines by row projection
        h_proj = np.sum(h_lines_img, axis=1)
        h_threshold = h_proj.max() * 0.3
        self.horizontal_lines = self._find_peaks(h_proj, h_threshold, mat_region.shape[0])
        
        # Find vertical lines by column projection
        v_proj = np.sum(v_lines_img, axis=0)
        v_threshold = v_proj.max() * 0.3
        self.vertical_lines = self._find_peaks(v_proj, v_threshold, mat_region.shape[1])
        
        return len(self.horizontal_lines) >= 5 and len(self.vertical_lines) >= 5
    
    def _find_peaks(self, projection: np.ndarray, threshold: float, total_size: int) -> List[float]:
        """Find peaks in a projection profile."""
        if projection.max() == 0:
            return []
        
        # Normalize
        normalized = projection.astype(float) / projection.max()
        
        # Find peaks
        peaks = []
        in_peak = False
        peak_start = 0
        
        for i, val in enumerate(normalized):
            if val > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif val < threshold * 0.5 and in_peak:
                in_peak = False
                peak_center = (peak_start + i) // 2
                peaks.append(peak_center)
        
        if in_peak:
            peaks.append((peak_start + len(normalized)) // 2)
        
        # Filter: we expect roughly equal spacing
        if len(peaks) < 4:
            return []
        
        # Verify roughly equal spacing
        diffs = np.diff(peaks)
        mean_diff = np.mean(diffs)
        if mean_diff < 10:  # Lines too close together
            return []
        
        # Keep peaks with reasonable spacing
        valid_peaks = []
        for p in peaks:
            if not valid_peaks or abs(p - valid_peaks[-1]) > mean_diff * 0.5:
                valid_peaks.append(p)
        
        return valid_peaks
    
    def _build_grid_from_lines(self):
        """Build grid nodes from detected line positions."""
        # Offset by mat region position
        x_min = int(self.mat_corners[:, 0].min())
        y_min = int(self.mat_corners[:, 1].min())
        
        self.grid_nodes = []
        for hy in self.horizontal_lines:
            for vx in self.vertical_lines:
                x = vx + x_min
                y = hy + y_min
                self.grid_nodes.append((x, y, True))  # True = detected
    
    def _create_regular_grid(self):
        """Create a regular grid mapped to the mat corners."""
        # Interpolate positions along the edges
        tl = self.mat_corners[0]
        tr = self.mat_corners[1]
        br = self.mat_corners[2]
        bl = self.mat_corners[3]
        
        self.grid_nodes = []
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                t = j / (self.GRID_SIZE - 1)  # horizontal fraction
                u = i / (self.GRID_SIZE - 1)  # vertical fraction
                
                # Bilinear interpolation of corners
                top = tl * (1 - t) + tr * t
                bottom = bl * (1 - t) + br * t
                point = top * (1 - u) + bottom * u
                
                self.grid_nodes.append((point[0], point[1], False))  # False = interpolated
    
    def _get_grid_with_offsets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return grid points in input image coordinates."""
        points = np.array([[x, y] for x, y, _ in self.grid_nodes], dtype=np.float32)
        confidence = np.array([d for _, _, d in self.grid_nodes], dtype=bool)
        
        return points, confidence
    
    def get_line_count(self) -> Tuple[int, int]:
        return len(self.horizontal_lines), len(self.vertical_lines)


def detect_grid_nodes(image: np.ndarray, 
                      mat_corners: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Public API: Detect grid nodes.
    
    Args:
        image: BGR numpy array
        mat_corners: 4x2 array of mat corners
        
    Returns:
        Tuple of (grid_points 169x2, confidence_mask 169 bool)
    """
    detector = GridDetector(image, mat_corners)
    return detector.detect_grid_lines()
