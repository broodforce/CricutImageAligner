"""
Module 1: Mat Detection & Orientation

Detects the green Cricut mat, finds its 4 corners, and determines
physical orientation using the hanging slot (triangle) cutout.
"""

import cv2
import numpy as np
from typing import Optional


def detect_mat_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the 4 corners of the Cricut mat.
    
    Uses HSV color thresholding + contour + polygon approximation.
    Works for both overhead and angled shots.
    
    Returns:
        4x2 float32 array [TL, TR, BR, BL] or None
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mat_mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
    
    kernel = np.ones((7, 7), np.uint8)
    mat_mask = cv2.morphologyEx(mat_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mat_mask = cv2.morphologyEx(mat_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    coverage = mat_mask.sum() / (255 * image.shape[0] * image.shape[1])
    if coverage < 0.03:
        return None
    
    contours, _ = cv2.findContours(mat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None
    
    # Approximate to quadrilateral
    peri = cv2.arcLength(largest, True)
    pts = None
    for eps_factor in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(largest, eps_factor * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            break
    
    if pts is None:
        rect = cv2.minAreaRect(largest)
        pts = cv2.boxPoints(rect).astype(np.float32)
    
    return _order_corners(pts)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL] using sum/diff heuristics."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]    # TL
    ordered[2] = pts[np.argmax(s)]    # BR
    ordered[1] = pts[np.argmin(d)]    # TR
    ordered[3] = pts[np.argmax(d)]    # BL
    return ordered


def detect_mat_mask(image: np.ndarray) -> np.ndarray:
    """Get binary mask of the mat region."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mat_mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
    kernel = np.ones((7, 7), np.uint8)
    mat_mask = cv2.morphologyEx(mat_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mat_mask = cv2.morphologyEx(mat_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mat_mask


def detect_triangle_edge(warped: np.ndarray) -> Optional[str]:
    """
    Detect which edge of the warped image has the triangle (hanging slot).
    
    The triangle is a small dark cutout at the center of one mat edge.
    
    Returns: 'top', 'bottom', 'left', 'right', or None
    """
    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    cx = range(int(w * 0.35), int(w * 0.65))
    cy = range(int(h * 0.35), int(h * 0.65))
    
    strips = {
        'top': gray[30:250, cx],
        'bottom': gray[h - 250:h - 30, cx],
        'left': gray[cy, 30:250],
        'right': gray[cy, w - 250:w - 30]
    }
    
    scores = {}
    for name, strip in strips.items():
        _, dark = cv2.threshold(strip, 100, 255, cv2.THRESH_BINARY_INV)
        scores[name] = np.sum(dark > 0)
    
    max_edge = max(scores, key=scores.get)
    if scores[max_edge] > 200:
        return max_edge
    return None


def correct_orientation(warped: np.ndarray, triangle_edge: Optional[str]) -> np.ndarray:
    """Rotate so the triangle (hanging slot) is at the bottom.
    
    Rotation logic:
    - Triangle on RIGHT  → rotate 90° CW  → right edge becomes bottom
    - Triangle on LEFT   → rotate 90° CCW → left edge becomes bottom
    - Triangle on TOP    → rotate 180°    → top edge becomes bottom
    - Triangle on BOTTOM → no rotation needed
    """
    if triangle_edge is None or triangle_edge == 'bottom':
        return warped
    elif triangle_edge == 'top':
        return cv2.rotate(warped, cv2.ROTATE_180)
    elif triangle_edge == 'right':
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    elif triangle_edge == 'left':
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return warped


def detect_notch_side(canvas: np.ndarray, grid_lo: int, grid_hi: int) -> Optional[str]:
    """
    Find the triangle notch (hanging slot) in a grid-rectified canvas.

    The notch is an enclosed, grey (floor-coloured) hole in the mat margin,
    centred on one edge. Floor touching the canvas border and white mat
    printing (logo, numbers) are rejected.

    Returns: 'top', 'bottom', 'left', 'right', or None
    """
    size = canvas.shape[0]
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (green == 0).astype(np.uint8))

    mid = size / 2
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8000 or x == 0 or y == 0 or x + w >= size or y + h >= size:
            continue
        cx, cy = centroids[i]
        if hsv[..., 2][labels == i].mean() > 200:       # white printing
            continue
        if cy < grid_lo and abs(cx - mid) < 300 and w > h:
            side = 'top'
        elif cy > grid_hi and abs(cx - mid) < 300 and w > h:
            side = 'bottom'
        elif cx < grid_lo and abs(cy - mid) < 300 and h > w:
            side = 'left'
        elif cx > grid_hi and abs(cy - mid) < 300 and h > w:
            side = 'right'
        else:
            continue
        if area > best_area:
            best, best_area = side, area
    return best


def rotate_notch_to_top(image: np.ndarray, notch_side: Optional[str]) -> np.ndarray:
    """Rotate so the triangle notch (hanging slot) is at the top (pi.md spec).

    With the notch at the top the inch ruler reads 1..12 left-to-right along
    the top, matching the Cricut Design Space mat view.
    """
    if notch_side == 'bottom':
        return cv2.rotate(image, cv2.ROTATE_180)
    if notch_side == 'right':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if notch_side == 'left':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


class MatDetector:
    """Wrapper for backward compatibility."""
    
    def __init__(self, image: np.ndarray):
        self.image = image
        self.mat_corners = None
        self.mat_mask = None
    
    def detect_mat(self) -> Optional[np.ndarray]:
        self.mat_corners = detect_mat_corners(self.image)
        self.mat_mask = detect_mat_mask(self.image)
        return self.mat_corners
