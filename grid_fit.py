"""
Grid lattice fitting: find the 13x13 one-inch grid lines on the green mat
and compute a homography that maps them onto an ideal 300 px/inch lattice.

Pipeline (all in "canvas" space = ideal grid + MARGIN on every side):
  1. Rough warp from mat corners (initialisation only).
  2. Line response: white top-hat restricted to the green mat, opened with
     long horizontal / vertical kernels (removes paper, tape, text, ticks).
  3. Lattice search: offset + k*spacing (k = 0..12) on the projection profiles.
  4. Per-line sub-pixel fit (one sample per column/row, robust LSQ).
  5. Line intersections -> RANSAC homography onto the ideal lattice.
  6. Repeat 2-5 once in the refined canvas for a tight fit.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from typing import Optional, Tuple, Dict

from orientation import detect_mat_corners

GRID_N = 12          # 12 x 12 one-inch cells
CELL = 300           # px per inch at 300 DPI
MARGIN = 300         # canvas margin around the grid (keeps the mat border)
CANVAS = GRID_N * CELL + 2 * MARGIN
ROUGH_SIZE = 4000

MIN_LINE_SUPPORT = 150   # samples needed to trust a fitted line


def line_responses(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (horizontal, vertical) line-response maps (float32)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    near_green = cv2.dilate(green, np.ones((25, 25), np.uint8))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((21, 21), np.uint8))
    tophat[near_green == 0] = 0

    rh = cv2.morphologyEx(tophat, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (151, 1)))
    rv = cv2.morphologyEx(tophat, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, 151)))
    return rh.astype(np.float32), rv.astype(np.float32)


def _lattice_search(profile: np.ndarray, s_range: Tuple[float, float],
                    o_range: Tuple[int, int]) -> Tuple[float, float]:
    """Find (offset, spacing) maximising profile support of 13 lines."""
    p = maximum_filter1d(gaussian_filter1d(profile, 4), 9)
    n = len(p)
    best = (-1.0, 0.0, 0.0)
    k = np.arange(GRID_N + 1)
    for s in np.arange(s_range[0], s_range[1], 0.5):
        o_max = min(o_range[1], int(n - 1 - GRID_N * s))
        if o_max <= o_range[0]:
            continue
        offs = np.arange(o_range[0], o_max)
        idx = np.round(offs[:, None] + k[None, :] * s).astype(int)
        scores = p[idx].sum(axis=1)
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (float(scores[i]), float(offs[i]), float(s))
    return best[1], best[2]


def _fit_lines(resp: np.ndarray, centres: np.ndarray, half_band: float,
               horizontal: bool) -> list:
    """
    Fit one line per expected centre.

    For horizontal lines each column contributes the sub-pixel centroid of
    the response peak inside the band; model y = a*x + b (a, b returned).
    Vertical lines are handled by transposing.
    Returns list of (a, b, support, samples) or None per line.
    """
    r = resp if horizontal else resp.T
    h, w = r.shape
    thr = max(10.0, np.percentile(r[r > 0], 50) * 0.5) if np.any(r > 0) else 10.0
    xs_all = np.arange(w)
    out = []
    for c in centres:
        lo, hi = int(max(0, c - half_band)), int(min(h, c + half_band + 1))
        if hi - lo < 5:
            out.append(None)
            continue
        band = r[lo:hi]
        am = band.argmax(axis=0)
        peak = band[am, xs_all]
        ok = peak > thr
        if ok.sum() < MIN_LINE_SUPPORT:
            out.append(None)
            continue
        xs = xs_all[ok]
        # sub-pixel centroid in +-4 px around the argmax
        ys = np.empty(len(xs), np.float64)
        for j, x in enumerate(xs):
            a0 = max(0, am[x] - 4)
            col = band[a0:am[x] + 5, x].astype(np.float64)
            ys[j] = lo + a0 + (col * np.arange(len(col))).sum() / col.sum()
        # robust LSQ: fit, reject, refit
        keep = np.ones(len(xs), bool)
        a = b = 0.0
        for tol in (12.0, 5.0, 3.0):
            if keep.sum() < MIN_LINE_SUPPORT:
                break
            a, b = np.polyfit(xs[keep], ys[keep], 1)
            keep = np.abs(ys - (a * xs + b)) < tol
        if keep.sum() < MIN_LINE_SUPPORT:
            out.append(None)
            continue
        out.append((a, b, int(keep.sum()), (xs[keep], ys[keep])))
    return out


def _intersections(h_lines, v_lines, dst_of):
    """Intersect fitted lines -> (src_pts, dst_pts)."""
    src, dst = [], []
    for i, hl in enumerate(h_lines):       # y = a x + b
        if hl is None:
            continue
        for j, vl in enumerate(v_lines):   # x = c y + d
            if vl is None:
                continue
            a, b = hl[0], hl[1]
            c, d = vl[0], vl[1]
            y = (a * d + b) / (1 - a * c)
            x = c * y + d
            src.append((x, y))
            dst.append(dst_of(j, i))
    return np.float32(src), np.float32(dst)


def _detect_in(canvas: np.ndarray, s_range, o_range, half_band):
    rh, rv = line_responses(canvas)
    oy, sy = _lattice_search(rh.sum(axis=1), s_range, o_range)
    ox, sx = _lattice_search(rv.sum(axis=0), s_range, o_range)
    k = np.arange(GRID_N + 1)
    h_lines = _fit_lines(rh, oy + k * sy, half_band(sy), horizontal=True)
    v_lines = _fit_lines(rv, ox + k * sx, half_band(sx), horizontal=False)
    return h_lines, v_lines


def _homography_step(canvas, h_lines, v_lines) -> Optional[np.ndarray]:
    """Homography from current canvas to ideal canvas."""
    n_h = sum(l is not None for l in h_lines)
    n_v = sum(l is not None for l in v_lines)
    if n_h < 3 or n_v < 3:
        return None
    src, dst = _intersections(
        h_lines, v_lines,
        lambda j, i: (MARGIN + j * CELL, MARGIN + i * CELL))
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return H


def residuals(h_lines, v_lines) -> Dict[str, float]:
    """Deviation of measured line samples from the ideal lattice (canvas px)."""
    devs = []
    for i, l in enumerate(h_lines):
        if l is not None:
            devs.append(l[3][1] - (MARGIN + i * CELL))
    for j, l in enumerate(v_lines):
        if l is not None:
            devs.append(l[3][1] - (MARGIN + j * CELL))
    if not devs:
        return {'rms': float('inf'), 'max': float('inf'), 'p95': float('inf'),
                'lines': 0}
    d = np.abs(np.concatenate(devs))
    return {'rms': float(np.sqrt((d ** 2).mean())), 'max': float(d.max()),
            'p95': float(np.percentile(d, 95)),
            'lines': sum(l is not None for l in list(h_lines) + list(v_lines))}


def grid_corners_homography(corners: np.ndarray) -> np.ndarray:
    """
    Homography raw image -> canvas from the 4 outer grid corners (TL, TR, BR,
    BL) in photo pixels, e.g. placed by hand in the GUI.
    """
    lo, hi = MARGIN, MARGIN + GRID_N * CELL
    return cv2.getPerspectiveTransform(
        np.float32(corners), np.float32([[lo, lo], [hi, lo], [hi, hi], [lo, hi]]))


def fit_grid(image: np.ndarray, verbose: bool = False,
             H_init: Optional[np.ndarray] = None) -> Optional[dict]:
    """
    Fit the mat grid in a raw photo.

    H_init: optional starting homography raw image -> canvas (see
    grid_corners_homography). When given, mat-corner detection is skipped and
    the fit is refined from it; it must be within ~1/3" of the true grid.

    Returns dict with
      H       : 3x3 homography raw image -> canvas (grid at MARGIN..MARGIN+3600)
      h_lines, v_lines : final line fits in canvas space
      stats   : residual statistics in canvas space
    or None if the grid could not be found.
    """
    # (spacing tolerance, offset tolerance, half band) per refinement pass
    passes = [(6, 40, 60), (6, 40, 60)]

    if H_init is not None:
        H = H_init.copy()
        passes.insert(0, (15, 110, 120))     # hand-placed: wider first pass
    else:
        corners = detect_mat_corners(image)
        if corners is None:
            if verbose:
                print("   ⚠️ Mat not detected")
            return None

        # Pass 0: rough warp from mat corners (mat edge -> canvas edge)
        R = ROUGH_SIZE
        M0 = cv2.getPerspectiveTransform(
            corners, np.float32([[0, 0], [R, 0], [R, R], [0, R]]))
        rough = cv2.warpPerspective(image, M0, (R, R), flags=cv2.INTER_LINEAR)

        # Grid is ~12/13 of the mat, so spacing ~ R/13; allow wide range
        h_lines, v_lines = _detect_in(rough, (R / 15.5, R / 12.2), (0, R // 6),
                                      half_band=lambda s: 0.35 * s)
        H1 = _homography_step(rough, h_lines, v_lines)
        if H1 is None:
            if verbose:
                print("   ⚠️ Not enough grid lines in rough warp")
            return None
        H = H1 @ M0

    # Refine in the ideal canvas (lines now near-axis-aligned)
    for it, (ds, do, band) in enumerate(passes):
        canvas = cv2.warpPerspective(image, H, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
        h_lines, v_lines = _detect_in(canvas, (CELL - ds, CELL + ds),
                                      (MARGIN - do, MARGIN + do),
                                      half_band=lambda s, b=band: b)
        if verbose:
            st = residuals(h_lines, v_lines)
            print(f"   Pass {it + 1}: {st['lines']}/26 lines, "
                  f"residual rms={st['rms']:.1f}px p95={st['p95']:.1f}px max={st['max']:.1f}px")
        Hn = _homography_step(canvas, h_lines, v_lines)
        if Hn is None:
            break
        H = Hn @ H

    canvas = cv2.warpPerspective(image, H, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
    h_lines, v_lines = _detect_in(canvas, (CELL - 3, CELL + 3),
                                  (MARGIN - 15, MARGIN + 15),
                                  half_band=lambda s: 40)
    stats = residuals(h_lines, v_lines)
    if verbose:
        print(f"   Final: {stats['lines']}/26 lines, residual rms={stats['rms']:.1f}px "
              f"p95={stats['p95']:.1f}px max={stats['max']:.1f}px")
    return {'H': H, 'canvas': canvas, 'h_lines': h_lines, 'v_lines': v_lines,
            'stats': stats}


# ---------------------------------------------------------------------------
# Phase 2: smooth residual distortion (lens barrel/pincushion, mat bending)
# ---------------------------------------------------------------------------

def _poly_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Bivariate polynomial design matrix in normalised canvas coords."""
    u = (x - CANVAS / 2) / (CANVAS / 2)
    v = (y - CANVAS / 2) / (CANVAS / 2)
    return np.stack([u ** i * v ** j
                     for i in range(degree + 1)
                     for j in range(degree + 1 - i)], axis=-1)


def _fit_field(xs, ys, d, degree):
    """Robust LSQ fit of displacement d at (xs, ys)."""
    A = _poly_terms(xs, ys, degree)
    keep = np.ones(len(d), bool)
    coef = np.zeros(A.shape[1])
    for tol in (8.0, 3.0, 2.0):
        coef, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        keep = np.abs(A @ coef - d) < tol
    return coef, float(np.sqrt(((A[keep] @ coef - d[keep]) ** 2).mean()))


def fit_distortion(h_lines, v_lines, degree: int = 4) -> dict:
    """
    Fit dy(x, y) from horizontal-line samples and dx(x, y) from vertical-line
    samples (measured - ideal, canvas px). A low-order 2-D polynomial is used
    so occluded areas are filled smoothly from the visible mat.
    """
    hx, hy, hd, vx, vy, vd = [], [], [], [], [], []
    for i, l in enumerate(h_lines):
        if l is not None:
            x, y = l[3]
            hx.append(x[::4]); hy.append(y[::4]); hd.append(y[::4] - (MARGIN + i * CELL))
    for j, l in enumerate(v_lines):
        if l is not None:
            y, x = l[3]          # transposed fit: samples are (y, x)
            vx.append(x[::4]); vy.append(y[::4]); vd.append(x[::4] - (MARGIN + j * CELL))
    cy, rms_y = _fit_field(np.concatenate(hx), np.concatenate(hy), np.concatenate(hd), degree)
    cx, rms_x = _fit_field(np.concatenate(vx), np.concatenate(vy), np.concatenate(vd), degree)
    return {'cx': cx, 'cy': cy, 'degree': degree, 'fit_rms': max(rms_x, rms_y)}


def render(image: np.ndarray, H: np.ndarray, distortion: Optional[dict] = None,
           size: int = GRID_N * CELL) -> np.ndarray:
    """
    Render the 12x12" grid area from the raw photo with ONE resample:
    output px -> canvas (+ residual displacement) -> raw photo via H^-1.
    """
    v, u = np.mgrid[0:size, 0:size].astype(np.float32)
    cxs = u + MARGIN
    cys = v + MARGIN
    if distortion is not None:
        A = _poly_terms(cxs.ravel(), cys.ravel(), distortion['degree']).astype(np.float32)
        dx = (A @ distortion['cx'].astype(np.float32)).reshape(size, size)
        dy = (A @ distortion['cy'].astype(np.float32)).reshape(size, size)
        del A
        cxs = cxs + dx
        cys = cys + dy
    Hi = np.linalg.inv(H).astype(np.float32)
    w = Hi[2, 0] * cxs + Hi[2, 1] * cys + Hi[2, 2]
    map_x = (Hi[0, 0] * cxs + Hi[0, 1] * cys + Hi[0, 2]) / w
    map_y = (Hi[1, 0] * cxs + Hi[1, 1] * cys + Hi[1, 2]) / w
    return cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))


def canvas_to_photo(pts: np.ndarray, H: np.ndarray,
                    distortion: Optional[dict] = None) -> np.ndarray:
    """
    Map ideal canvas points (N x 2) to raw photo pixels exactly as render()
    samples them, i.e. where the model believes those grid points are.
    """
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    if distortion is not None:
        A = _poly_terms(x, y, distortion['degree'])
        x, y = x + A @ distortion['cx'], y + A @ distortion['cy']
    q = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(q, np.linalg.inv(H)).reshape(-1, 2)


def measure_output(output: np.ndarray) -> Dict[str, float]:
    """
    Independent QA: re-detect grid lines in a 3600x3600 output and report how
    far they are from the ideal k*300 px positions.
    """
    padded = cv2.copyMakeBorder(output, MARGIN, MARGIN, MARGIN, MARGIN,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    h_lines, v_lines = _detect_in(padded, (CELL - 1, CELL + 1),
                                  (MARGIN - 3, MARGIN + 3),
                                  half_band=lambda s: 25)
    return residuals(h_lines, v_lines)
