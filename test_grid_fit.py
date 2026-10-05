"""
Regression test: every test photo must yield a 3600x3600 output whose
re-detected grid lines sit within a few px of k*300.

Run: python -m pytest test_grid_fit.py -v   (or: python test_grid_fit.py)
"""

import glob
import os

import cv2
import pytest

from grid_fit import fit_grid, fit_distortion, render, measure_output, MARGIN, GRID_N, CELL
from orientation import detect_notch_side

TEST_IMAGES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'Test', '*.jpeg')))


@pytest.mark.skipif(not TEST_IMAGES, reason='no images in Test/')
@pytest.mark.parametrize('path', TEST_IMAGES, ids=os.path.basename)
def test_grid_alignment(path):
    image = cv2.imread(path)
    fit = fit_grid(image)
    assert fit is not None, 'grid not found'
    assert fit['stats']['lines'] == 26

    out = render(image, fit['H'], fit_distortion(fit['h_lines'], fit['v_lines']))
    assert out.shape[:2] == (3600, 3600)

    qa = measure_output(out)
    assert qa['lines'] >= 24
    assert qa['rms'] < 2.5, qa       # < 0.01"
    assert qa['p95'] < 4.0, qa

    assert detect_notch_side(fit['canvas'], MARGIN, MARGIN + GRID_N * CELL) is not None


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
