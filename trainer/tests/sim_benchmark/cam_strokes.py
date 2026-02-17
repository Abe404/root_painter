"""CNC-inspired stroke generation using contour-parallel pocketing.

Alternative to the hand-rolled raster zigzag in sim_user.py.
Uses Shapely polygon offsets to generate concentric contour paths,
similar to how CNC pocket milling clears material with a round end mill.

Approach:
  1. Convert binary error mask → polygon(s) via skimage find_contours
  2. Build the paint region: GT class mask eroded by brush_radius,
     intersected with dilated error neighbourhood
  3. Repeatedly shrink polygon inward by ~85% brush diameter
  4. Each contour ring becomes a stroke path

If the brush doesn't fit (thin regions), try progressively smaller
brushes down to radius 2.  This mirrors the CNC practice of roughing
with a large end mill then finishing with a smaller one.

Dependencies: shapely, scikit-image.
"""
import numpy as np
from skimage import measure
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import disk
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely import buffer as shapely_buffer
from shapely.validation import make_valid


def _mask_to_polygons(mask, min_area=10):
    """Convert a binary mask to a list of Shapely Polygons."""
    contours = measure.find_contours(mask.astype(float), 0.5)
    polys = []
    for c in contours:
        if len(c) < 6:
            continue
        poly = Polygon(c)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area < min_area:
            continue
        if isinstance(poly, MultiPolygon):
            for g in poly.geoms:
                if g.area >= min_area:
                    polys.append(g)
        elif hasattr(poly, 'geoms'):
            # GeometryCollection from make_valid
            for g in poly.geoms:
                if hasattr(g, 'area') and g.area >= min_area:
                    polys.append(g)
        else:
            polys.append(poly)
    return polys


def _sample_contour(coords, spacing):
    """Resample polygon coordinates at regular spacing."""
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum_dist[-1]
    if total < spacing:
        return coords
    n_samples = max(3, int(total / spacing))
    sample_dists = np.linspace(0, total, n_samples)
    sampled = np.array([
        np.interp(sample_dists, cum_dist, coords[:, dim])
        for dim in range(2)
    ]).T
    return sampled


def _extract_geoms(geometry):
    """Yield individual polygons from any Shapely geometry."""
    if geometry.is_empty:
        return
    if isinstance(geometry, MultiPolygon):
        for g in geometry.geoms:
            yield g
    elif hasattr(geometry, 'geoms'):
        for g in geometry.geoms:
            if hasattr(g, 'exterior'):
                yield g
    elif hasattr(geometry, 'exterior'):
        yield geometry


def generate_paths(error_mask, gt_class_mask, brush_radius,
                   overlap=0.85, min_area=10):
    """Generate stroke paths to cover an error region.

    Uses contour-parallel pocketing with automatic fallback to smaller
    brushes for thin regions (like CNC roughing then finishing).

    Parameters
    ----------
    error_mask : 2D bool array
        Error pixels to correct.
    gt_class_mask : 2D bool array
        Full GT mask for the correct class.
    brush_radius : int
        Starting brush radius in pixels.

    Returns
    -------
    list of (paths, radius) tuples.
    Each paths entry is a list of (N, 2) arrays (row, col stroke paths).
    """
    h, w = error_mask.shape
    results = []
    remaining = error_mask.copy()

    # Try progressively smaller brushes
    radii = [brush_radius]
    r = brush_radius
    while r > 2:
        r = max(2, r // 2)
        radii.append(r)

    for br in radii:
        if not np.any(remaining):
            break

        # Build paint region for this brush size
        error_nbhd = binary_dilation(remaining, structure=disk(br))
        # Erode GT by brush_radius+2 so the full disk stays inside.
        # The +2 accounts for subpixel contour positions and rounding.
        erosion = br + 2
        if erosion > 1:
            gt_eroded = binary_erosion(gt_class_mask, structure=disk(erosion),
                                       border_value=1)
        else:
            gt_eroded = gt_class_mask
        paint_region = gt_eroded & error_nbhd
        if not np.any(paint_region):
            continue

        polys = _mask_to_polygons(paint_region, min_area=max(3, min_area))
        if not polys:
            continue

        step = max(1, int(br * 2 * overlap))
        paths = []

        for poly in polys:
            current = poly
            iteration = 0
            while not current.is_empty and iteration < 50:
                for geom in _extract_geoms(current):
                    if geom.area < 2:
                        continue
                    coords = np.array(geom.exterior.coords)
                    sampled = _sample_contour(coords, spacing=max(2, br))
                    if len(sampled) >= 2:
                        paths.append(sampled)
                current = shapely_buffer(current, -step)
                iteration += 1

        if paths:
            results.append((paths, br))
            # Update remaining: simulate painting to see what's covered
            covered = _simulate_paint(paths, br, h, w)
            remaining = remaining & ~covered

    return results


def _simulate_paint(paths, brush_radius, h, w):
    """Simulate painting to compute covered pixels."""
    canvas = np.zeros((h, w), dtype=bool)
    br = brush_radius
    for path in paths:
        for pt in path:
            r, c = int(round(pt[0])), int(round(pt[1]))
            r0, r1 = max(0, r - br), min(h, r + br + 1)
            c0, c1 = max(0, c - br), min(w, c + br + 1)
            rr = np.arange(r0, r1)[:, None]
            cc = np.arange(c0, c1)[None, :]
            mask = ((rr - r)**2 + (cc - c)**2) <= br**2
            canvas[r0:r1, c0:c1] |= mask
    return canvas


if __name__ == '__main__':
    """Compare strategies on synthetic error shapes."""
    size = 200
    y, x = np.ogrid[:size, :size]

    # GT: ellipse
    gt = (((x - 100) / 50)**2 + ((y - 100) / 30)**2 <= 1)

    test_cases = {
        'crescent (FN)': {
            'pred': (((x - 110) / 45)**2 + ((y - 95) / 35)**2 <= 1),
            'error_type': 'fn',
        },
        'blob (FP)': {
            'pred': gt | (((x - 60) / 15)**2 + ((y - 60) / 15)**2 <= 1),
            'error_type': 'fp',
        },
        'thin strip (FN)': {
            # Prediction misses a thin strip at the bottom of the ellipse
            'pred': gt & ~((y > 120) & (y < 130) & gt),
            'error_type': 'fn',
        },
    }

    for name, tc in test_cases.items():
        pred = tc['pred']
        if tc['error_type'] == 'fn':
            error = gt & ~pred
            gt_class = gt
        else:
            error = ~gt & pred
            gt_class = ~gt

        error_area = int(np.sum(error))
        if error_area == 0:
            print(f"\n{name}: no error pixels, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"{name}: {error_area} error pixels")
        print(f"{'='*50}")

        for br in [4, 6, 8]:
            results = generate_paths(error, gt_class, br)
            # Count coverage
            total_covered = np.zeros((size, size), dtype=bool)
            total_strokes = 0
            total_spill = 0
            for paths, radius in results:
                painted = _simulate_paint(paths, radius, size, size)
                total_covered |= painted
                total_strokes += len(paths)
                total_spill += int(np.sum(painted & ~gt_class))

            covered = int(np.sum(total_covered & error))
            pct = 100 * covered / error_area
            print(f"  br={br}: {total_strokes} strokes, "
                  f"coverage={covered}/{error_area} ({pct:.1f}%), "
                  f"spill={total_spill}px, "
                  f"radii={[r for _, r in results]}")
