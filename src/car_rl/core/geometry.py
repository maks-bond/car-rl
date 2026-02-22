from __future__ import annotations

import math
from typing import Optional


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def segment_intersects(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    r = _sub(a2, a1)
    s = _sub(b2, b1)
    denom = _cross(r, s)
    q_minus_p = _sub(b1, a1)

    if abs(denom) < 1e-12:
        return False

    t = _cross(q_minus_p, s) / denom
    u = _cross(q_minus_p, r) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def point_to_segment_distance(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ab = _sub(b, a)
    ap = _sub(p, a)
    ab_norm_sq = dot(ab, ab)
    if ab_norm_sq < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])

    t = dot(ap, ab) / ab_norm_sq
    t = max(0.0, min(1.0, t))
    closest = (a[0] + t * ab[0], a[1] + t * ab[1])
    return math.hypot(p[0] - closest[0], p[1] - closest[1])


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return _cross(_sub(b, a), _sub(c, a))


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    eps = 1e-12
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def point_in_convex_polygon(p: tuple[float, float], poly: tuple[tuple[float, float], ...]) -> bool:
    # Works for CW or CCW convex polygons; points on edge are considered inside.
    if len(poly) < 3:
        return False

    sign = 0
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        o = _orientation(a, b, p)
        if abs(o) < 1e-12 and _on_segment(a, b, p):
            return True
        if o > 0:
            if sign < 0:
                return False
            sign = 1
        elif o < 0:
            if sign > 0:
                return False
            sign = -1
    return True


def ray_segment_intersection_distance(
    origin: tuple[float, float],
    direction: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> Optional[float]:
    # Ray: origin + t * direction (t >= 0), direction should be unit-length.
    # Segment: seg_a + u * (seg_b - seg_a), 0 <= u <= 1.
    seg = _sub(seg_b, seg_a)
    denom = _cross(direction, seg)
    if abs(denom) < 1e-12:
        return None

    rel = _sub(seg_a, origin)
    t = _cross(rel, seg) / denom
    u = _cross(rel, direction) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None
