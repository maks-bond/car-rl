from __future__ import annotations

import math


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
