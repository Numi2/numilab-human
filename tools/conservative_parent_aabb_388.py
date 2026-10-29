"""Directed binary64 lower bounds; only excludes provably distant parent AABBs."""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np


def parent_aabb_candidate_mask(point, bounds_min, bounds_max, exact_upper_squared):
    point=np.asarray(point,dtype=np.float64)
    low=np.asarray(bounds_min,dtype=np.float64)
    high=np.asarray(bounds_max,dtype=np.float64)
    if point.shape!=(3,) or low.ndim!=2 or low.shape[1]!=3 or high.shape!=low.shape:
        raise ValueError('parent AABB filter dimensions are invalid')
    if not np.isfinite(point).all() or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high):
        raise ValueError('parent AABB filter requires finite ordered bounds')
    if not isinstance(exact_upper_squared,Fraction) or exact_upper_squared<0:
        raise ValueError('parent AABB filter requires a nonnegative exact rational upper bound')
    # Each elementary operation is rounded down by one representable step.
    # max/min only select existing bounds. Clamping a negative lower bound to
    # zero is safe because all exact distances/products/sums are nonnegative.
    with np.errstate(over='ignore',under='ignore',invalid='raise'):
        left=np.nextafter(low-point,-np.inf)
        right=np.nextafter(point-high,-np.inf)
        gap=np.maximum(np.maximum(left,right),0.)
        square=np.maximum(np.nextafter(gap*gap,-np.inf),0.)
        lower=np.zeros(len(low),dtype=np.float64)
        for axis in range(3):
            lower=np.maximum(np.nextafter(lower+square[:,axis],-np.inf),0.)
    try:
        rounded_upper=float(exact_upper_squared)
    except OverflowError:
        rounded_upper=math.inf
    upper=float(np.nextafter(rounded_upper,math.inf))
    return lower<=upper
