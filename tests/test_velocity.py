"""Guard tests for velocity windows.

These exist to catch the one failure that matters: a window that reaches
forward in time. A centred or forward-looking window would inflate every
downstream metric and nothing else in the pipeline would notice.
"""

import numpy as np

from kavach.features.velocity import _rolling_count_sum


def _brute(codes, times, vals, w, i):
    m = (codes == codes[i]) & (times > times[i] - w) & (times <= times[i])
    return m.sum(), vals[m].sum()


def test_matches_brute_force_on_unique_timestamps():
    rng = np.random.default_rng(0)
    n = 2000
    times = np.sort(rng.choice(5_000_000, n, replace=False)).astype(float)
    codes = rng.integers(0, 7, n).astype(float)
    vals = rng.random(n) * 100
    c, s = _rolling_count_sum(codes, times, vals, 86_400)
    for i in rng.choice(n, 200, replace=False):
        bc, bs = _brute(codes, times, vals, 86_400, i)
        assert int(c[i]) == bc
        assert abs(s[i] - bs) < 1e-3


def test_window_never_looks_forward():
    """A row placed far in the future must not affect an earlier row."""
    times = np.array([0.0, 100.0, 200.0, 10_000_000.0])
    codes = np.zeros(4)
    vals = np.ones(4)
    c, _ = _rolling_count_sum(codes, times, vals, 86_400)
    assert c[0] == 1 and c[1] == 2 and c[2] == 3
    assert c[3] == 1  # sees none of the old ones


def test_groups_do_not_bleed_into_each_other():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    codes = np.array([0.0, 1.0, 0.0, 1.0])
    vals = np.ones(4)
    c, _ = _rolling_count_sum(codes, times, vals, 86_400)
    assert list(c) == [1, 1, 2, 2]
