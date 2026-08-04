#!/usr/bin/env python3
"""One continuous NQ tape, stitched from every cached contract month."""
from __future__ import annotations

import argparse

import numpy as np

import tape


ALL = "ALL"
ROOT = "NQ"                     # ALL stitches this instrument only

# A real NQ quarterly roll spread. Anything a stitch calls a "roll" has to land
# in this band or it is an outlier that happened to look like one. Measured on
# this data: +241.75, +250.00, +214.50, +281.75.
ROLL_BAND = (150.0, 350.0)

# Largest tolerable move between the close of one weekday session and the open
# of the next, in the adjusted series. Over a 24-hour tape those sessions are
# separated only by the 17:00-18:00 maintenance break: measured median 8 points,
# 90th percentile 33, maximum 134.5, none above 150. A session assigned to the
# wrong contract sits a roll spread out of line and trips this immediately.
SESSION_GAP_LIMIT = 150.0

# Seconds both contracts must print in for a roll spread to be measurable. The
# thinnest real roll on this data had 10,821.
MIN_OVERLAP_SECONDS = 200


def month_key(contract: str) -> int:
    """'NQ 12-25' -> 2512, so contract months sort in calendar order."""
    mm, yy = contract.split()[1].split("-")
    return int(yy) * 100 + int(mm)


def front_months(daily: dict) -> dict:
    """Front month per session: most volume, and the switch NEVER REVERSES.

    `daily` maps contract name to {day_index: volume}. Returns
    {day_index: contract}.

    The monotone clause is the whole function. Volume genuinely oscillates
    during roll week -- the deferred contract takes the lead, gives it back the
    next day, takes it again -- and a picker that answers independently each day
    follows every one of those swings. Doing that across the September 2025 roll
    put a 240-point round trip into the middle of the series that the market
    never traded. Once we roll, we never roll back.
    """
    days = sorted({d for s in daily.values() for d in s})
    chosen, cur = {}, None
    for d in days:
        pool = {c: s[d] for c, s in daily.items() if d in s}
        if not pool:
            continue
        if cur is not None:
            forward = {c: v for c, v in pool.items()
                       if month_key(c) >= month_key(cur)}
            pool = forward or pool
        cur = max(pool, key=pool.get)
        chosen[d] = cur
    return chosen


def second_last(ts: np.ndarray, px: np.ndarray, day: int):
    """Seconds of one session, and the last price printed in each.

    Bucketed to whole seconds because two contracts never share an exact tick
    timestamp -- they print independently, microseconds apart.
    """
    lo = (np.int64(day) * 86400 + tape.NET_EPOCH_S) * tape.TPS
    hi = lo + np.int64(86400) * tape.TPS
    i0, i1 = np.searchsorted(ts, (lo, hi))
    if i1 <= i0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    s = ts[i0:i1] // tape.TPS
    last = np.r_[s[1:] != s[:-1], True]          # last tick of each second
    return s[last], px[i0:i1][last]


def measure_roll(old_ts, old_px, new_ts, new_px, day):
    """The roll spread, measured. Returns (spread, common_seconds).

    Both contracts are in the tape for the same seconds around the roll, so the
    spread is read off rather than assumed: the median per-second difference on
    the day they change hands. Measured this way the four rolls on this data
    come out at +241.75, +250.00, +214.50 and +281.75 with an interquartile
    range near 2 points, so a guess would have been wrong by tens of points.

    Returns (None, n) when the overlap is too thin to be a measurement.
    """
    sa, pa = second_last(old_ts, old_px, day)
    sb, pb = second_last(new_ts, new_px, day)
    common = np.intersect1d(sa, sb, assume_unique=True)
    if len(common) < MIN_OVERLAP_SECONDS:
        return None, len(common)
    diff = pb[np.searchsorted(sb, common)] - pa[np.searchsorted(sa, common)]
    return float(np.median(diff)), len(common)


def selfcheck():
    # Contract months in calendar order, with the roll week in the middle.
    # Volume oscillates across the roll -- 12-25 wins day 3, loses day 4, wins
    # day 5 -- which is what really happens and what a naive picker gets wrong.
    daily = {
        "NQ 09-25": {1: 900, 2: 800, 3: 400, 4: 500, 5: 100},
        "NQ 12-25": {3: 600, 4: 300, 5: 900, 6: 900, 7: 900},
    }
    f = front_months(daily)
    assert f[1] == "NQ 09-25", f
    assert f[2] == "NQ 09-25", f
    assert f[3] == "NQ 12-25", f
    # THE POINT OF THE WHOLE FUNCTION. Day 4 has more 09-25 volume than 12-25,
    # and a day-by-day majority picker rolls back to it. That flip-flop wrote a
    # 240-point round trip into the middle of the series that never happened.
    assert f[4] == "NQ 12-25", "roll must be monotone: never step back"
    assert f[5] == "NQ 12-25", f
    assert f[7] == "NQ 12-25", f

    # A contract absent from a day cannot be chosen for it.
    assert set(front_months({"NQ 09-25": {1: 5}})) == {1}

    assert month_key("NQ 09-25") == 2509
    assert month_key("NQ 03-26") == 2603
    assert month_key("NQ 12-25") < month_key("NQ 03-26"), "years must dominate"

    # Two contracts printing in the same seconds on the same day, the new one a
    # known 241.75 above the old. Ticks land at irregular offsets inside each
    # second, which is what the real tape looks like -- two contracts never
    # share an exact tick timestamp, so the match has to be per second.
    base = (np.int64(20000) * 86400 + tape.NET_EPOCH_S) * tape.TPS      # some day, in .NET ticks
    day = int(np.int64(20000))
    secs = np.arange(1000, dtype=np.int64)
    old_ts = base + secs * tape.TPS + np.int64(3_000_000)
    new_ts = base + secs * tape.TPS + np.int64(7_000_000)
    old_px = (23000 + np.sin(secs / 50.0) * 20).astype(np.float32)
    new_px = (old_px + 241.75).astype(np.float32)

    sp, n = measure_roll(old_ts, old_px, new_ts, new_px, day)
    assert n == 1000, n
    assert abs(sp - 241.75) < 0.01, sp

    # Too little overlap is not a measurement.
    sp2, n2 = measure_roll(old_ts[:50], old_px[:50], new_ts[:50], new_px[:50], day)
    assert sp2 is None and n2 == 50, (sp2, n2)

    # A different day shares no seconds at all.
    sp3, n3 = measure_roll(old_ts, old_px, new_ts, new_px, day + 1)
    assert sp3 is None and n3 == 0, (sp3, n3)

    print("selfcheck OK: front months monotone across an oscillating roll")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    ap.print_help()


if __name__ == "__main__":
    main()
