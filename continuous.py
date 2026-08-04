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


def audit(ts: np.ndarray, px: np.ndarray, rolls: list) -> list:
    """Everything that must be true of a stitched tape. [] means sound.

    A mis-stitched tape does not raise. It produces a plausible, wrong
    backtest, and every number the tool exists to get right is computed on top
    of it. So the build refuses to write when any of these fails, and last
    week's tape stays in place.
    """
    bad = []
    if len(ts) < 2:
        return ["tape is empty"]
    if not (np.diff(ts) >= 0).all():
        bad.append("timestamps are not sorted")

    day = tape.day_index(ts)
    lo, hi = ROLL_BAND

    # A roll-sized step between two consecutive ticks of the SAME session. No
    # market moves 150 points between prints inside a session; a contract change
    # does.
    step = np.abs(np.diff(px.astype(np.float64)))
    same = np.diff(day) == 0
    n_mid = int((same & (step >= lo) & (step <= hi)).sum())
    if n_mid:
        bad.append(f"{n_mid} roll-sized step(s) inside a session")

    # The strong one: a whole session on the wrong contract. Consecutive
    # sessions are separated only by the maintenance break, so the adjusted
    # price barely moves across it -- measured median 8 points, 90th percentile
    # 33, maximum 134.5. A session a roll spread out of line shows up here even
    # when the step onto it hid inside an overnight gap, which is exactly how
    # the morning of 2025-03-18 stayed on the wrong contract undetected during
    # the NQData build.
    edges = np.flatnonzero(np.r_[True, np.diff(day) != 0])
    ends = np.r_[edges[1:] - 1, len(ts) - 1]
    gaps = np.abs(px[edges[1:]].astype(np.float64) - px[ends[:-1]])
    over = np.flatnonzero(gaps > SESSION_GAP_LIMIT)
    if len(over):
        first = tape.date_str(int(day[edges[1:][over[0]]]))
        bad.append(f"{len(over)} session gap(s) over {SESSION_GAP_LIMIT:g} pts, "
                   f"first into {first} ({gaps[over[0]]:.1f} pts)")

    for r in rolls:
        if r["spread"] is None:
            bad.append(f"roll on day {r['day']} has no measured spread")
        elif not lo <= r["spread"] <= hi:
            bad.append(f"roll {r['from']}->{r['to']} spread {r['spread']:.2f} "
                       f"is outside {lo:g}-{hi:g}")
    return bad


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

    # Add outliers to verify median robustness over mean. Real tape carries bad
    # prints (trades at stale edges, HFT reprints), so median resists outliers
    # that would shift a naive mean by several points. With 10 prints ~500 points
    # above the spread, median stays 241.75 but mean would be ~246.7.
    outlier_indices = np.array([50, 150, 250, 350, 450, 550, 650, 750, 850, 950], dtype=np.int64)
    new_px[outlier_indices] += 500.0

    sp, n = measure_roll(old_ts, old_px, new_ts, new_px, day)
    assert n == 1000, n
    assert abs(sp - 241.75) < 0.01, sp

    # Too little overlap is not a measurement.
    sp2, n2 = measure_roll(old_ts[:50], old_px[:50], new_ts[:50], new_px[:50], day)
    assert sp2 is None and n2 == 50, (sp2, n2)

    # A different day shares no seconds at all.
    sp3, n3 = measure_roll(old_ts, old_px, new_ts, new_px, day + 1)
    assert sp3 is None and n3 == 0, (sp3, n3)

    # A sound two-session tape: one session per day, a small overnight move.
    def _tape(sess_closes):
        ts, px = [], []
        for i, close in enumerate(sess_closes):
            d = 20000 + i
            s0 = (np.int64(d) * 86400 + tape.NET_EPOCH_S) * tape.TPS
            n = 400
            ts.append(s0 + (36000 + np.arange(n, dtype=np.int64)) * tape.TPS)
            px.append(np.full(n, close, np.float32))
        return np.concatenate(ts), np.concatenate(px)

    good_ts, good_px = _tape([23000.0, 23010.0, 23005.0])
    ok_rolls = [dict(day=20001, **{"from": "NQ 09-25", "to": "NQ 12-25"},
                     spread=241.75, n=26104)]
    assert audit(good_ts, good_px, ok_rolls) == [], audit(good_ts, good_px, ok_rolls)

    # A whole session sitting one roll spread out of line. This is the failure
    # that matters: it does not raise, it just quietly produces a wrong
    # backtest, and it is invisible to a check that only looks a few ticks ahead
    # because the step onto it hides in an overnight gap.
    bad_ts, bad_px = _tape([23000.0, 23010.0 - 241.75, 23005.0])
    fails = audit(bad_ts, bad_px, ok_rolls)
    assert any("session" in f for f in fails), fails

    # Out-of-band roll spread.
    assert audit(good_ts, good_px,
                 [dict(day=20001, **{"from": "a", "to": "b"},
                       spread=12.0, n=999)]), "12 points is not a roll"

    # Unsorted time.
    assert audit(good_ts[::-1], good_px, ok_rolls), "must reject unsorted ts"

    # A roll-sized step INSIDE one session.
    mid_ts, mid_px = _tape([23000.0])
    mid_px[200:] += 241.75
    assert audit(mid_ts, mid_px, ok_rolls), "must reject an intraday roll step"

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
