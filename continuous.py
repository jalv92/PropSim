#!/usr/bin/env python3
"""One continuous NQ tape, stitched from every cached contract month."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import tape


ALL = "ALL"
ROOT = "NQ"                     # ALL stitches this instrument only

# A real NQ quarterly roll spread. Anything a stitch calls a "roll" has to land
# in this band or it is an outlier that happened to look like one. Measured on
# this data: +237.25, +256.00, +210.25, +281.75.
ROLL_BAND = (150.0, 350.0)

# Largest tolerable move between the close of one weekday session and the open
# of the next weekday session, in the adjusted series (weekends and holidays
# are excluded -- see audit()). This boundary sits at the REAL 17:00-18:00 ET
# maintenance halt now that `day` is session_day, not calendar day -- and the
# halt moves a lot more than a mid-session midnight crossing used to.
# Measured on the real 107.7M-tick stitched tape, 174 consecutive-weekday
# boundaries: median 5.50, p90 24.55, p99 84.64, max 133.75. The smallest real
# roll spread is 210.25 -- a wrong-contract session offset by that spread must
# still clear this gate. The usable band is exactly (133.75, 210.25): a
# 1.57x span, not a comfortable one on either side. 175.0 sits roughly in the
# middle -- 31% above the worst halt move seen so far, 17% below the smallest
# defect this exists to catch. Both margins are THIN. A future halt move
# bigger than 133.75 (this is one tape's max, not a proven ceiling) can still
# trip this and block a rebuild; a wrong-contract session offset by exactly
# 210.25 could still hide if a halt move happens to cancel ~35 points of it.
# This check is not the primary defense against that defect any more, though:
# build() assigns contracts by CALENDAR day (front_months/mine), so a whole
# wrong-contract session lands as a step at midnight -- which now sits INSIDE
# a session_day key, not at a session-gap boundary -- and the intraday-step
# check catches it there. This check's remaining job is narrower: flag a
# session-to-session move too big to be a real halt, which is still worth
# keeping as a second line of defense, just not the one to lean on.
SESSION_GAP_LIMIT = 175.0

# Seconds both contracts must print in for a roll spread to be measurable.
# The thinnest real roll on this data had 1,567 -- 7.8x headroom over this
# threshold, not the 54x a smaller-tape measurement once suggested. Worth
# saying plainly: that is a real cushion, but a much smaller one than it looks
# if this constant is read in isolation.
MIN_OVERLAP_SECONDS = 200

# NQ's session runs roughly 18:00 ET one evening to 17:00 ET the next day --
# only the maintenance halt (17:00-18:00 ET) sits in between.
SESSION_OPEN_H = 18

# A calendar-day gap this big between two sessions that ARE both present in
# the tape is no longer a weekend. Measured on the real tape's 273 gaps
# between consecutive sessions: 231 of 1 day (weekday to weekday), 40 of 2
# (Friday to Sunday), zero of 3 (a holiday Monday/Friday would show as this --
# none happened to land in this window, but a real one must not be reported
# as a hole). Anything past that -- the real tape has one 9-day gap and one
# 47-day gap -- is missing data, not a market closure, and worth telling
# someone about before they backtest across it and wonder where their trades
# went.
MAX_NORMAL_GAP_DAYS = 3


def session_day(ts: np.ndarray) -> np.ndarray:
    """A trading-session key for audit() only: a session belongs to the day it
    CLOSES on, not the calendar day printed on each tick.

    tape.day_index buckets at midnight. Most of the time that's harmless --
    the daily step across the maintenance halt is a few points (max observed
    85.50 across the full 107.7M-tick tape, see the step check below). But the
    same midnight bucketing also puts a stale pre-open filler tick (NT8 emits a
    flat, 1-lot "last price" print every minute or so through the halt) and
    the real reopen print in the SAME calendar-day bucket whenever the halt's
    other side is a whole weekend rather than an hour -- every Sunday reopen.
    That put 3 real weekend gaps (up to 354 points, every one landing at
    exactly 18:00:00 ET) inside single day_index buckets on the first
    5-contract build of ALL, tripping the intraday-step check on real, correct
    prices. Measured over all five NQ caches (109M ticks): 3 offenders under
    plain day_index, 0 under this session-anchored key -- and it isn't
    weakening the check, it's asking it the right question. A future reader
    tempted to "simplify" this back to day_index(ts) would reintroduce all 3.

    Deliberately local to audit(): tape.day_index itself stays calendar-day,
    because the engine also uses it for the daily governor and for flattening
    at session close, where "day" has to mean calendar day.

    Shifting ts forward by (24 - SESSION_OPEN_H) hours moves the day_index
    floor boundary from midnight to SESSION_OPEN_H:00 ET, so an 18:00 tick
    lands in the next calendar day's bucket -- joining the overnight session
    it actually opens instead of the one it happened to print on.
    """
    return tape.day_index(ts + (24 - SESSION_OPEN_H) * 3600 * tape.TPS)


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
    come out at +237.25, +256.00, +210.25 and +281.75 with an interquartile
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

    day = session_day(ts)
    lo, hi = ROLL_BAND

    # An intraday step of 150 points or more is either a contract change or
    # corruption, and both must stop the write. No market moves 150 points
    # between prints inside a session. Real data: max observed 85.50 over the
    # full 107.7M-tick tape, zero above 150. So an unbounded check costs
    # nothing in false positives and catches everything suspicious. This is
    # also the check that actually catches a wrong-contract session now
    # (see SESSION_GAP_LIMIT's comment): build() assigns contracts by
    # calendar day, so a whole session on the wrong contract steps at
    # midnight -- inside a session_day key, not at a session-gap boundary.
    #
    # No float64 here: px is already exact float32 (see build()'s comment on
    # why), and this was measured as the single largest allocation in the
    # real 107.7M-tick build -- diff+abs on a float64 up-cast of the whole
    # tape cost over 1 GB more than doing the identical, still-exact
    # arithmetic directly in float32.
    step = np.abs(np.diff(px))
    same = np.diff(day) == 0
    n_mid = int((same & (step >= lo)).sum())
    if n_mid:
        bad.append(f"{n_mid} intraday step(s) of {lo:g}+ points")

    # The strong one: a whole session on the wrong contract. Consecutive
    # *weekdays* are separated only by the maintenance break, so the adjusted
    # price barely moves across it (see SESSION_GAP_LIMIT). A session a roll
    # spread out of line shows up here even when the step onto it hid inside an
    # overnight gap, which is exactly how the morning of 2025-03-18 stayed on
    # the wrong contract undetected during the NQData build.
    #
    # Weekends and holidays are NOT consecutive weekdays -- PropSim's tape is
    # 24-hour, so Friday's close sits next to Sunday's Globex open in the array
    # with a real multi-day gap between them, and that gap is legitimately
    # roll-sized. Restricting the check to boundaries exactly one calendar day
    # apart, both Monday-Friday, is what keeps it from firing on every weekend.
    edges = np.flatnonzero(np.r_[True, np.diff(day) != 0])
    ends = np.r_[edges[1:] - 1, len(ts) - 1]
    from_day = day[ends[:-1]]
    into_day = day[edges[1:]]
    weekday_from = (from_day + 3) % 7          # 0=Monday .. 6=Sunday, day 0 = Thu 1970-01-01
    weekday_into = (into_day + 3) % 7
    consecutive_weekday = ((into_day - from_day) == 1) & (weekday_from < 5) & (weekday_into < 5)
    gaps = np.abs(px[edges[1:]] - px[ends[:-1]])
    over = np.flatnonzero(consecutive_weekday & (gaps > SESSION_GAP_LIMIT))
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


def cache_path() -> Path:
    return tape.CACHE_DIR / f"{ALL}.npz"


def meta_path() -> Path:
    return tape.CACHE_DIR / f"{ALL}.json"


def load_meta():
    try:
        return json.loads(meta_path().read_text())
    except (OSError, ValueError):
        return None


def find_holes(days_u: np.ndarray) -> list:
    """Gaps between consecutive present sessions bigger than a long weekend.

    A diff over the (already sorted, already unique) session-day array: a
    normal week is gap 1, a normal weekend is gap 2, a holiday long weekend is
    gap 3 -- see MAX_NORMAL_GAP_DAYS. Anything past that is a coverage hole:
    no cached contract had data for that stretch, not the market being
    closed. Nothing else in the audit looks for this -- a hole doesn't
    corrupt a price or misplace a roll, it just makes a backtest run on a
    fraction of the trades someone expects with nothing on screen saying why.
    """
    gaps = np.diff(days_u)
    over = np.flatnonzero(gaps > MAX_NORMAL_GAP_DAYS)
    return [dict(**{"from": tape.date_str(int(days_u[i]))},
                 to=tape.date_str(int(days_u[i + 1])),
                 sessions_missing=int(gaps[i] - 1))
            for i in over]


def to_raw(px, ts, meta):
    """Adjusted price -> the price that actually traded.

    The roll table is five numbers in the sidecar, so this costs nothing per
    tick. Storing an adjustment column would spend 368 MB holding a
    piecewise-constant function with five distinct values.
    """
    if not meta or not meta.get("roll_ts"):
        return px
    cum = np.asarray(meta["roll_cum"], np.float64)
    idx = np.searchsorted(np.asarray(meta["roll_ts"], np.int64), ts, side="right")
    off = np.r_[cum, 0.0][idx]
    return (px - off).astype(px.dtype)


def contracts_on_disk() -> list:
    """Cached contracts of the ALL instrument, in calendar order."""
    return sorted((c for c in tape.cached_contracts()
                   if c.split()[0] == ROOT and c != ALL and " " in c),
                  key=month_key)


def daily_volume(contract: str) -> dict:
    """{day_index: volume} for one contract, reading only ts and vol.

    np.savez stores each array separately, so this touches 12 bytes per tick
    instead of 17 and never holds more than one contract at a time.
    """
    z = np.load(tape.cache_path(contract))
    d = tape.day_index(z["ts"])
    v = z["vol"].astype(np.float64)
    u, inv = np.unique(d, return_inverse=True)
    return dict(zip(u.tolist(), np.bincount(inv, weights=v).tolist()))


def build(on_progress=None) -> dict:
    """Stitch every cached NQ contract into ALL.npz. Writes nothing if the audit fails.

    Two passes, one contract resident at a time. Holding all of them plus the
    concatenated result at once is roughly 3.4 GB and does not survive a laptop.
    """
    cs = contracts_on_disk()
    if len(cs) < 2:
        raise SystemExit(f"ALL needs at least two cached {ROOT} contracts, "
                         f"found {len(cs)}. Prepare them on the Data tab first.")

    total = len(cs) * 2
    step = [0]

    def tick(ticks=0):
        step[0] += 1
        if on_progress:
            on_progress(step[0], total, ticks)

    daily = {}
    for c in cs:
        daily[c] = daily_volume(c)
        tick()

    front = front_months(daily)
    mine = {c: sorted(d for d, k in front.items() if k == c) for c in cs}

    parts_ts, parts_px, parts_vol, parts_side = [], [], [], []
    rolls = []
    prev_c = None
    prev_ts = prev_px = prev_lastd = None

    for c in cs:
        days = mine.get(c) or []
        z = np.load(tape.cache_path(c))
        ts, px = z["ts"], z["px"]
        if days and prev_c is not None:
            # Measure on the day the OUTGOING contract last owned, not the day
            # the new one first owns -- those are different days on real data
            # (e.g. NQ 06-26 ends 06-11, NQ 09-26 starts 06-12) and the outgoing
            # contract has already stopped printing on the new contract's first
            # day, so matching there finds nothing. prev_lastd is the day both
            # still trade. The roll's recorded `day` stays days[0] -- that is
            # where ownership switches and where the adjustment boundary sits,
            # even though the spread itself was observed the day before.
            sp, n = measure_roll(prev_ts, prev_px, ts, px, prev_lastd)
            rolls.append(dict(day=int(days[0]), **{"from": prev_c, "to": c},
                              spread=sp, n=int(n)))
        if days:
            keep = np.isin(tape.day_index(ts), days)
            parts_ts.append(ts[keep]); parts_px.append(px[keep])
            parts_vol.append(z["vol"][keep]); parts_side.append(z["side"][keep])
            # Keep only the LAST session of this contract for the next roll
            # measurement, not the whole contract.
            lastd = days[-1]
            m = tape.day_index(ts) == lastd
            prev_ts, prev_px, prev_c, prev_lastd = ts[m], px[m], c, lastd
        tick(sum(len(p) for p in parts_ts))
        del z, ts, px

    ts = np.concatenate(parts_ts); parts_ts.clear()
    # Stays float32 -- cache stores px that way already, and NQ prices are
    # quarter-point multiples well under float32's 2^24 exact-integer bound
    # (need price*4 <= 16,777,216, real range is ~25,000), so every add below
    # is exact. A float64 round-trip here was the single largest allocation
    # in the real 92M-tick build (~740 MB just for the up-cast, held for the
    # rest of the function) for no precision this data needs.
    px = np.concatenate(parts_px); parts_px.clear()
    vol = np.concatenate(parts_vol); parts_vol.clear()
    side = np.concatenate(parts_side); parts_side.clear()
    order = np.argsort(ts, kind="stable")
    # One array at a time, not `a, b, c, d = a[o], b[o], c[o], d[o]` -- that
    # form builds all four reordered copies before any of the four originals
    # is freed (the whole right-hand tuple evaluates first), doubling peak
    # for the full set at once. Reassigning one name at a time lets each old
    # array drop to refcount 0 -- and get freed -- before the next copy is made.
    ts = ts[order]
    px = px[order]
    vol = vol[order]
    side = side[order]
    del order

    # Back-adjust. The deferred contract trades ABOVE the front, so each roll is
    # an artificial step UP; removing it raises the OLDER segment to meet the
    # newer one. adj is therefore POSITIVE going back in time, and the newest
    # segment is unadjusted -- so recent adjusted prices ARE real prices.
    usable = [r for r in rolls if r["spread"] is not None]
    roll_ts, roll_cum = [], []
    for r in usable:
        first = (np.int64(r["day"]) * 86400 + tape.NET_EPOCH_S) * tape.TPS
        roll_ts.append(int(first))
        px[ts < first] += r["spread"]
    run = 0.0
    for r in reversed(usable):
        run += r["spread"]
        roll_cum.append(run)
    roll_cum.reverse()

    fails = audit(ts, px, rolls)
    if fails:
        raise SystemExit("ALL was NOT written — the stitch failed its audit:\n  "
                         + "\n  ".join(fails))

    days_u = np.unique(tape.day_index(ts))
    meta = dict(
        contracts=cs, ticks=int(len(ts)), sessions=int(len(days_u)),
        first=tape.date_str(int(days_u[0])), last=tape.date_str(int(days_u[-1])),
        roll_ts=roll_ts, roll_cum=roll_cum,
        rolls=[dict(r, date=tape.date_str(r["day"])) for r in rolls],
        holes=find_holes(days_u),
        inputs=fingerprint(),
    )
    tape.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path(), ts=ts, px=px, vol=vol, side=side)
    meta_path().write_text(json.dumps(meta, indent=1))
    return meta


def fingerprint() -> dict:
    """What ALL was built from, so staleness is a comparison and not a guess."""
    out = {}
    for c in contracts_on_disk():
        p = tape.cache_path(c)
        out[c] = dict(mtime=int(p.stat().st_mtime), size=int(p.stat().st_size))
    return out


def is_stale():
    """Does ALL still match the caches it was built from? (stale, why)."""
    m = load_meta()
    if not m:
        return True, "not built yet"
    if not cache_path().exists():
        return True, "ALL.npz is missing"
    now, was = fingerprint(), m.get("inputs") or {}
    added = sorted(set(now) - set(was))
    changed = sorted(c for c in set(now) & set(was) if now[c] != was[c])
    if added:
        return True, f"new contract(s) cached: {', '.join(added)}"
    if changed:
        return True, f"re-parsed since: {', '.join(changed)}"
    return False, ""


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

    # A sound multi-session tape: one session per day, a small overnight move.
    # Base day 20003 is a Monday, so consecutive indices are consecutive
    # weekdays -- exactly the boundaries the session-gap check now looks at.
    def _tape(sess_closes):
        ts, px = [], []
        for i, close in enumerate(sess_closes):
            d = 20003 + i
            s0 = (np.int64(d) * 86400 + tape.NET_EPOCH_S) * tape.TPS
            n = 400
            ts.append(s0 + (36000 + np.arange(n, dtype=np.int64)) * tape.TPS)
            px.append(np.full(n, close, np.float32))
        return np.concatenate(ts), np.concatenate(px)

    # Like _tape but with explicit day numbers, for the weekend-vs-weekday check.
    def _tape_at(days_closes):
        ts, px = [], []
        for d, close in days_closes:
            s0 = (np.int64(d) * 86400 + tape.NET_EPOCH_S) * tape.TPS
            n = 400
            ts.append(s0 + (36000 + np.arange(n, dtype=np.int64)) * tape.TPS)
            px.append(np.full(n, close, np.float32))
        return np.concatenate(ts), np.concatenate(px)

    good_ts, good_px = _tape([23000.0, 23010.0, 23005.0])
    ok_rolls = [dict(day=20004, **{"from": "NQ 09-25", "to": "NQ 12-25"},
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
                 [dict(day=20004, **{"from": "a", "to": "b"},
                       spread=12.0, n=999)]), "12 points is not a roll"

    # Unsorted time.
    assert audit(good_ts[::-1], good_px, ok_rolls), "must reject unsorted ts"

    # A roll-sized step INSIDE one session.
    mid_ts, mid_px = _tape([23000.0])
    mid_px[200:] += 241.75
    assert audit(mid_ts, mid_px, ok_rolls), "must reject an intraday roll step"

    # Unbounded check: a large corruption (1000 points) well above the roll band
    # must also be rejected, proving the upper bound is gone.
    large_ts, large_px = _tape([23000.0])
    large_px[200:] += 1000.0
    assert audit(large_ts, large_px, ok_rolls), "must reject a large intraday corruption"

    # Weekends are not consecutive weekdays. Day 20000 is a Friday, 20002 the
    # following Sunday -- adjacent sessions in a 24-hour tape, real 300-point
    # move across the break, and the check must stay quiet about it.
    fri_sun_ts, fri_sun_px = _tape_at([(20000, 23000.0), (20002, 23300.0)])
    assert audit(fri_sun_ts, fri_sun_px, []) == [], \
        "a weekend gap must not trip the session-gap check"

    # The same 300-point gap between two consecutive weekdays (20003 Monday,
    # 20004 Tuesday) IS a real failure and must still be caught.
    mon_tue_ts, mon_tue_px = _tape_at([(20003, 23000.0), (20004, 23300.0)])
    fails = audit(mon_tue_ts, mon_tue_px, [])
    assert any("session" in f for f in fails), fails

    # The actual bug: a stale pre-open tick and the real Sunday reopen print
    # both land in the SAME calendar day, so a genuine weekend price move
    # used to look like an intraday corruption. 13:00 and 18:00 on the same
    # calendar day straddle the session boundary (18:00 opens the NEXT
    # session) -- a 300-point gap between them is a normal session-open gap,
    # not a same-session jump, and must not trip the intraday check.
    day0 = 20002
    s0 = (np.int64(day0) * 86400 + tape.NET_EPOCH_S) * tape.TPS
    straddle_ts = np.array([s0 + 13 * 3600 * tape.TPS, s0 + 18 * 3600 * tape.TPS], np.int64)
    straddle_px = np.array([23000.0, 23300.0], np.float32)
    fails = audit(straddle_ts, straddle_px, [])
    assert not any("intraday" in f for f in fails), fails

    # The same 300-point gap between two ticks that are BOTH inside one
    # 18:00-17:00 session (19:00 one evening, 10:00 the next morning -- one
    # session, spanning midnight) must still be caught. This is the case that
    # got through before the fix: a same-session corruption must not become
    # invisible just because fixing the false positive above also had to
    # treat 19:00 and next-morning 10:00 as the same trading day.
    s1 = (np.int64(day0 + 1) * 86400 + tape.NET_EPOCH_S) * tape.TPS
    within_ts = np.array([s0 + 19 * 3600 * tape.TPS, s1 + 10 * 3600 * tape.TPS], np.int64)
    within_px = np.array([23000.0, 23300.0], np.float32)
    fails = audit(within_ts, within_px, [])
    assert any("intraday" in f for f in fails), fails

    # to_raw undoes the back-adjustment. Adjustment is POSITIVE going back in
    # time -- the deferred contract trades above the front, so each roll puts an
    # artificial step UP into the raw series, and removing it means raising the
    # older segment to meet the newer one. Getting this sign backwards doubles
    # the step instead of cancelling it, which is what left a 568-point cliff in
    # the NQData build (286 of real spread plus 282 of wrongly-signed fix).
    meta = dict(roll_ts=[1000, 2000], roll_cum=[400.0, 150.0])
    ts_q = np.array([500, 1500, 2500], np.int64)
    px_q = np.array([100.0, 200.0, 300.0], np.float32)
    raw = to_raw(px_q, ts_q, meta)
    assert abs(raw[0] - (100.0 - 400.0)) < 1e-6, raw     # oldest, both rolls
    assert abs(raw[1] - (200.0 - 150.0)) < 1e-6, raw     # middle, one roll
    assert abs(raw[2] - 300.0) < 1e-6, raw               # newest, unadjusted
    assert to_raw(px_q, ts_q, None).tolist() == px_q.tolist(), \
        "no roll table means no mapping"

    # find_holes: consecutive gaps of 1, 1, 2, 3 days (a normal week, a normal
    # weekend, a holiday long weekend) must all be silent -- only the last
    # gap here (49 days) is past MAX_NORMAL_GAP_DAYS and counts as a hole.
    days = np.array([20000, 20001, 20002, 20004, 20007, 20056], np.int64)
    holes = find_holes(days)
    assert len(holes) == 1, holes
    assert holes[0]["from"] == tape.date_str(20007), holes
    assert holes[0]["to"] == tape.date_str(20056), holes
    assert holes[0]["sessions_missing"] == 48, holes

    stale, why = is_stale()
    assert isinstance(stale, bool) and isinstance(why, str), (stale, why)
    if cache_path().exists():
        m = load_meta()
        assert m and m.get("inputs"), "a built ALL must record what built it"

    print("selfcheck OK: front months monotone across an oscillating roll")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="stitch ALL from the caches")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.status:
        m = load_meta()
        print(f"cached {ROOT} contracts: {', '.join(contracts_on_disk()) or 'none'}")
        if not m:
            return print("ALL: not built")
        print(f"ALL: {m['ticks']:,} ticks, {m['sessions']} sessions "
              f"{m['first']}..{m['last']}")
        for r in m["rolls"]:
            sp = "unmeasured" if r["spread"] is None else f"{r['spread']:+.2f} pts"
            print(f"  {r['date']}  {r['from']} -> {r['to']}  {sp}  (n={r['n']})")
        for h in m.get("holes", []):
            print(f"  HOLE: {h['from']} -> {h['to']}  "
                  f"({h['sessions_missing']} sessions missing)")
        return
    if args.build:
        def prog(done, total, ticks):
            print(f"\r  {done}/{total}  {ticks:,} ticks", end="", flush=True)
        m = build(on_progress=prog)
        print(f"\rALL: {m['ticks']:,} ticks, {m['sessions']} sessions "
              f"{m['first']}..{m['last']}, {len(m['rolls'])} rolls "
              f"-> {cache_path()}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
