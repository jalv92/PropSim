#!/usr/bin/env python3
"""One continuous NQ tape, stitched from every cached contract month."""
from __future__ import annotations

import argparse


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
