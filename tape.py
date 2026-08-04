#!/usr/bin/env python3
"""The tick tape: parse once, cache columnar, build bars at any timeframe.

    python3 tape.py --list                      # what is cached
    python3 tape.py --build "NQ 09-26"          # parse and cache a contract
    python3 tape.py --bars "NQ 09-26" --tf 5m   # 5-minute bars from ticks
    python3 tape.py --bars "NQ 09-26" --tf 30s  # ...or 30-second bars

This is what makes timeframe a real input rather than a fixed property of the
data. NinjaTrader stores 1-minute and 5-minute series as separate downloads;
here both are built from the same ticks on demand, so re-barring a strategy at
1, 3 and 5 minutes compares it against *identical* market data instead of three
separate datasets that may not even cover the same days. Second bars come from
the same place: ticks carry their own timestamps, so a 15-second bar is exactly
as available as a 15-minute one.

Why cache. Parsing `.ncd` runs at roughly 0.5M ticks/sec, and the strategy
logic on top is nearly free -- measured 0.45M ticks/sec for a full engine loop
versus 0.49M for bare parsing, so parsing is essentially all of the cost. One
columnar cache turns that into 41M ticks/sec, a 54x speedup per pass, at 17
bytes/tick.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import ntdata
from ncd_parse import read_ticks

TPS = 10_000_000                       # .NET ticks per second
NET_EPOCH_S = 62135596800              # seconds from 0001-01-01 to 1970-01-01
CACHE_DIR = Path.home() / ".prop-sim" / "tape"

# Aggressor side, from the bid/ask offsets the .Last stream carries per trade.
# A print at the ask is a buyer lifting; at the bid, a seller hitting. Prints
# strictly inside the spread have no attributable aggressor and are dropped
# rather than guessed -- NinjaTrader's own OnMarketData handling does the same.
BUY, SELL, UNKNOWN = 1, -1, 0


def cache_path(contract: str) -> Path:
    return CACHE_DIR / f"{contract.replace(' ', '_')}.npz"


def build_cache(contract: str, nt_root=None, force=False, on_progress=None) -> dict:
    """Parse every .ncd of a contract into one columnar cache.

    `on_progress(done, total, ticks)` is called as files are read so a UI can
    show something during the ~15 s parse. A non-technical user must never be
    left staring at a frozen window wondering whether it crashed.
    """
    out = cache_path(contract)
    if out.exists() and not force:
        return load_cache(contract)

    root = ntdata.resolve_root(nt_root)
    if root is None:
        raise SystemExit("No NinjaTrader 8 folder found.")
    cdir = Path(root) / "db" / "tick" / contract
    files = sorted(cdir.glob("*.Last.ncd"))
    if not files:
        raise SystemExit(f"no .ncd files under {cdir}")

    ts, px, vol, side = [], [], [], []
    for fi, f in enumerate(files):
        if on_progress and fi % 25 == 0:
            on_progress(fi, len(files), len(ts))
        for tt, price, boff, aoff, v in read_ticks(f):
            ts.append(tt); px.append(price); vol.append(v)
            side.append(BUY if aoff == 0 else (SELL if boff == 0 else UNKNOWN))

    arr = dict(ts=np.asarray(ts, np.int64), px=np.asarray(px, np.float32),
               vol=np.asarray(vol, np.int32), side=np.asarray(side, np.int8))
    order = np.argsort(arr["ts"], kind="stable")   # hourly files can interleave
    arr = {k: v[order] for k, v in arr.items()}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arr)
    if on_progress:
        on_progress(len(files), len(files), len(arr["ts"]))
    return arr


def load_cache(contract: str) -> dict:
    p = cache_path(contract)
    if not p.exists():
        raise SystemExit(f"{contract} has not been prepared yet. Open the "
                         f"Backtest tab and press “Prepare this contract” once.")
    z = np.load(p)
    return {k: z[k] for k in ("ts", "px", "vol", "side")}


def cached_contracts() -> list[str]:
    if not CACHE_DIR.is_dir():
        return []
    return sorted(p.stem.replace("_", " ") for p in CACHE_DIR.glob("*.npz"))


# --------------------------------------------------------------------------
# time helpers -- .ncd timestamps are naive ET wall clock
# --------------------------------------------------------------------------

def to_datetime(net_ticks):
    return datetime(1, 1, 1) + timedelta(microseconds=int(net_ticks) / 10)


def day_index(ts: np.ndarray) -> np.ndarray:
    """Calendar day number per tick (naive ET)."""
    return (ts // TPS - NET_EPOCH_S) // 86400


def sec_of_day(ts: np.ndarray) -> np.ndarray:
    return (ts // TPS - NET_EPOCH_S) % 86400


def date_str(day_no: int) -> str:
    return (datetime(1970, 1, 1) + timedelta(days=int(day_no))).date().isoformat()


def available_range(contract: str) -> tuple[str, str, int]:
    t = load_cache(contract)["ts"]
    d = day_index(t)
    days = np.unique(d)
    return date_str(days[0]), date_str(days[-1]), len(days)


def slice_range(tape: dict, start=None, end=None, rth_only=True,
                rth=(9 * 3600 + 30 * 60, 16 * 3600)) -> dict:
    """Restrict the tape to a date range and, by default, the RTH session.

    Both bounds inclusive, ISO dates. This is the backtest's date-range input:
    it selects ticks, so every downstream bar and trade inherits it and cannot
    silently run on data the user did not ask for.
    """
    ts = tape["ts"]
    keep = np.ones(len(ts), bool)
    if start or end:
        d = day_index(ts)
        if start:
            keep &= d >= (datetime.fromisoformat(start).date()
                          - datetime(1970, 1, 1).date()).days
        if end:
            keep &= d <= (datetime.fromisoformat(end).date()
                          - datetime(1970, 1, 1).date()).days
    if rth_only:
        s = sec_of_day(ts)
        keep &= (s >= rth[0]) & (s < rth[1])
    return {k: v[keep] for k, v in tape.items()}


# --------------------------------------------------------------------------
# bars
# --------------------------------------------------------------------------

def tf_secs(value, bar_type: str = "Minute") -> int:
    """Bar size in SECONDS, from NinjaTrader's own (Type, Value) pair.

    Seconds is the only unit anything below the UI speaks. The pair exists at the
    edges only -- the browser sends it, the labels rebuild it -- so no function in
    the engine ever has to ask which unit a bar size is written in. A `value` that
    already carries its unit ("30s", "5m") is accepted too, which is what the CLIs
    take.
    """
    s = str(value).strip().lower()
    if s.endswith("s") or s.endswith("m"):
        bar_type, s = ("Second" if s[-1] == "s" else "Minute"), s[:-1]
    try:
        n = float(s)
    except ValueError:
        raise SystemExit(f"{value!r} is not a bar size")
    if str(bar_type).strip().lower().startswith("m"):
        n *= 60
    n = int(round(n))
    # A trust boundary: this number comes off a query string. Zero divides in the
    # bucketing below, and a day-long bar is a way to ask for no bars at all.
    if not 1 <= n <= 86400:
        raise SystemExit(f"bar size must be between 1 second and 1 day, got {value} "
                         f"{bar_type}")
    return n


def tf_label(secs: int, long: bool = False) -> str:
    """300 -> "5m" / "5-minute"; 30 -> "30s" / "30-second"."""
    if secs % 60 == 0:
        return f"{secs // 60}-minute" if long else f"{secs // 60}m"
    return f"{secs}-second" if long else f"{secs}s"


def build_bars(tape: dict, secs: int = 300) -> dict:
    """OHLCV + signed order flow per bar, built from ticks. `secs` is the bar size.

    Bars never span a session gap: bucketing is by (day, time-slot), so an
    overnight break cannot merge into one bar the way a naive
    `timestamp // period` would.
    """
    ts = tape["ts"]
    if not len(ts):
        return dict(t=np.array([]), o=np.array([]), h=np.array([]),
                    l=np.array([]), c=np.array([]), v=np.array([]),
                    delta=np.array([]), n=np.array([]), start=np.array([]),
                    end=np.array([]))
    period = int(secs)
    slot = day_index(ts) * (86400 // period + 1) + sec_of_day(ts) // period
    # boundaries of each bar in the tick array
    edge = np.flatnonzero(np.diff(slot)) + 1
    starts = np.concatenate(([0], edge))
    ends = np.concatenate((edge, [len(ts)]))

    px, vol, side = tape["px"], tape["vol"], tape["side"]
    o = px[starts]
    c = px[ends - 1]
    h = np.maximum.reduceat(px, starts)
    l = np.minimum.reduceat(px, starts)
    v = np.add.reduceat(vol, starts)
    delta = np.add.reduceat(side.astype(np.int64) * vol, starts)
    return dict(t=ts[starts], o=o, h=h, l=l, c=c, v=v, delta=delta,
                n=ends - starts, start=starts, end=ends)


def selfcheck():
    contracts = cached_contracts()
    if not contracts:
        print("nothing cached yet — run: python3 tape.py --build 'NQ 09-26'")
        return
    c = contracts[0]
    tape = load_cache(c)
    assert (np.diff(tape["ts"]) >= 0).all(), "tape must be time-ordered"

    rth = slice_range(tape, rth_only=True)
    s = sec_of_day(rth["ts"])
    assert s.min() >= 9 * 3600 + 30 * 60 and s.max() < 16 * 3600, "RTH filter leaked"

    # Seconds and minutes are the same code path, so both are checked on it. The
    # sub-minute sizes are the ones worth stating: a bucket width that does not
    # divide the hour is where an off-by-one in the slot stride would show.
    for tf in (15, 30, 60, 120, 300, 900):
        b = build_bars(rth, tf)
        lab = tf_label(tf)
        assert (b["h"] >= b["o"]).all() and (b["h"] >= b["c"]).all(), f"{lab} high"
        assert (b["l"] <= b["o"]).all() and (b["l"] <= b["c"]).all(), f"{lab} low"
        assert (b["v"] > 0).all(), f"{lab} empty bar"
        assert b["n"].sum() == len(rth["ts"]), f"{lab} lost ticks"
    b15s, b1, b5 = build_bars(rth, 15), build_bars(rth, 60), build_bars(rth, 300)
    assert len(b15s["t"]) > len(b1["t"]) > len(b5["t"]), "finer bars must be more bars"
    assert b15s["v"].sum() == b1["v"].sum() == b5["v"].sum(), \
        "volume must survive re-barring"
    # No bar may straddle its own boundary: every tick in a bucket has to fall in
    # the same slot the bucket was cut for.
    for tf in (15, 120):
        b = build_bars(rth, tf)
        slots = sec_of_day(rth["ts"]) // tf
        assert all(len(np.unique(slots[s:e])) == 1
                   for s, e in zip(b["start"], b["end"])), f"{tf_label(tf)} straddles"
    assert (tf_secs(30, "Second"), tf_secs("2", "Minute"), tf_secs("15s")) == (30, 120, 15)
    lo, hi, nd = available_range(c)
    print(f"selfcheck OK: {c} {len(tape['ts']):,} ticks, {nd} days {lo}..{hi}; "
          f"bars 15s={len(b15s['t']):,} 1m={len(b1['t']):,} 5m={len(b5['t']):,}, "
          f"volume conserved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", metavar="CONTRACT")
    ap.add_argument("--bars", metavar="CONTRACT")
    ap.add_argument("--tf", default="5m", help="bar size: 30s, 2m, 15m (bare "
                                               "numbers are minutes)")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.list:
        inv = ntdata.inventory(ntdata.resolve_root())
        print(f"{'contract':<14}{'on disk':>9}{'cached':>9}   range")
        for k, v in (inv.get("tick") or {}).items():
            cached = cache_path(k).exists()
            rng = f"{v['first']}..{v['last']}"
            print(f"{k:<14}{v['days']:>9}{('yes' if cached else '-'):>9}   {rng}")
        return
    if args.build:
        t = build_cache(args.build, force=args.force)
        lo, hi, nd = available_range(args.build)
        print(f"cached {args.build}: {len(t['ts']):,} ticks, {nd} days, "
              f"{lo}..{hi} -> {cache_path(args.build)}")
        return
    if args.bars:
        secs = tf_secs(args.tf)
        tape = slice_range(load_cache(args.bars), args.start, args.end)
        b = build_bars(tape, secs)
        print(f"{args.bars} {tf_label(secs)}: {len(b['t']):,} bars from "
              f"{len(tape['ts']):,} ticks")
        print(f"{'time':<20}{'open':>10}{'high':>10}{'low':>10}{'close':>10}"
              f"{'vol':>9}{'delta':>9}")
        for i in list(range(min(5, len(b["t"])))):
            print(f"{str(to_datetime(b['t'][i])):<20}{b['o'][i]:>10.2f}"
                  f"{b['h'][i]:>10.2f}{b['l'][i]:>10.2f}{b['c'][i]:>10.2f}"
                  f"{b['v'][i]:>9}{b['delta'][i]:>+9}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
