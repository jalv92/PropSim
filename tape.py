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
    truncated = 0
    for fi, f in enumerate(files):
        if on_progress and fi % 25 == 0:
            on_progress(fi, len(files), len(ts))
        n0 = len(ts)
        try:
            for tt, price, boff, aoff, v in read_ticks(f):
                ts.append(tt); px.append(price); vol.append(v)
                side.append(BUY if aoff == 0 else (SELL if boff == 0 else UNKNOWN))
        except Exception as e:
            # ponytail: one corrupt/unsupported hour truncates instead of
            # aborting the whole build -- matches ncd_parse.minute_features(),
            # which has the same guard. Real data hit this once in 5,560
            # hourly files across all NQ contracts: a single byte carrying an
            # undocumented flag value that even the upstream reference parser
            # (bboyle1234/NTDFileReader) can't decode either, so this is a
            # one-off anomaly in that hour, not a systematic bug. The
            # generator already yielded some ticks before it raised -- those
            # stay (verified: 202509120700.Last.ncd raised at byte 287 after
            # 39 ticks, and those 39 are in the cache) -- only the REST of
            # that hour is lost, so "truncated" is the honest word, not
            # "skipped".
            truncated += 1
            print(f"  truncated {f.name} after {len(ts) - n0} ticks: {e}")

    if not ts:
        raise SystemExit(f"{contract}: 0 ticks parsed from {len(files)} files "
                         f"({truncated} failed) -- refusing to write an empty cache")

    arr = dict(ts=np.asarray(ts, np.int64), px=np.asarray(px, np.float32),
               vol=np.asarray(vol, np.int32), side=np.asarray(side, np.int8))
    order = np.argsort(arr["ts"], kind="stable")   # hourly files can interleave
    arr = {k: v[order] for k, v in arr.items()}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # `truncated` travels IN the cache file, not just this process's return
    # value -- a rebuild months later that hits the same anomaly must not be
    # indistinguishable from a clean one just because nobody was watching
    # stdout when it ran.
    np.savez(out, truncated=np.int64(truncated), **arr)
    if on_progress:
        on_progress(len(files), len(files), len(arr["ts"]))
    return {**arr, "truncated": truncated}


def load_cache(contract: str, start=None, end=None) -> dict:
    """The tape, optionally narrowed to a date range as it loads.

    An .npz member has no seek-a-slice path: np.load always decompresses a
    key into a full ndarray, so narrowing does not save anything on the read
    itself. What it saves is what happens after. The tape is sorted by time,
    so a date range is a CONTIGUOUS SLICE, found with two searchsorted calls
    on `ts`. A numpy slice is a view backed by the full buffer, so each key's
    slice is `.copy()`-ed into just the wanted rows -- one key at a time, so
    the full array is freed before the next key is read -- which lets the
    full buffer be released instead of held alive by the view. `slice_range`
    still owns the RTH filter, since that one is not contiguous, but it now
    runs on this narrowed result rather than on the whole tape, so the
    boolean-mask copy it used to build on top of the full arrays is gone too.
    When the requested range covers the whole tape (or no range is given),
    nothing is copied -- that would just double the peak for no benefit.
    """
    # ALL is stitched from the other caches and carries a sidecar roll table.
    # Imported here rather than at module scope: `continuous` imports `tape`,
    # and a top-level import either way makes that circular.
    if contract == "ALL":
        import continuous as _c
        p = _c.cache_path()
        if not p.exists():
            raise SystemExit("ALL has not been built yet. Open the Data tab "
                             "and press “Update ALL”.")
    else:
        p = cache_path(contract)
        if not p.exists():
            raise SystemExit(f"{contract} has not been prepared yet. Open the "
                             f"Backtest tab and press “Prepare this contract” once.")
    z = np.load(p)
    ts_full = z["ts"]
    i0, i1 = 0, len(ts_full)
    if start or end:
        epoch = datetime(1970, 1, 1).date()
        if start:
            d0 = (datetime.fromisoformat(start).date() - epoch).days
            i0 = int(np.searchsorted(
                ts_full, (np.int64(d0) * 86400 + NET_EPOCH_S) * TPS, "left"))
        if end:
            d1 = (datetime.fromisoformat(end).date() - epoch).days + 1
            i1 = int(np.searchsorted(
                ts_full, (np.int64(d1) * 86400 + NET_EPOCH_S) * TPS, "left"))
    narrow = (i0, i1) != (0, len(ts_full))
    out = {"ts": ts_full[i0:i1].copy() if narrow else ts_full}
    del ts_full
    for k in ("px", "vol", "side"):
        full = z[k]
        out[k] = full[i0:i1].copy() if narrow else full
    return out


def truncated_hours(contract: str) -> int:
    """How many hourly files were truncated by a parse error when this cache
    was built, read back from the cache file itself -- durable, so it answers
    the same whether the cache was just built or has sat there for months.
    0 for a cache built before this field existed, same as a clean one.
    """
    p = cache_path(contract)
    if not p.exists():
        return 0
    z = np.load(p)
    return int(z["truncated"]) if "truncated" in z.files else 0


def cached_contracts() -> list[str]:
    """Every prepared contract, ALL first because it is the broadest sample."""
    if not CACHE_DIR.is_dir():
        return []
    names = sorted(p.stem.replace("_", " ") for p in CACHE_DIR.glob("*.npz"))
    return ([n for n in names if n == "ALL"]
            + [n for n in names if n != "ALL"])


def sample_contract():
    """A single real contract month, for code that just needs representative data.

    ALL is a stitched tape of every contract and can run to gigabytes; it is
    the wrong thing to reach for when the caller only wants "some ticks to
    exercise". Selfchecks that took cached_contracts()[0] silently switched to
    it the moment it was first built, and still passed, which is how a
    thirteen-million-tick check quietly becomes a hundred-million-tick one.
    """
    return next((c for c in cached_contracts() if c != "ALL"), None)


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
    """First session, last session, session count.

    ALL answers from its sidecar. Reading it off the tape would load 1.5 GB to
    produce three values, and the Data tab asks on every refresh.
    """
    if contract == "ALL":
        import continuous as _c
        m = _c.load_meta()
        if m:
            return m["first"], m["last"], m["sessions"]
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
    # ALL (~517 MB cached) may now sort first; keep this check fast and
    # independent of whether the ALL tape has been built by picking a real
    # contract month instead.
    c = sample_contract() or contracts[0]
    tape = load_cache(c)
    assert (np.diff(tape["ts"]) >= 0).all(), "tape must be time-ordered"

    # Loading a date range must return exactly what loading everything and then
    # slicing returns. It exists because prepare() otherwise materialises the
    # whole tape and copies the part it wants: at ALL's 1.5 GB that peaks at
    # 3.1 GB and does not survive a laptop.
    lo_d, hi_d, _ = available_range(c)
    ranged = load_cache(c, start=lo_d, end=lo_d)
    manual = slice_range(tape, start=lo_d, end=lo_d, rth_only=False)
    assert len(ranged["ts"]) == len(manual["ts"]), \
        (len(ranged["ts"]), len(manual["ts"]))
    assert (ranged["ts"] == manual["ts"]).all(), "ranged load must match slicing"
    assert (ranged["px"] == manual["px"]).all(), "ranged load lost prices"
    assert len(load_cache(c, start=hi_d, end=hi_d)["ts"]) > 0, "last day empty"

    rth = slice_range(tape, rth_only=True)
    s = sec_of_day(rth["ts"])
    assert s.min() >= 9 * 3600 + 30 * 60 and s.max() < 16 * 3600, "RTH filter leaked"

    # A contract whose every hour fails to parse must raise, not write an
    # empty cache -- a future NT8 format change has to be a loud crash, not a
    # silently empty tape nobody notices until a backtest runs on nothing.
    # One crafted file: a valid 28-byte header followed by a record whose
    # time-flag bits (6) no known parser of this format decodes -- the same
    # failure real data hit once in NQ 12-25, just with zero surviving ticks.
    import struct, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp) / "db" / "tick" / "FAKE 01-99"
        cdir.mkdir(parents=True)
        header = struct.pack("<Iddq", 0, 0.25, 23000.0, 0)
        (cdir / "202501010000.Last.ncd").write_bytes(header + bytes([0b00000110, 0]))
        try:
            build_cache("FAKE 01-99", nt_root=tmp)
            assert False, "must raise on an all-corrupt contract"
        except SystemExit as e:
            assert "0 ticks" in str(e), e
        assert not cache_path("FAKE 01-99").exists(), "must not write an empty cache"

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

    # ALL is a VALUE in the contract slot, not a new type: everything
    # downstream -- slice_range, build_bars, engine.prepare, optimize.sweep --
    # must keep working without knowing it exists.
    import continuous as cont
    if cont.cache_path().exists():
        a = load_cache(cont.ALL)
        assert set(a) == {"ts", "px", "vol", "side"}, sorted(a)
        assert (np.diff(a["ts"]) >= 0).all(), "ALL must be time-ordered"
        lo_a, hi_a, nd_a = available_range(cont.ALL)
        # 79 sessions are on disk as of the current cache (full stitch of all
        # NQ contracts is Task 7's job) -- 50 is a floor well under that,
        # not the target session count.
        assert nd_a > 50, f"ALL should span many sessions, got {nd_a}"
        assert cached_contracts()[0] == cont.ALL, "ALL must sort first"
        b = build_bars(slice_range(a, rth_only=True), 300)
        assert len(b["t"]) > 0, "ALL must bar like any other contract"
        # Prove the sidecar path is TAKEN, not merely that it agrees with the tape.
        # Both return the same three values -- that is what the sidecar is for --
        # so equality cannot tell them apart; the previous attempt at this
        # assertion passed with the branch deleted. A sentinel can tell them apart:
        # these values exist nowhere in the tape.
        real = cont.load_meta
        try:
            cont.load_meta = lambda: {"first": "1999-01-01", "last": "1999-12-31",
                                      "sessions": 7}
            assert available_range(cont.ALL) == ("1999-01-01", "1999-12-31", 7), \
                "available_range('ALL') must read the sidecar, not load the tape"
        finally:
            cont.load_meta = real
        print(f"  ALL: {len(a['ts']):,} ticks, {nd_a} sessions {lo_a}..{hi_a}")
    else:
        print("  ALL not built yet — run: python3 continuous.py --build")


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
        n = truncated_hours(args.build)
        warn = f" ({n} hour(s) truncated by parse errors)" if n else ""
        print(f"cached {args.build}: {len(t['ts']):,} ticks, {nd} days, "
              f"{lo}..{hi}{warn} -> {cache_path(args.build)}")
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
