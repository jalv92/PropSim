#!/usr/bin/env python3
"""Measure a slippage floor from your own tape, instead of guessing one.

    python3 slippage.py "NQ 09-26"
    python3 slippage.py "NQ 09-26" --files 80
    python3 slippage.py --selfcheck

WHY THIS IS NOT COSMETIC. Measured on 416 real trades during this project's
research, moving entry slippage from 0 to 5 ticks moved the t-statistic from
+2.19 to +0.45. The relevant denominator is the edge, not the risk unit: against
a 6.3-tick edge, 3 ticks of slippage is 48% of it. So the assumption decides the
answer, which is exactly why the engine takes it as a parameter -- and why the
parameter should come from data.

WHAT IS MEASURABLE HERE, AND WHAT IS NOT. Real depth (L2) would give queue
position, and it is not available: the Market Replay `.nrd` event stream carries
an out-of-spec volume opcode that silently corrupts any decoder written outside
NinjaTrader, so PropSim reads only its header. What IS available is better than
nothing and honest about what it is: the `.Last` tick stream carries the bid and
ask **at every trade**, so the spread you actually faced is recoverable, tick by
tick, for your own instrument and your own months.

Two numbers come out of that:

    spread     A market order crosses it. Half the spread is the minimum cost of
               going to market, and it is a FLOOR: no execution beats it.
    tick-gap   How far price jumped between consecutive prints. A stop resting in
               that gap fills on the far side of it, so the gap distribution
               bounds stop slippage in the fast moments that matter.

The output is a floor and a stress value, never an auto-applied constant. A tool
that quietly lowers your slippage assumption because a calm month measured well
is a tool that flatters you.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

import ntdata
from ncd_parse import read_ticks


def _tick_size(path: Path) -> float:
    """The instrument's increment, straight from the file header."""
    raw = path.read_bytes()[:28]
    if len(raw) < 28:
        return 0.0
    inc, _p0, _t0 = struct.unpack_from("<ddq", raw, 4)
    return float(inc)


RTH = (9 * 3600 + 30 * 60, 16 * 3600)      # same window the engine trades
TPS = 10_000_000
NET_EPOCH_S = 62135596800


def measure(contract: str, nt_root=None, files=40, rth_only=True) -> dict:
    """Spread and tick-gap distributions from a sample of a contract's tape.

    RTH-ONLY BY DEFAULT, and this is not a detail. Measured across all 24 hours
    the median NQ spread came out at 4 ticks, which is true of 3am and irrelevant
    to a strategy the engine only ever runs between 09:30 and 16:00. A slippage
    floor calibrated on the overnight session would be roughly 4x too pessimistic
    and would bury every real edge. The measurement window must match the trading
    window.
    """
    root = ntdata.resolve_root(nt_root)
    if root is None:
        raise SystemExit("No NinjaTrader 8 folder found.")
    cdir = Path(root) / "db" / "tick" / contract
    allf = sorted(cdir.glob("*.Last.ncd"))
    if not allf:
        raise SystemExit(f"no .ncd files under {cdir}")

    # Spread the candidates across the WHOLE period rather than taking the first
    # N: volatility is seasonal, and the first fortnight of a contract is not
    # representative of the month it expires in. Candidates are over-sampled
    # because most hourly files hold no RTH ticks at all.
    step = max(1, len(allf) // max(files * 6, 1))
    candidates = allf[::step]

    tick = _tick_size(allf[0]) or 0.25
    spreads: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    n_ticks = 0
    used = 0
    for f in candidates:
        if used >= files:
            break
        px, sp = [], []
        try:
            for tt, price, boff, aoff, _v in read_ticks(f):
                if rth_only:
                    sod = (tt // TPS - NET_EPOCH_S) % 86400
                    if not (RTH[0] <= sod < RTH[1]):
                        continue
                px.append(price)
                sp.append(boff + aoff)      # both are offsets in TICKS from last
        except Exception:
            continue                        # a corrupt hour is skipped, not fatal
        if len(px) < 500:
            continue                        # too thin to say anything
        n_ticks += len(px)
        used += 1
        spreads.append(np.asarray(sp, np.float32))
        d = np.abs(np.diff(np.asarray(px, np.float64))) / tick
        gaps.append(d[d > 0])

    if not spreads:
        raise SystemExit(f"could not read any {'RTH ' if rth_only else ''}ticks "
                         f"for {contract}")
    sample = spreads       # only its length is used below
    sp = np.concatenate(spreads)
    gp = np.concatenate(gaps) if gaps else np.array([1.0])

    spread_p50 = float(np.percentile(sp, 50))
    spread_p90 = float(np.percentile(sp, 90))
    gap_p90 = float(np.percentile(gp, 90))
    floor = round(spread_p50 / 2.0, 2)
    stress = round(max(spread_p90 / 2.0, gap_p90), 2)
    return dict(
        contract=contract, files=len(sample), of_files=len(allf), ticks=n_ticks,
        tick_size=tick, session="RTH 09:30-16:00" if rth_only else "all hours",
        spread_mean=round(float(sp.mean()), 3),
        spread_p50=spread_p50, spread_p90=spread_p90,
        wide_pct=round(float((sp > 1).mean()) * 100, 2),
        gap_mean=round(float(gp.mean()), 3), gap_p90=gap_p90,
        gap_over_1_pct=round(float((gp > 1).mean()) * 100, 2),
        floor_ticks=floor, stress_ticks=stress,
        note=(f"Half the median spread is {floor:g} ticks — no execution beats "
              f"that. In the fastest 10% of moments price jumped {gap_p90:g} "
              f"ticks between prints, so a stop resting there fills that far "
              f"through: {stress:g} ticks is the stress assumption. Measured on "
              f"{n_ticks:,} of your own ticks across {len(sample)} sampled hours "
              f"({'RTH only' if rth_only else 'all 24 hours'}); real queue "
              f"position needs L2, which cannot be decoded outside NinjaTrader."))


def selfcheck():
    """Runs against whatever tape this machine has; skips cleanly if none."""
    root = ntdata.resolve_root()
    tick_dir = (Path(root) / "db" / "tick") if root else None
    contracts = ([p.name for p in sorted(tick_dir.iterdir()) if p.is_dir()]
                 if tick_dir and tick_dir.is_dir() else [])
    if not contracts:
        print("no tick data on this machine — nothing to measure")
        return
    r = measure(contracts[0], files=4)

    # A spread cannot be negative, and on a liquid future it cannot be huge.
    assert 0 < r["spread_p50"] <= 20, r
    assert r["spread_p90"] >= r["spread_p50"], r
    # The stress value must never sit below the floor, or the two would invite
    # picking the smaller number and calling it conservative.
    assert r["stress_ticks"] >= r["floor_ticks"], r
    assert r["ticks"] > 1000, r
    assert 0 < r["tick_size"] < 100, r

    # THE REGRESSION THAT MATTERS: the overnight session is genuinely wider, so
    # measuring all 24 hours must not be allowed to masquerade as the RTH number
    # the engine actually needs. If these two ever come out equal, the session
    # filter has stopped working.
    allh = measure(contracts[0], files=4, rth_only=False)
    assert allh["spread_p50"] >= r["spread_p50"], (r, allh)
    print(f"selfcheck OK: {r['contract']} RTH spread p50 {r['spread_p50']:g} / p90 "
          f"{r['spread_p90']:g} ticks over {r['ticks']:,} ticks; "
          f"floor {r['floor_ticks']:g}, stress {r['stress_ticks']:g}; "
          f"all-hours median {allh['spread_p50']:g} confirms the session filter bites")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contract", nargs="?")
    ap.add_argument("--files", type=int, default=40, help="hours to sample")
    ap.add_argument("--root")
    ap.add_argument("--all-hours", action="store_true",
                    help="measure the 24h session instead of RTH")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if not args.contract:
        raise SystemExit('usage: python3 slippage.py "NQ 09-26"')
    r = measure(args.contract, args.root, args.files, not args.all_hours)
    print(f"{r['contract']}: {r['ticks']:,} ticks from {r['files']} of "
          f"{r['of_files']} hourly files, tick size {r['tick_size']:g}, "
          f"{r['session']}\n")
    print(f"  spread      mean {r['spread_mean']:g}  median {r['spread_p50']:g}  "
          f"p90 {r['spread_p90']:g} ticks   ({r['wide_pct']:g}% wider than 1 tick)")
    print(f"  tick gap    mean {r['gap_mean']:g}  p90 {r['gap_p90']:g} ticks   "
          f"({r['gap_over_1_pct']:g}% of moves jumped more than 1 tick)")
    print(f"\n  floor  {r['floor_ticks']:g} ticks/side")
    print(f"  stress {r['stress_ticks']:g} ticks/side")
    print(f"\n{r['note']}")


if __name__ == "__main__":
    main()
