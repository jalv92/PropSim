#!/usr/bin/env python3
"""Backtest engine: run a strategy over the tick tape at any timeframe.

    python3 engine.py --list
    python3 engine.py --strategy ma_cross --contract "NQ 09-26" --tf 5
    python3 engine.py --strategy sweep_follow --contract "NQ 09-26" --tf 1 \
                      --start 2026-06-15 --end 2026-07-20

Three inputs define a run, and all three are explicit here because leaving any
of them implicit is how incomparable results get compared:

    contract + date range   which market data
    strategy + parameters   what is being tested
    timeframe               what the strategy actually sees

Timeframe is a first-class input rather than a property of the data: bars are
built from ticks on demand, so 1m and 5m runs of the same strategy are measured
against *identical* ticks.

ARCHITECTURE, and it follows a measurement. Signal generation is
path-independent and vectorises over bars. Fill resolution is path-dependent
and cannot -- but it only has to run between entry and exit, which at a few
trades per day is 1-2% of ticks. Running a scalar loop over every tick, as the
obvious implementation does, is roughly 50x more work than the problem needs.

FILLS ARE PESSIMISTIC BY CONSTRUCTION. Entries pay slippage; when a stop and a
target both fall inside the same unresolved window the stop is taken. This is
deliberate: a backtest that flatters itself is worse than no backtest, and
measured on real trades a 0-to-5-tick slippage assumption moved a t-statistic
from +2.19 to +0.45 -- the assumption decides the answer, so it is a parameter
and never a hidden constant.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

import tape as tp

TICK_SIZE = 0.25            # NQ/MNQ; overridden per instrument by the caller
POINT_VALUE = 20.0


# --------------------------------------------------------------------------
# strategy library
# --------------------------------------------------------------------------

@dataclass
class Param:
    default: float
    lo: float
    hi: float
    desc: str


class Strategy:
    """A strategy turns bars (and optionally the raw tape) into entries.

    `entries` returns parallel arrays: the TICK index to enter at, the
    direction, and the stop/target prices. The engine resolves the outcome; a
    strategy never decides its own fill, which keeps the pessimism in one place.
    """
    name = "base"
    label = "Base"
    uses_ticks = False
    params: dict[str, Param] = {}

    def entries(self, bars, tape, p):
        raise NotImplementedError


class MACross(Strategy):
    name, label = "ma_cross", "Moving-average cross"
    params = {
        "fast": Param(20, 3, 100, "fast MA, bars"),
        "slow": Param(60, 10, 400, "slow MA, bars"),
        "stop_ticks": Param(40, 4, 200, "stop distance, ticks"),
        "rr": Param(2.0, 0.5, 6.0, "target as a multiple of the stop"),
    }

    def entries(self, bars, tape, p):
        c = bars["c"].astype(np.float64)
        f, s = int(p["fast"]), int(p["slow"])
        if len(c) < s + 2:
            return _empty()
        fa = _sma(c, f)
        sl = _sma(c, s)
        up = fa > sl
        cross = np.zeros(len(c), bool)
        cross[1:] = up[1:] != up[:-1]
        idx = np.flatnonzero(cross)
        idx = idx[idx >= s]
        if not len(idx):
            return _empty()
        direc = np.where(up[idx], 1, -1).astype(np.int8)
        # enter on the first tick of the NEXT bar: acting on the signal bar's
        # own close would be lookahead
        nxt = idx + 1
        ok = nxt < len(bars["start"])
        idx, direc, nxt = idx[ok], direc[ok], nxt[ok]
        entry_tick = bars["start"][nxt]
        px = bars["o"][nxt].astype(np.float64)
        stop = px - direc * p["stop_ticks"] * TICK_SIZE
        target = px + direc * p["stop_ticks"] * p["rr"] * TICK_SIZE
        return entry_tick, direc, stop, target


class ORB(Strategy):
    name, label = "orb", "Opening-range breakout"
    params = {
        "range_min": Param(15, 5, 90, "opening range, minutes"),
        "stop_ticks": Param(40, 4, 200, "stop distance, ticks"),
        "rr": Param(2.0, 0.5, 6.0, "target as a multiple of the stop"),
        "one_per_day": Param(1, 0, 1, "at most one trade per day"),
    }

    def entries(self, bars, tape, p):
        t = bars["t"]
        if not len(t):
            return _empty()
        day = tp.day_index(t)
        sod = tp.sec_of_day(t)
        open_s = 9 * 3600 + 30 * 60
        in_range = sod < open_s + p["range_min"] * 60
        et, dr, st, tg = [], [], [], []
        for d in np.unique(day):
            m = day == d
            rng = m & in_range
            if not rng.any():
                continue
            hi, lo = bars["h"][rng].max(), bars["l"][rng].min()
            after = np.flatnonzero(m & ~in_range)
            for i in after:
                up = bars["h"][i] > hi
                dn = bars["l"][i] < lo
                if not (up or dn):
                    continue
                direc = 1 if up else -1
                level = hi if up else lo
                # LOOKAHEAD TRAP, and the first version fell in it. The breakout
                # is detected from the bar's HIGH/LOW, which is only known once
                # the bar closes -- but the trade happens the instant price
                # crosses the level, mid-bar. Entering at the bar's first tick
                # would buy below the level using end-of-bar information, which
                # is free money that does not exist. It scored t=4.78 and 79%
                # win rate before this was fixed.
                #
                # So find the actual crossing tick inside the bar. The tape is
                # right here; there is no reason to approximate.
                a, b = int(bars["start"][i]), int(bars["end"][i])
                seg = tape["px"][a:b]
                cross = np.flatnonzero(seg > level) if up else np.flatnonzero(seg < level)
                if not len(cross):
                    continue
                entry_i = a + int(cross[0])
                et.append(entry_i); dr.append(direc)
                st.append(level - direc * p["stop_ticks"] * TICK_SIZE)
                tg.append(level + direc * p["stop_ticks"] * p["rr"] * TICK_SIZE)
                if p["one_per_day"]:
                    break
        if not et:
            return _empty()
        return (np.array(et, np.int64), np.array(dr, np.int8),
                np.array(st), np.array(tg))


class SweepFollow(Strategy):
    """Trade with large aggressive sweeps on the tape.

    The order-flow class NinjaTrader cannot backtest correctly at all: it needs
    Tick Replay for the aggressor side AND intrabar fill resolution for the
    stop, and those two are mutually exclusive there. Here both come from the
    same tick stream.
    """
    name, label = "sweep_follow", "Aggressive-sweep follow"
    uses_ticks = True
    params = {
        "min_volume": Param(150, 20, 1000, "contracts in a sweep to signal"),
        "cluster_ms": Param(150, 20, 1000, "max gap folded into one sweep"),
        "cooldown_min": Param(5, 0, 60, "rest after a trade closes, minutes"),
        "stop_ticks": Param(40, 4, 200, "stop distance, ticks"),
        "rr": Param(2.0, 0.5, 6.0, "target as a multiple of the stop"),
    }

    def entries(self, bars, tape, p):
        ts, px, vol, side = tape["ts"], tape["px"], tape["vol"], tape["side"]
        good = side != 0
        ts, px, vol, side = ts[good], px[good], vol[good], side[good]
        if not len(ts):
            return _empty()
        gap = int(p["cluster_ms"]) * (tp.TPS // 1000)
        newcl = np.empty(len(ts), bool)
        newcl[0] = True
        newcl[1:] = (side[1:] != side[:-1]) | (np.diff(ts) > gap)
        starts = np.flatnonzero(newcl)
        ends = np.append(starts[1:], len(ts))
        csum = np.concatenate(([0], np.cumsum(vol.astype(np.int64))))
        clvol = csum[ends] - csum[starts]
        hit = clvol >= p["min_volume"]
        last = ends[hit] - 1                      # the sweep's final print
        if not len(last):
            return _empty()
        direc = side[last].astype(np.int8)
        fill = px[last].astype(np.float64)
        stop = fill - direc * p["stop_ticks"] * TICK_SIZE
        target = fill + direc * p["stop_ticks"] * p["rr"] * TICK_SIZE
        # map back to indices in the FULL tape (the caller resolves there)
        full_idx = np.flatnonzero(good)[last]
        return full_idx, direc, stop, target


LIBRARY = {s.name: s for s in (MACross, ORB, SweepFollow)}


def _empty():
    return (np.array([], np.int64), np.array([], np.int8),
            np.array([]), np.array([]))


def _sma(x, n):
    c = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


# --------------------------------------------------------------------------
# fill resolution
# --------------------------------------------------------------------------

@dataclass
class Costs:
    commission: float = 5.0        # per round trip, currency
    slippage_ticks: float = 2.0    # each way; swept, never assumed away
    tick_size: float = TICK_SIZE
    point_value: float = POINT_VALUE


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    mae: float
    mfe: float
    reason: str

    @property
    def date(self):
        return self.entry_time.date().isoformat()


def resolve(tape, entry_idx, direc, stop, target, costs: Costs,
            cooldown_min=0.0, timeout_min=240.0) -> list[Trade]:
    """Walk ticks from each entry to its exit. One position at a time.

    Bounded forward scans, deliberately: an unbounded per-trade scan to the end
    of the array measured 329s against 2.7s for a plain loop -- 120x SLOWER
    while looking vectorised.
    """
    ts, px = tape["ts"], tape["px"]
    n = len(ts)
    out: list[Trade] = []
    cool = cooldown_min * 60 * tp.TPS
    horizon = int(timeout_min * 60 * tp.TPS)
    free_at = -1
    slip = costs.slippage_ticks * costs.tick_size

    for k in range(len(entry_idx)):
        i0 = int(entry_idx[k])
        if i0 >= n or ts[i0] < free_at:
            continue
        d = int(direc[k])
        fill = float(px[i0]) + d * slip          # pay the spread on entry
        st, tg = float(stop[k]), float(target[k])
        stop_end = np.searchsorted(ts, ts[i0] + horizon, "right")
        lo = hi = fill
        exit_i, exit_px, why = None, None, "timeout"
        for i in range(i0 + 1, min(stop_end, n)):
            p = float(px[i])
            if p < lo: lo = p
            if p > hi: hi = p
            hit_stop = (p <= st) if d > 0 else (p >= st)
            hit_targ = (p >= tg) if d > 0 else (p <= tg)
            if hit_stop:                          # stop wins a tie, on purpose
                exit_i, exit_px, why = i, st, "stop"; break
            if hit_targ:
                exit_i, exit_px, why = i, tg, "target"; break
        if exit_i is None:
            exit_i = min(stop_end, n) - 1
            exit_px = float(px[exit_i])
        exit_px -= d * slip                       # and again on exit
        gross = (exit_px - fill) * d * costs.point_value
        adverse = (lo if d > 0 else hi)
        favorable = (hi if d > 0 else lo)
        out.append(Trade(
            entry_time=tp.to_datetime(ts[i0]), exit_time=tp.to_datetime(ts[exit_i]),
            direction=d, entry_price=fill, exit_price=exit_px,
            pnl=gross - costs.commission,
            mae=min((adverse - fill) * d * costs.point_value, 0.0),
            mfe=max((favorable - fill) * d * costs.point_value, 0.0),
            reason=why))
        free_at = ts[exit_i] + cool
    return out


def backtest(contract, strategy_name, timeframe=5, start=None, end=None,
             params=None, costs: Costs | None = None, rth_only=True):
    """One run. Returns (trades, meta) -- meta records all three inputs."""
    strat = LIBRARY[strategy_name]()
    p = {k: v.default for k, v in strat.params.items()}
    p.update(params or {})
    costs = costs or Costs()

    full = tp.load_cache(contract)
    t = tp.slice_range(full, start, end, rth_only=rth_only)
    if not len(t["ts"]):
        raise SystemExit("no ticks in that range")
    bars = tp.build_bars(t, timeframe)

    ei, dr, st, tg = strat.entries(bars, t, p)
    cd = p.get("cooldown_min", 0.0)
    trades = resolve(t, ei, dr, st, tg, costs, cooldown_min=cd)

    days = np.unique(tp.day_index(t["ts"]))
    meta = dict(contract=contract, strategy=strategy_name, label=strat.label,
                timeframe=timeframe, start=tp.date_str(days[0]),
                end=tp.date_str(days[-1]), days=len(days), rth_only=rth_only,
                params=p, bars=len(bars["t"]), ticks=len(t["ts"]),
                signals=len(ei), trades=len(trades),
                commission=costs.commission, slippage_ticks=costs.slippage_ticks)
    return trades, meta


def summarise(trades, meta):
    if not trades:
        return dict(**meta, pnl=0.0, wr=0.0, per_day=0.0, t_daily=float("nan"))
    pnl = np.array([t.pnl for t in trades])
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t.date] = by_day.get(t.date, 0.0) + t.pnl
    daily = np.array(list(by_day.values()))
    # Cluster-robust: intraday trades share a session and are not independent,
    # so the gate statistic is computed on daily totals, not per trade.
    t_daily = (daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
               if len(daily) > 1 and daily.std(ddof=1) > 0 else float("nan"))
    return dict(**meta, pnl=float(pnl.sum()), wr=float((pnl > 0).mean()),
                mean=float(pnl.mean()), per_day=len(trades) / max(meta["days"], 1),
                t_daily=float(t_daily), t_days=len(daily))


def selfcheck():
    """Regressions for the traps this engine has actually fallen into."""
    contracts = tp.cached_contracts()
    if not contracts:
        print("no tape cached — run: python3 tape.py --build 'NQ 09-26'")
        return
    c = contracts[0]

    # 1. LOOKAHEAD. ORB's opening range is a TIME window, so its high/low is the
    #    same whether drawn as 1m or 15m bars, and the entry is the tick that
    #    crosses it. Identical results across timeframes is therefore the
    #    CORRECT behaviour -- and the first implementation, which entered at the
    #    bar's first tick using end-of-bar information, scored t=4.78 instead.
    res = {tf: summarise(*backtest(c, "orb", tf)) for tf in (1, 5, 15)}
    pnls = {tf: round(r["pnl"], 2) for tf, r in res.items()}
    assert len(set(pnls.values())) == 1, f"ORB must be timeframe-invariant: {pnls}"

    # 2. A bar-counting strategy must NOT be timeframe-invariant, or bars are
    #    not really being rebuilt.
    ma = {tf: summarise(*backtest(c, "ma_cross", tf)) for tf in (1, 15)}
    assert ma[1]["trades"] != ma[15]["trades"], "MA cross should differ by timeframe"

    # 3. Pessimism must be monotonic: more slippage can never help.
    a = summarise(*backtest(c, "orb", 5, costs=Costs(slippage_ticks=0)))
    b = summarise(*backtest(c, "orb", 5, costs=Costs(slippage_ticks=5)))
    assert b["pnl"] < a["pnl"], f"slippage did not hurt: {a['pnl']} -> {b['pnl']}"

    # 4. A date range must actually restrict the data.
    full = summarise(*backtest(c, "orb", 5))
    half = summarise(*backtest(c, "orb", 5, start=None, end="2026-06-30"))
    assert half["days"] < full["days"], "date range had no effect"

    # 5. Every trade must resolve to a real exit reason.
    tr, _ = backtest(c, "orb", 5)
    assert all(t.reason in ("stop", "target", "timeout") for t in tr)
    assert all(t.mae <= 0 <= t.mfe for t in tr), "MAE/MFE signs"

    print(f"selfcheck OK: ORB timeframe-invariant (${pnls[1]:,.0f} at 1/5/15m); "
          f"MA cross varies ({ma[1]['trades']} vs {ma[15]['trades']} trades); "
          f"slippage monotonic (${a['pnl']:,.0f} -> ${b['pnl']:,.0f}); "
          f"date range {full['days']} -> {half['days']} days")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--strategy"); ap.add_argument("--contract")
    ap.add_argument("--tf", type=int, default=5)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--slippage", type=float, default=2.0)
    ap.add_argument("--sweep-tf", action="store_true",
                    help="run 1/3/5/15m on identical ticks")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    if args.list or not args.strategy:
        print(f"{'name':<16}{'ticks?':>7}   parameters")
        for n, S in LIBRARY.items():
            print(f"{n:<16}{('yes' if S.uses_ticks else '-'):>7}   "
                  + ", ".join(f"{k}={v.default:g}" for k, v in S.params.items()))
        print(f"\ncached contracts: {', '.join(tp.cached_contracts()) or 'none'}")
        return

    costs = Costs(slippage_ticks=args.slippage)
    tfs = [1, 3, 5, 15] if args.sweep_tf else [args.tf]
    print(f"{'tf':>4}{'bars':>9}{'signals':>9}{'trades':>8}{'/day':>7}"
          f"{'P&L':>12}{'WR':>7}{'t(daily)':>10}")
    for tf in tfs:
        tr, meta = backtest(args.contract, args.strategy, tf,
                            args.start, args.end, costs=costs)
        s = summarise(tr, meta)
        print(f"{tf:>4}{s['bars']:>9,}{s['signals']:>9,}{s['trades']:>8,}"
              f"{s['per_day']:>7.1f}{s['pnl']:>12,.0f}{s['wr']:>7.1%}"
              f"{s['t_daily']:>10.2f}")
    print(f"\n{meta['contract']} {meta['start']}..{meta['end']} "
          f"({meta['days']} days), slippage {args.slippage} ticks/side")


if __name__ == "__main__":
    main()
