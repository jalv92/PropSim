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

    A setup that enters on a resting limit (a retrace to a level) may return a
    fifth array of limit prices; the engine then fills there instead of at the
    tick that reached it.
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
        # own close would be lookahead. And that next bar has to be in the SAME
        # session -- a cross detected at 16:00 must not open a position at 09:30
        # the following morning on a stale signal, across an unseen overnight gap.
        day = tp.day_index(bars["t"])
        nxt = idx + 1
        ok = (nxt < len(bars["start"])) & (day[np.minimum(nxt, len(day) - 1)] == day[idx])
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


class FVG(Strategy):
    """Fair-value gap: trade the retrace back into a three-bar imbalance.

    A bullish gap exists when bar i's low sits ABOVE bar i-2's high -- price
    left a band nobody traded through. The trade is the pullback into that band.

    THE LOOKAHEAD TRAP IS THE SAME ONE ORB FELL INTO, and it bites harder here:
    the gap is only known once bar i has closed, and the entry is a LIMIT level
    that price touches mid-bar. Entering at a bar's open because that bar's low
    reached the level would fill at a price the market never offered on the way
    in. So the entry tick is found by scanning the tape for the first print that
    actually trades at or through the level.
    """
    name, label = "fvg", "Fair-value gap retrace"
    uses_ticks = True
    params = {
        "min_ticks": Param(8, 1, 200, "smallest gap worth trading, ticks"),
        "expiry_bars": Param(12, 1, 200, "bars the gap stays valid"),
        "stop_ticks": Param(40, 4, 200, "stop distance, ticks"),
        "rr": Param(2.0, 0.5, 6.0, "target as a multiple of the stop"),
        "one_per_day": Param(0, 0, 1, "at most one trade per day"),
    }

    def entries(self, bars, tape, p):
        h, l = bars["h"], bars["l"]
        n = len(h)
        if n < 4:
            return _empty()
        gap = p["min_ticks"] * TICK_SIZE
        px, ts = tape["px"], tape["ts"]
        exp = int(p["expiry_bars"])
        day = tp.day_index(bars["t"])
        # last bar of each bar's own session, so a retrace window cannot wait for
        # the level overnight and get filled by tomorrow's opening gap
        last_bar = np.searchsorted(day, day, "right") - 1
        et, dr, st, tg, lim = [], [], [], [], []
        used_days = set()

        # i indexes the third bar of the pattern, so everything below is known at
        # its close. A gap needs the two bars NOT to overlap at all:
        #   bullish   low[i]  >  high[i-2]
        #   bearish   high[i] <  low[i-2]
        # The first version tested |low[i] - high[i-2]| and called the negative
        # side a bearish gap, which is simply "bar i's low is below bar i-2's
        # high" -- true of almost every ordinary overlapping bar. It generated 43
        # trades a day and +$85K over 23 days, which is what a definition error
        # looks like from the outside: not a crash, a fantastic result.
        bull_gap = l[2:] - h[:-2]
        bear_gap = l[:-2] - h[2:]
        cand = np.flatnonzero((bull_gap >= gap) | (bear_gap >= gap))
        for k in cand:
            i = k + 2
            if i + 1 >= n:
                continue
            d = day[i]
            if p["one_per_day"] and d in used_days:
                continue
            bull = bull_gap[k] >= gap
            level = l[i] if bull else h[i]         # near edge of the gap
            a = int(bars["start"][i + 1])
            b = int(bars["end"][min(i + exp, int(last_bar[i]), n - 1)])
            seg = px[a:b]
            if not len(seg):
                continue
            hit = np.flatnonzero(seg <= level) if bull else np.flatnonzero(seg >= level)
            if not len(hit):
                continue
            j = a + int(hit[0])
            direc = 1 if bull else -1
            et.append(j); dr.append(direc)
            st.append(level - direc * p["stop_ticks"] * TICK_SIZE)
            tg.append(level + direc * p["stop_ticks"] * p["rr"] * TICK_SIZE)
            lim.append(level)
            used_days.add(d)
        if not et:
            return _empty()
        order = np.argsort(np.asarray(et))
        return (np.asarray(et, np.int64)[order], np.asarray(dr, np.int8)[order],
                np.asarray(st)[order], np.asarray(tg)[order],
                # the entry IS the level: a limit resting in the gap
                np.asarray(lim)[order])


class VWAPRevert(Strategy):
    """Fade a stretch away from the session VWAP.

    VWAP is computed from the ticks of the CURRENT session only -- carrying it
    across the overnight break would anchor the morning to yesterday's business
    and quietly turn a mean-reversion signal into a gap-continuation one.

    The stretch is measured in ticks rather than in standard deviations: a
    rolling sigma of the distance is itself a function of the distance, which
    makes the threshold move with the thing it is supposed to be measuring.
    """
    name, label = "vwap_revert", "VWAP reversion"
    uses_ticks = True
    params = {
        "stretch_ticks": Param(60, 4, 600, "distance from VWAP to fade, ticks"),
        "min_bar": Param(6, 0, 78, "skip this many bars after the open"),
        "cooldown_min": Param(15, 0, 240, "rest after a trade closes, minutes"),
        "stop_ticks": Param(40, 4, 200, "stop distance, ticks"),
        "rr": Param(1.0, 0.3, 6.0, "target as a multiple of the stop"),
    }

    def entries(self, bars, tape, p):
        t = bars["t"]
        if len(t) < 3:
            return _empty()
        px, vol = tape["px"].astype(np.float64), tape["vol"].astype(np.float64)
        starts, ends = bars["start"], bars["end"]
        pv = np.add.reduceat(px * vol, starts)
        vv = np.add.reduceat(vol, starts)
        day = tp.day_index(t)
        new_day = np.empty(len(t), bool)
        new_day[0] = True
        new_day[1:] = day[1:] != day[:-1]
        # per-session cumulative sums, restarted at each new day
        cpv = np.zeros(len(t))
        cvv = np.zeros(len(t))
        acc_pv = acc_vv = 0.0
        bar_of_day = np.zeros(len(t), np.int32)
        cnt = 0
        for i in range(len(t)):
            if new_day[i]:
                acc_pv = acc_vv = 0.0
                cnt = 0
            acc_pv += pv[i]
            acc_vv += vv[i]
            cpv[i], cvv[i] = acc_pv, acc_vv
            bar_of_day[i] = cnt
            cnt += 1
        vwap = cpv / np.maximum(cvv, 1.0)

        stretch = p["stretch_ticks"] * TICK_SIZE
        far = bars["c"] - vwap
        sig = np.flatnonzero((np.abs(far) >= stretch)
                             & (bar_of_day >= int(p["min_bar"])))
        # act on the NEXT bar's first tick: the signal uses this bar's close. Same
        # session only -- a stretch measured at the close is not a reason to buy
        # tomorrow's open.
        nxt = sig + 1
        ok = (nxt < len(starts)) & (day[np.minimum(nxt, len(day) - 1)] == day[sig])
        sig, nxt = sig[ok], nxt[ok]
        if not len(sig):
            return _empty()
        direc = np.where(far[sig] > 0, -1, 1).astype(np.int8)   # fade the stretch
        entry_tick = starts[nxt]
        fill = px[entry_tick]
        stop = fill - direc * p["stop_ticks"] * TICK_SIZE
        target = fill + direc * p["stop_ticks"] * p["rr"] * TICK_SIZE
        return entry_tick.astype(np.int64), direc, stop, target


LIBRARY = {s.name: s for s in (MACross, ORB, SweepFollow, FVG, VWAPRevert)}


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
    # Worst fall from the RUNNING peak inside this trade, currency, >= 0. This
    # is what an intraday trailing floor actually tests, and neither excursion
    # nor both together can stand in for it: a trade that ran +1500, fell to
    # -600, then ran +1800 has MAE -600 and MFE +1800 with the low arriving
    # FIRST, so any model built on the two extremes and their order says the
    # floor had not ratcheted yet and the account lived. It did not -- the peak
    # at +1500 had already moved the floor and the fall to -600 is 2100 deep.
    #
    # With this, the breach test inside a trade is exact for ANY path shape:
    #
    #     breach  <=>  max(hwm - balance - mae, intra_mdd) >= max_dd
    #
    # (the first term is the fall from a peak the ACCOUNT already held, the
    # second the fall from a peak set inside this trade). `None` means UNKNOWN
    # -- a source reporting two magnitudes and no path, i.e. the NinjaTrader
    # database and the add-on export -- and the simulator then substitutes
    # `mfe - mae`, its upper bound, rather than believing a number nobody
    # measured.
    intra_mdd: float | None = None

    @property
    def date(self):
        return self.entry_time.date().isoformat()


MAX_GAP_S = 120.0           # a longer silence inside a session is missing data


def resolve(tape, entry_idx, direc, stop, target, costs: Costs,
            cooldown_min=0.0, timeout_min=240.0, limit_px=None,
            max_gap_s=MAX_GAP_S, day=None) -> list[Trade]:
    """Walk ticks from each entry to its exit. One position at a time.

    Bounded forward scans, deliberately: an unbounded per-trade scan to the end
    of the array measured 329s against 2.7s for a plain loop -- 120x SLOWER
    while looking vectorised.

    `limit_px` models a RESTING LIMIT order instead of a market order: the fill
    is the limit price, not the price of the tick that reached it. A retrace
    entry is a limit by nature, and the difference is not cosmetic -- filling at
    the touched tick in a fast move puts the entry far past the level the stop
    was measured from, which is how a stop-out came to book a +$17,635 profit
    before this existed.
    """
    ts, px = tape["ts"], tape["px"]
    n = len(ts)
    out: list[Trade] = []
    cool = cooldown_min * 60 * tp.TPS
    horizon = int(timeout_min * 60 * tp.TPS)
    free_at = -1
    slip = costs.slippage_ticks * costs.tick_size

    # EVERY POSITION IS FLAT AT THE SESSION CLOSE, and this is not a preference.
    # The tape is RTH-filtered, so 16:00 of one day sits immediately next to 09:30
    # of the next IN THE ARRAY. A scan that ignores the boundary lets the next
    # session's opening gap "breach" today's stop: one FVG trade entered at
    # 30611.75 was filled out at 29718.50 -- 893 points, -$17,870 -- on a stop 40
    # ticks away. It also fed a 893-point overnight gap into the intraday drawdown
    # test, which is the number this whole product exists to get right.
    # A function of `ts` alone, so a sweep computes it once in `prepare` and
    # hands it down. Recomputing it here cost 34% of a sweep's wall clock: two
    # integer divisions over 12M ticks, per combination, for an answer that
    # cannot change while the tape does not.
    if day is None:
        day = tp.day_index(ts)
    gap_limit = int(max_gap_s * tp.TPS)

    for k in range(len(entry_idx)):
        i0 = int(entry_idx[k])
        if i0 >= n or ts[i0] < free_at:
            continue
        d = int(direc[k])
        if limit_px is None:
            fill = float(px[i0]) + d * slip      # market order pays the spread
        else:
            fill = float(limit_px[k])            # resting limit fills at its price
        session_end = int(np.searchsorted(day, day[i0], "right"))
        st, tg = float(stop[k]), float(target[k])
        # INVARIANT: a trade cannot open already past its own stop or target.
        # Nobody fills you and then pays you for being stopped out, and nobody
        # hands you a target you were already through. Both were reachable when
        # the levels came from a signal price and the fill came from a later tick.
        if (fill <= st if d > 0 else fill >= st):
            continue
        if (fill >= tg if d > 0 else fill <= tg):
            continue
        stop_end = min(int(np.searchsorted(ts, ts[i0] + horizon, "right")),
                       session_end)
        lo = hi = fill
        mdd = 0.0                   # running fall from the running peak
        exit_i, exit_px, why = None, None, ("close" if stop_end == session_end
                                            else "timeout")
        for i in range(i0 + 1, min(stop_end, n)):
            # A HOLE IN THE TAPE IS NOT A PRICE MOVE. This contract's data has
            # missing hours -- 2026-07-17 jumps 3,538 seconds in one step, and a
            # second hole of 377 seconds moved price 98 points. Resolving a stop
            # across one of those books a fill the market never printed: it turned
            # a 40-tick stop into a -$1,960 loss. Close at the last real tick and
            # label it, so the run reports missing data instead of inventing a
            # catastrophe from it.
            if ts[i] - ts[i - 1] > gap_limit:
                exit_i, exit_px, why = i - 1, float(px[i - 1]), "gap"
                break
            p = float(px[i])
            if p < lo: lo = p
            if p > hi: hi = p
            # Measured against the RUNNING peak, so it is taken before the exit
            # tests below -- a stop-out's own breaching tick is part of the fall
            # that an intraday floor would have been tested against.
            drop = (hi - p) if d > 0 else (p - lo)
            if drop > mdd: mdd = drop
            hit_stop = (p <= st) if d > 0 else (p >= st)
            hit_targ = (p >= tg) if d > 0 else (p <= tg)
            if hit_stop:                          # stop wins a tie, on purpose
                # Fill at the WORSE of the stop level and the price that breached
                # it. A stop is a market order: when price jumps straight through
                # it, the fill is on the far side of the jump, not at the level.
                # Filling at the level was worth real money -- trades showed a
                # worst excursion of -$510 while booking the nominal -$212 loss,
                # and the tick-gap distribution says the fastest 10% of moves jump
                # 5 ticks between prints.
                exit_px = min(st, p) if d > 0 else max(st, p)
                exit_i, why = i, "stop"; break
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
            reason=why, intra_mdd=mdd * costs.point_value))
        free_at = ts[exit_i] + cool
    return out


def prepare(contract, timeframe=5, start=None, end=None, rth_only=True) -> dict:
    """Load, slice and bar the tape once, for reuse across many runs.

    Measured on 28 days of NQ: loading the cache, slicing to RTH and building
    bars costs 0.66 s, while the strategy and the fill loop cost 0.15 s. A sweep
    that calls `backtest` naively therefore spends 80% of its time re-reading
    data that did not change -- hoisting this out makes a 200-combination sweep
    roughly five times faster.
    """
    full = tp.load_cache(contract)
    t = tp.slice_range(full, start, end, rth_only=rth_only)
    if not len(t["ts"]):
        raise SystemExit("no ticks in that range")
    dayi = tp.day_index(t["ts"])
    days = np.unique(dayi)
    # EVERYTHING BELOW IS A FUNCTION OF THE TAPE, NOT OF THE PARAMETERS, and it
    # used to be recomputed inside every `backtest` call. Profiled on a 16-
    # combination sweep over 12.1M ticks, the three of them -- day_index, unique
    # and the two diffs -- were 73% of the run, against 19% for the fill loop
    # that does the actual simulating. A sweep is not compute-bound on the
    # simulation; it was compute-bound on constants.
    dt = np.diff(t["ts"])
    holes = np.flatnonzero((dt > MAX_GAP_S * tp.TPS) & (np.diff(dayi) == 0))
    return dict(contract=contract, tape=t, bars=tp.build_bars(t, timeframe),
                timeframe=timeframe, rth_only=rth_only, days=len(days),
                start=tp.date_str(days[0]), end=tp.date_str(days[-1]),
                dayi=dayi, n_holes=len(holes),
                hole_days=sorted({tp.date_str(int(dayi[i])) for i in holes}))


def backtest(contract, strategy_name, timeframe=5, start=None, end=None,
             params=None, costs: Costs | None = None, rth_only=True, ctx=None):
    """One run. Returns (trades, meta) -- meta records all three inputs.

    `ctx` is an optional prepared tape from `prepare()`; pass it to run many
    parameter sets against the same data without re-reading it.
    """
    strat = LIBRARY[strategy_name]()
    p = {k: v.default for k, v in strat.params.items()}
    p.update(params or {})
    costs = costs or Costs()

    if ctx is None:
        ctx = prepare(contract, timeframe, start, end, rth_only)
    elif ctx["timeframe"] != timeframe:
        raise SystemExit(f"prepared tape is {ctx['timeframe']}m, asked for {timeframe}m")
    t, bars = ctx["tape"], ctx["bars"]

    res = strat.entries(bars, t, p)
    ei, dr, st, tg = res[:4]
    # A retrace strategy rests a limit at its level; a breakout crosses the
    # spread. A strategy that trades on a limit says so by returning a fifth
    # array, so the fill model is a property of the setup rather than an accident
    # of who wrote the entry function.
    lim = res[4] if len(res) > 4 else None
    cd = p.get("cooldown_min", 0.0)
    trades = resolve(t, ei, dr, st, tg, costs, cooldown_min=cd, limit_px=lim,
                     day=ctx["dayi"])

    # Data quality, reported rather than assumed: a silence longer than MAX_GAP_S
    # inside a session means hourly files are missing, and every bar spanning one
    # is fiction. The user needs to see that before believing a P&L built on it.
    # Measured once per tape in `prepare`, not once per parameter set.
    meta = dict(contract=contract, strategy=strategy_name, label=strat.label,
                timeframe=timeframe, start=ctx["start"],
                end=ctx["end"], days=ctx["days"], rth_only=rth_only,
                params=p, bars=len(bars["t"]), ticks=len(t["ts"]),
                signals=len(ei), trades=len(trades),
                data_holes=ctx["n_holes"], hole_days=ctx["hole_days"],
                gap_exits=sum(1 for x in trades if x.reason == "gap"),
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

    # 4. A date range must actually restrict the data. The cut is derived from what
    #    this contract actually holds: a hardcoded date silently stopped restricting
    #    anything the day a contract whose tape ends earlier became `contracts[0]`,
    #    and the test failed for the data rather than for the code.
    lo, hi, _ = tp.available_range(c)
    cut = tp.date_str(np.unique(tp.day_index(tp.load_cache(c)["ts"]))[-2])
    full = summarise(*backtest(c, "orb", 5))
    half = summarise(*backtest(c, "orb", 5, start=None, end=cut))
    assert half["days"] < full["days"], (
        f"date range had no effect: {c} spans {lo}..{hi}, cut at {cut}")

    # 5. Every trade must resolve to a real exit reason.
    tr, _ = backtest(c, "orb", 5)
    assert all(t.reason in ("stop", "target", "timeout", "close", "gap") for t in tr)
    assert all(t.mae <= 0 <= t.mfe for t in tr), "MAE/MFE signs"

    # 6. THE INVARIANT THAT CAUGHT A REAL BUG. A stop-out cannot make money and a
    #    target cannot lose it. Both were violated when a strategy measured its
    #    stop from a signal LEVEL while the engine filled at a later tick: one
    #    stop-out booked +$17,635 and the strategy showed +$34K over 23 days.
    #    Checked across every strategy, because the mistake is not local to one.
    costs = Costs()
    slip_cost = costs.slippage_ticks * costs.tick_size * costs.point_value
    for name in LIBRARY:
        trades, meta = backtest(c, name, 5, costs=costs)
        # 7. NO TRADE MAY SPAN TWO DATES. The tape is RTH-filtered, so the array
        #    hides the overnight break entirely: 16:00 sits next to 09:30. Every
        #    strategy fell through it, and the damage was not subtle -- an FVG
        #    long entered at 30611.75 was stopped out at 29718.50 by the next
        #    morning's gap, booking -$17,870 on a 40-tick stop and feeding an
        #    893-point overnight excursion into the intraday drawdown test.
        for t in trades:
            assert t.entry_time.date() == t.exit_time.date(), (
                f"{name}: trade spans {t.entry_time} -> {t.exit_time}")
        for t in trades:
            if t.reason == "stop":
                assert t.pnl < 0, f"{name}: profitable stop-out {t.pnl:+.0f}"
                # A stop fills at the price that breached it, and that price is
                # by definition the worst seen -- anything worse earlier would
                # have triggered first. So the loss is exactly the excursion plus
                # exit slippage and commission. Any drift here means the stop has
                # gone back to filling at its level through a gap.
                want = t.mae - slip_cost - costs.commission
                assert abs(t.pnl - want) < 0.01, (
                    f"{name}: stop paid {t.pnl:.2f}, excursion implies {want:.2f}")
            elif t.reason == "target":
                assert t.pnl > 0, f"{name}: losing target exit {t.pnl:+.0f}"
        # 8b. THE FALL FROM THE RUNNING PEAK is what an intraday trailing floor
        #     tests, and it is bounded on both sides by the two excursions.
        #     Below |MAE| would mean the trade never fell as far as its own low
        #     (the peak is at worst the entry, which is where the fall is
        #     measured from). Above MFE-MAE would mean it fell further than the
        #     distance between its best and worst prices. A violation of either
        #     bound means the running peak is being read from the wrong side of
        #     the trade, and every Apex-style pass rate built on it is wrong.
        for t in trades:
            assert -t.mae <= t.intra_mdd + 0.01, (
                f"{name}: fall {t.intra_mdd:.2f} shallower than its low {-t.mae:.2f}")
            assert t.intra_mdd <= t.mfe - t.mae + 0.01, (
                f"{name}: fall {t.intra_mdd:.2f} exceeds peak-to-trough "
                f"{t.mfe - t.mae:.2f}")
            assert t.mfe >= t.pnl - 0.01, (
                f"{name}: peak {t.mfe:.2f} below realised {t.pnl:.2f}")
            # And for a stop-out the upper bound is TIGHT. A stop ends AT its
            # worst price -- anything worse earlier would have triggered first
            # -- so every peak the trade ever made is behind the low, and the
            # fall from the running peak is exactly peak-to-trough. This is the
            # one exit whose path shape is known without looking, which makes it
            # the check that catches an off-by-one in the running maximum.
            if t.reason == "stop":
                assert abs(t.intra_mdd - (t.mfe - t.mae)) < 0.01, (
                    f"{name}: stop-out fell {t.intra_mdd:.2f}, peak-to-trough "
                    f"is {t.mfe - t.mae:.2f}")
        # 8. A stop cannot lose an unbounded amount. Slipping through its level is
        #    real: measured on this tape the median gap-through is 0-3 ticks and
        #    p90 is 3-8, with a tail to 62 ticks for the strategies that enter in
        #    the fastest moments. What is NOT real is 392 or 3,573 ticks, which is
        #    what resolving a fill across a hole in the tape or across the
        #    overnight break produced. The bound sits above the measured tail and
        #    far below those, so it catches the class of bug without encoding noise.
        nominal = meta["params"]["stop_ticks"] * costs.tick_size * costs.point_value
        bound = -(nominal + 100 * costs.tick_size * costs.point_value)
        for t in trades:
            if t.reason == "stop":
                assert t.pnl >= bound, (
                    f"{name}: stop lost {t.pnl:,.0f}, bound {bound:,.0f} "
                    f"({meta['data_holes']} data holes in range)")

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

    # Your own strategies count as strategies everywhere, not only in the UI.
    try:
        import plugins
        plugins.register_all()
    except Exception:
        pass

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
