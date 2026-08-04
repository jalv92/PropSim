#!/usr/bin/env python3
"""Backtest engine: run a strategy over the tick tape at any timeframe.

    python3 engine.py --list
    python3 engine.py --strategy ma_cross --contract "NQ 09-26" --tf 5m
    python3 engine.py --strategy sweep_follow --contract "NQ 09-26" --tf 30s \
                      --start 2026-06-15 --end 2026-07-20

Three inputs define a run, and all three are explicit here because leaving any
of them implicit is how incomparable results get compared:

    contract + date range   which market data
    strategy + parameters   what is being tested
    timeframe               what the strategy actually sees

Timeframe is a first-class input rather than a property of the data: bars are
built from ticks on demand, so 1m and 5m runs of the same strategy are measured
against *identical* ticks. It is carried in SECONDS (`tf_secs`) so that a
30-second bar and a 2-minute bar are the same kind of number; NinjaTrader's
Type/Value pair is rebuilt only for display, by `tape.tf_label`.

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
    # A DECISION, NOT A DIAL. A session window is a statement about when you are
    # willing to trade; sweeping it answers "which hours of these 27 days paid
    # best", which is fitting the calendar and charges the ledger for the
    # privilege. The optimizer leaves these at their default and says so, the
    # same way it already does for a flag.
    fixed: bool = False


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
    # A strategy whose session is not RTH says so HERE rather than relying on the
    # caller to remember. An overnight setup handed an RTH-filtered tape does not
    # fail -- it silently finds nothing, which reads exactly like "no edge".
    full_session = False
    # Seconds past midnight ET at which this setup's TRADING day starts. It is 0
    # -- the calendar day -- for everything that trades one RTH session, and it
    # matters only to the daily governor: `tape.day_index` cuts at midnight, so a
    # setup whose session opens at 18:00 would otherwise have its evening windows
    # and the next morning's 09:30 window counted as two different days, while
    # NinjaTrader resets its daily P&L once, at the session begin.
    session_offset_s = 0
    params: dict[str, Param] = {}

    def entries(self, bars, tape, p):
        raise NotImplementedError

    def risk_ticks(self, p) -> float:
        """Widest this setup may place its stop from the entry, ticks.

        The selfcheck bounds every stop-out against it, so a strategy whose stop
        is structural rather than a fixed distance has to say how wide that can
        get -- otherwise the check either crashes or silently stops checking.
        """
        return p.get("stop_ticks", 200)


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


class RangeBreak(Strategy):
    """Break out of a consolidation: a tight range, then the first tick through it.

    The range is the last `lookback_bars` CLOSED bars, and it only counts as an
    accumulation if its width sits between `min_range_ticks` and
    `max_range_ticks` -- a floor as well as a ceiling, because a two-tick range
    in dead tape is not a coiled market, it is an empty one, and its stop is
    thinner than the slippage it pays to get in.

    An armed range is good for the NEXT bar only. That is not a restriction:
    if price stays inside, the window slides one bar and re-arms itself with the
    updated levels, so there is nothing for an expiry parameter to do.

    THE LOOKAHEAD TRAP IS ORB'S, and the same fix. The range is known at bar i's
    close, but the trade happens the instant price crosses the level, mid-bar.
    Entering at the next bar's open would buy at a price chosen with knowledge of
    where that bar ended up. So the entry is the first tick in the tape that
    actually trades through the level.

    STOP, TARGET AND BREAKEVEN ARE ALL IN ATR, so the trade is priced in what
    the market is currently doing rather than in a tick count that was right on
    some other week. ATR is read at the SIGNAL bar, which has closed; the entry
    happens in the next one.

    `max_stop_ticks` is not a preference. An ATR stop has no upper bound of its
    own, and on NQ 1m the ATR(14) runs 45 ticks at its 10th percentile and 398
    at its worst -- a volatility spike would otherwise silently size a $2,000
    stop into a $50K evaluation. A trade whose stop would exceed it is skipped,
    not shrunk: shrinking it would keep the trade and quietly change the setup.
    """
    name, label = "range_break", "Consolidation breakout"
    uses_ticks = True
    params = {
        # The width defaults come from the WIDTH DISTRIBUTION of the tape, not
        # from a backtest: over 27 days of NQ 1m, a 10-bar window spans 109
        # ticks at its 10th percentile, 146 at its 25th and 218 at its median.
        # So "consolidation" is set at roughly the quietest quartile, and the
        # floor sits below the 10th -- narrower than that on NQ is dead tape,
        # not a coil. Picking these by which pair made money would be fitting
        # the defaults to 27 days; that is what the sweep is for, on data you
        # then hold back.
        "lookback_bars": Param(10, 3, 60, "bars that form the range"),
        "min_range_ticks": Param(30, 1, 200, "narrowest range worth trading, ticks"),
        "max_range_ticks": Param(140, 4, 400, "widest range still a consolidation, ticks"),
        "atr_period": Param(14, 2, 100, "bars in the ATR"),
        "atr_stop": Param(1.0, 0.1, 5.0, "stop distance, in ATRs"),
        "atr_target": Param(2.0, 0.2, 10.0, "target distance, in ATRs"),
        "max_stop_ticks": Param(160, 8, 800, "skip the trade if the ATR stop is wider"),
        # Percent of the run from the entry to the TARGET, not of the risk: at
        # the defaults (stop 1 ATR, target 2) half the run is 1 ATR, which is
        # exactly 1R. Zero turns it off.
        "be_trigger_pct": Param(50, 0, 100, "move the stop to breakeven after "
                                            "this % of the way to the target"),
        # Minutes after midnight, the same unit the session parameters already
        # use elsewhere. HHMM reads nicer and is a trap: a step of 465 turns 930
        # into 1395, and 13:95 is not a time.
        "session_start": Param(570, 0, 1439, "no entries before this minute "
                                             "(570 = 09:30)", fixed=True),
        "last_entry": Param(900, 1, 1440, "no entries after this minute "
                                          "(900 = 15:00)", fixed=True),
        "one_per_day": Param(0, 0, 1, "at most one trade per day"),
    }

    def risk_ticks(self, p) -> float:
        return p["max_stop_ticks"]          # the trade is skipped past this

    def entries(self, bars, tape, p):
        h, l = bars["h"], bars["l"]
        n = len(h)
        N = int(p["lookback_bars"])
        if n < N + 2:
            return _empty()
        day = tp.day_index(bars["t"])
        px, ts = tape["px"], tape["ts"]
        t0, t1 = int(p["session_start"]) * 60, int(p["last_entry"]) * 60
        lo_w = p["min_range_ticks"] * TICK_SIZE
        hi_w = p["max_range_ticks"] * TICK_SIZE
        atr = _atr(bars, int(p["atr_period"]), day)
        max_stop = p["max_stop_ticks"] * TICK_SIZE
        be_pct = p["be_trigger_pct"] / 100.0

        top = _roll(h, N, np.max)
        bot = _roll(l, N, np.min)
        # A window that straddles the overnight break is not a range: it spans a
        # gap nobody traded through, which is exactly the thing this strategy is
        # supposed to find INSIDE a session.
        same = np.zeros(n, bool)
        same[N - 1:] = day[N - 1:] == day[:n - N + 1]
        armed = same & (top - bot >= lo_w) & (top - bot <= hi_w)

        et, dr, st, tg, be = [], [], [], [], []
        used_days = set()
        i = N - 1
        while i < n - 1:
            j = i + 1                       # the bar the breakout may happen in
            if not armed[i] or not np.isfinite(atr[i]) or day[j] != day[i]:
                i += 1
                continue
            a, b = int(bars["start"][j]), int(bars["end"][j])
            seg = px[a:b]
            up = np.flatnonzero(seg > top[i])
            dn = np.flatnonzero(seg < bot[i])
            # Both sides can print inside one bar; the first one is the trade.
            k_up = int(up[0]) if len(up) else len(seg)
            k_dn = int(dn[0]) if len(dn) else len(seg)
            if k_up == k_dn == len(seg):
                i += 1
                continue
            direc = 1 if k_up <= k_dn else -1
            e = a + min(k_up, k_dn)
            # The time window is tested on the ENTRY TICK, not on the bar: the
            # entry is intrabar, and a 14:59 bar can break at 15:00:20.
            sod = int(tp.sec_of_day(ts[e:e + 1])[0])
            d = day[j]
            if not (t0 <= sod < t1) or (p["one_per_day"] and d in used_days):
                i += 1
                continue
            risk = p["atr_stop"] * float(atr[i])
            if risk > max_stop:             # a spike-wide stop is not this setup
                i += 1
                continue
            ref = float(px[e])
            reward = p["atr_target"] * float(atr[i])
            et.append(e); dr.append(direc)
            st.append(ref - direc * risk)
            tg.append(ref + direc * reward)
            be.append(ref + direc * be_pct * reward if be_pct > 0 else ref)
            used_days.add(d)
            # Resume AFTER the breakout bar, so one consolidation fires once: the
            # sliding window would otherwise re-detect the same range from every
            # remaining bar of it and queue a stack of stale entries behind the
            # first, each one opening as the previous position closed.
            i = j + 1
        if not et:
            return _empty()
        # Sixth array: the breakeven trigger. Fifth is the limit price, which
        # this setup does not use -- it crosses the spread on a breakout -- so
        # `None` there says "market order" and keeps the positions meaningful.
        return (np.array(et, np.int64), np.array(dr, np.int8),
                np.array(st), np.array(tg), None, np.array(be))


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


class LatigoBreak(Strategy):
    """LatigoBreak: opening-candle breakout with a whipsaw veto, per session window.

    A 1:1 port of `LatigoBreakStrategy.cs` (v3 core -- the v4 big-print flow gate
    is deliberately NOT here, see below). At each enabled window it freezes the
    first `candle_seconds` of tape as an opening range, watches for the first
    print at least one tick beyond it, and refuses the break unless it SURVIVES:
    `hold_seconds` outside with a cumulative excursion of at least
    `extension_r30` x the opening range. A re-entry of one tick back inside
    before that is a whipsaw -- the break is vetoed, the window re-arms, and a
    print already through the OPPOSITE level opens the other side at once.

    THE THREE WINDOWS ARE FIXED OFFSETS FROM THE 18:00 ET GLOBEX REOPEN, exactly
    as in NT8 where they are offsets from `ActualSessionBegin`: +0 (18:00),
    +7200 (20:00), +55800 (09:30 the next morning). Wall-clock offsets are exact
    because every DST change falls inside the closed weekend. Two of the three
    live outside RTH, hence `full_session`.

    DELTAS FROM THE NT8 STRATEGY, all of them forced by the architecture and none
    of them silent:

    1. Brackets are priced off the SIGNAL print, not off the actual fill. NT8
       prices them in `OnExecutionUpdate`; here the fill is the engine's business
       and it adds slippage afterwards, which makes the effective stop slightly
       tighter than NT8's. That is the same convention every other strategy here
       uses, and it errs pessimistic.
    2. (was a delta, now matched) The ATR is Wilder's, extended with the bar in
       formation up to the entry tick -- what `Calculate.OnEachTick` hands NT8 at
       that instant. See `_emit`. It is not lookahead: nothing after the entry
       tick is read.
    3. NT8 skips a whole window it was not already flat at, and re-arms only when
       a trade CLOSES. Entry generation here is path-independent -- it cannot know
       when a position closes -- so the window emits its candidates and the
       engine's one-position-at-a-time rule drops the ones that land while a
       trade is open. With the default `max_trades_per_window=1` the two are the
       same thing.
    4. `_SANITY_STOP_TICKS` is a data-integrity guard, NOT a parameter, and it is
       deliberately not exposed: a tunable that silently drops trades is a second
       strategy wearing the first one's name. It sat at 200 ticks as a dial once
       and threw away 188 of 340 signals over 47 days -- the median ATR stop here
       is 215 ticks -- which made every comparison against NinjaTrader a
       comparison of two different trade sets. The constant sits above the
       measured maximum (799 ticks) and far below what a fill resolved across a
       hole in the tape produces, so on real data it never fires.
    5. The v4 flow gate (big-print support) is out of scope by request. It needs
       aggressor-classified clusters and its own corpus, and the offline scan has
       not cleared it -- porting it here would put an unvalidated filter in the
       optimizer's search space.
    """
    name, label = "latigo_break", "LatigoBreak (opening-candle break, whipsaw veto)"
    uses_ticks = True
    full_session = True                 # 18:00 and 20:00 ET are outside RTH
    session_offset_s = 18 * 3600        # the daily governor resets at the reopen

    # See delta 4. Above the measured maximum, below tape-hole nonsense.
    _SANITY_STOP_TICKS = 1200

    # Seconds from the 18:00 ET session begin, in TRADING-DAY order: the third
    # window lands at 09:30 of the following calendar morning.
    _OFFSETS = (0, 7200, 55800)
    _FLAGS = ("trade_globex_reopen", "trade_evening", "trade_us_open")
    _SESSION_BEGIN = 18 * 3600

    # PARAMETER NAMES ARE THE NT8 PROPERTY NAMES, in snake_case, AND THE LIST IS
    # CLOSED: every dial here exists in LatigoBreakStrategy.cs and every dial
    # there exists here. That is not tidiness, it is the only way a PropSim run
    # and a Market Replay run are the same experiment -- an extra parameter on
    # this side silently makes them two. The three NT8 properties with no
    # counterpart are the ones with nothing to simulate: "Account-wide (all
    # markets)" governs OTHER instances on other instruments, "Show drawings" is
    # chart furniture, and the "07. Flow gate" group is the v4 filter left out of
    # scope by delta 5.
    params = {
        # Windows are DECISIONS, not dials -- flags, so the sweep leaves them be.
        "trade_globex_reopen": Param(1, 0, 1, "hunt the 18:00 ET Globex reopen"),
        "trade_evening": Param(1, 0, 1, "hunt the 20:00 ET window"),
        "trade_us_open": Param(1, 0, 1, "hunt the 09:30 ET US open"),
        "entry_window_minutes": Param(30, 1, 480, "hunt breaks/entries only this "
                                                  "long after a window opens",
                                      fixed=True),
        "use_whipsaw_filter": Param(1, 0, 1, "0 = naive chase at the break print"),
        "hold_seconds": Param(30, 0, 300, "seconds the break must survive outside"),
        "extension_r30": Param(0.25, 0, 3, "excursion beyond the level required to "
                                           "confirm, as a fraction of the range"),
        "candle_seconds": Param(30, 5, 300, "opening candle, seconds"),
        "min_r30_ticks": Param(4, 0, 100, "skip the window if the opening range is "
                                          "narrower, ticks"),
        "contracts": Param(1, 1, 100, "position size, contracts", fixed=True),
        "atr_period": Param(14, 2, 100, "bars in the ATR"),
        "atr_stop_mult": Param(2.0, 0.25, 10, "stop distance, in ATRs"),
        "atr_target_mult": Param(2.0, 0.25, 10, "target distance, in ATRs"),
        "max_trades_per_window": Param(1, 1, 10, "re-arm this many times per window"),
        "use_breakeven": Param(0, 0, 1, "move the stop to the entry once the run is "
                                        "covered"),
        # Kept out of the DEFAULT grid because it is inert while `use_breakeven`
        # is off: sweeping it there would multiply every combination by four and
        # charge the ledger for cells that cannot differ. Pass a range for it
        # explicitly once breakeven is on.
        "breakeven_percent": Param(50, 1, 99, "percent of the entry->target run that "
                                              "arms breakeven", fixed=True),
        "breakeven_offset_ticks": Param(0, 0, 100, "ticks past the entry the "
                                                   "breakeven stop sits", fixed=True),
        # Zero = off, as in NT8. Both are DOLLAR thresholds and therefore depend
        # on `contracts`: one four-lot LatigoBreak trade is worth $3-4k here, so
        # a $3,000 target is in practice "one trade a day" and a $2,000 loss
        # limit is narrower than the stop it is supposed to bound.
        # 500/300 because that is NT8's SetDefaults, not because they are good
        # numbers -- at one contract they stop most days after a single trade.
        # Defaulting them to 0 would have been the tidier-looking choice and the
        # wrong one: it hands anyone who does not touch the panel a strategy that
        # is not the one running in NinjaTrader, which is the entire class of bug
        # this parameter list was closed to prevent. 0 disables either side.
        "daily_profit_target": Param(500, 0, 100_000, "flatten and stop for the day "
                                                      "at this profit, USD; 0 = off",
                                     fixed=True),
        "daily_loss_limit": Param(300, 0, 100_000, "flatten and stop for the day at "
                                                   "this loss, USD; 0 = off",
                                  fixed=True),
    }

    def risk_ticks(self, p) -> float:
        # Not a parameter any more: the widest stop the guard below will pass.
        return self._SANITY_STOP_TICKS

    def entries(self, bars, tape, p):
        ts, px = tape["ts"], tape["px"]
        if len(ts) < 2 or len(bars["t"]) < 3:
            return _empty()
        enabled = [bool(p[f]) for f in self._FLAGS]
        if not any(enabled):
            return _empty()

        day = tp.day_index(bars["t"])
        atr = _atr_wilder(bars, int(p["atr_period"]), day)
        bstart = bars["start"]
        half = TICK_SIZE * 0.5
        filt = bool(p["use_whipsaw_filter"])
        hold_ticks = int(p["hold_seconds"] * tp.TPS)
        max_stop = self._SANITY_STOP_TICKS * TICK_SIZE
        max_trades = int(p["max_trades_per_window"])
        be_frac = p["breakeven_percent"] / 100.0 if p["use_breakeven"] else 0.0
        candle_s = float(p["candle_seconds"])
        # Floored so the candle can never starve the hunt, and capped below at
        # the next enabled window's open so one window cannot swallow the next.
        want = max(p["entry_window_minutes"] * 60.0, candle_s + 60.0)

        et, dr, st, tg, be = [], [], [], [], []
        for d in np.unique(day):
            base = (int(d) * 86400 + tp.NET_EPOCH_S + self._SESSION_BEGIN)
            for wi in range(3):
                if not enabled[wi]:
                    continue
                deadline = want
                for j in range(wi + 1, 3):
                    if enabled[j]:
                        deadline = min(deadline, self._OFFSETS[j] - self._OFFSETS[wi])
                        break
                w0 = (base + self._OFFSETS[wi]) * tp.TPS
                a = int(np.searchsorted(ts, w0))
                cend = int(np.searchsorted(ts, w0 + int(candle_s * tp.TPS)))
                b = int(np.searchsorted(ts, w0 + int(deadline * tp.TPS), "right"))
                if cend <= a or b <= cend:
                    continue                    # empty candle, or no hunt left
                h = float(px[a:cend].max())
                l = float(px[a:cend].min())
                r30 = int(round((h - l) / TICK_SIZE))
                if r30 < p["min_r30_ticks"]:
                    continue                    # degenerate candle, window skipped
                # ceil, as in NT8: a fractional tick of extension is a whole tick
                need_ext = int(np.ceil(p["extension_r30"] * r30))

                seg = px[cend:b].astype(np.float64)
                tseg = ts[cend:b]
                up = seg >= h + half
                dn = seg <= l - half
                inside = ~up & ~dn

                k, armed_inside, taken = 0, True, 0
                while k < len(seg) and taken < max_trades:
                    # A break needs an inside->outside transition. After a
                    # re-arm the state machine is not inside yet, so find that
                    # first -- otherwise the tick after an entry would re-break
                    # the level it is already through.
                    if not armed_inside:
                        ins = np.flatnonzero(inside[k:])
                        if not len(ins):
                            break
                        k += int(ins[0])
                        armed_inside = True
                    out = np.flatnonzero(up[k:] | dn[k:])
                    if not len(out):
                        break
                    i = k + int(out[0])
                    side = 1 if up[i] else -1

                    while True:                 # one break episode
                        level = h if side > 0 else l
                        # Whipsaw: the first print back inside by a full tick.
                        rei = (np.flatnonzero(seg[i + 1:] <= h - half) if side > 0
                               else np.flatnonzero(seg[i + 1:] >= l + half))
                        j_w = i + 1 + int(rei[0]) if len(rei) else len(seg)
                        if not filt:
                            m = i                       # naive chase at the break
                        else:
                            # Both conditions are monotone once true, so the
                            # confirmation is simply the later of the two: the
                            # tick that completes the hold, and the tick whose
                            # excursion first reaches the required extension.
                            i_t = int(np.searchsorted(tseg, tseg[i] + hold_ticks))
                            if need_ext <= 0:
                                i_e = i
                            else:
                                thr = level + side * (need_ext - 0.5) * TICK_SIZE
                                hit = (np.flatnonzero(seg[i:] >= thr) if side > 0
                                       else np.flatnonzero(seg[i:] <= thr))
                                i_e = i + int(hit[0]) if len(hit) else len(seg)
                            m = max(i_t, i_e)
                        # A tie goes to the whipsaw: NT8 tests the re-entry first
                        # on the tick, and only then the confirmation.
                        if m < j_w and m < len(seg):
                            e = cend + m
                            if self._emit(e, side, seg[m], atr, bars, px, p,
                                          max_stop, be_frac, et, dr, st, tg, be):
                                taken += 1
                            k, armed_inside = m + 1, False
                            break
                        if j_w >= len(seg):
                            k = len(seg)
                            break               # window ran out mid-episode
                        # Gap-through: the same print that vetoed this break is
                        # already past the opposite level, which opens it now.
                        if dn[j_w] if side > 0 else up[j_w]:
                            side, i = -side, j_w
                            continue
                        k, armed_inside = j_w + 1, True
                        break
        if not et:
            return _empty()
        return (np.array(et, np.int64), np.array(dr, np.int8),
                np.array(st), np.array(tg), None, np.array(be))

    def _emit(self, e, side, ref, atr, bars, px, p, max_stop, be_frac,
              et, dr, st, tg, be) -> bool:
        """Price one entry off the ATR NT8 HELD AT THAT TICK. False = skipped.

        NinjaTrader runs `Calculate.OnEachTick`, so `ComputeAtr()` reads
        `TrueRange(0)` of the bar STILL FORMING and folds it into Wilder's
        smoothing. Reading the previous closed bar instead was the single
        largest remaining difference against Market Replay: on the 2026-07-27
        18:01 trade NT8 priced its target off an ATR of 6.90 points where this
        read 6.35, which put the target 2.00 points nearer and cost $160 on four
        contracts -- and, because the same ATR sets the stop, quietly changed
        which trades stopped out at all.

        THE FORMING BAR IS BUILT ONLY FROM TICKS UP TO AND INCLUDING `e`, which
        is the whole point: it is what NT8 could see at that instant, not what
        the bar turned out to be. Taking the finished bar's high and low would
        be lookahead wearing the same numbers.
        """
        bstart = bars["start"]
        bi = int(np.searchsorted(bstart, e, "right")) - 1
        if bi < 1:
            return False
        prev = atr[bi - 1]              # Wilder's value at the last CLOSED bar
        if not np.isfinite(prev) or prev <= 0:
            return False               # ATR not formed; NT8's chart fallback has
        seg = px[int(bstart[bi]):e + 1]                  # no meaning on a full tape
        if not len(seg):
            return False
        h, l = float(seg.max()), float(seg.min())
        pc = float(bars["c"][bi - 1])   # previous close, as NinjaTrader's TrueRange
        tr = max(h - l, abs(h - pc), abs(l - pc))
        a = prev + (tr - prev) / int(p["atr_period"])
        if not np.isfinite(a) or a <= TICK_SIZE:
            return False
        risk = p["atr_stop_mult"] * float(a)
        if risk > max_stop:
            return False               # a spike-wide stop is not this setup
        reward = p["atr_target_mult"] * float(a)
        ref = float(ref)
        et.append(e)
        dr.append(side)
        st.append(_round_tick(ref - side * risk))
        tg.append(_round_tick(ref + side * reward))
        be.append(ref + side * be_frac * reward if be_frac > 0 else ref)
        return True


def _round_tick(px: float) -> float:
    return round(px / TICK_SIZE) * TICK_SIZE


LIBRARY = {s.name: s for s in (MACross, ORB, RangeBreak, SweepFollow, FVG,
                               VWAPRevert, LatigoBreak)}


def _empty():
    return (np.array([], np.int64), np.array([], np.int8),
            np.array([]), np.array([]))


def _atr(bars, n, day):
    """Average true range over `n` closed bars, aligned on the last one.

    A simple mean rather than Wilder's smoothing: the difference is a lag
    constant, and this one can be read off the chart you are looking at.

    THE TRUE RANGE RESETS AT THE SESSION BOUNDARY. Reaching for "the previous
    close" across the overnight break measures the gap, not the bar, and the
    first bar of the morning would carry the whole of last night into the stop
    it sizes.
    """
    h, l, c = bars["h"], bars["l"], bars["c"]
    prev = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    new_day = np.empty(len(c), bool)
    new_day[0] = True
    new_day[1:] = day[1:] != day[:-1]
    tr[new_day] = (h - l)[new_day]
    return _roll(tr, n, np.mean)


def _atr_wilder(bars, n, day):
    """Wilder's ATR over CLOSED bars, aligned on the last one.

    NinjaTrader's recursion exactly, seeding included: a plain mean of the true
    ranges seen so far until `n` bars exist, then
    `atr += (tr - atr) / n` (`LatigoBreakStrategy.cs:1009-1021`). The seed
    matters more than it looks -- Wilder's smoothing never forgets its first
    value, so seeding it differently is a permanent offset, not a warm-up.

    Wilder against the simple mean is worth about 1% here and it is not why the
    two systems disagreed; the forming bar, folded in by the caller, is. This
    exists so that the recursion the caller extends is the one NT8 extends.

    The session reset of `_atr` is deliberately NOT copied: NinjaTrader's
    `TrueRange` reaches across the break, and for this strategy the two never
    differ anyway -- `day` cuts at midnight, which is nine hours from the 09:30
    window and in the future for the 18:00 and 20:00 ones, so the boundary never
    lands inside the 15 bars any entry averages.
    """
    h, l, c = bars["h"], bars["l"], bars["c"]
    prev = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    tr[0] = h[0] - l[0]                 # no previous close to reach for
    out = np.empty(len(tr))
    run = 0.0
    for i in range(len(tr)):
        if i < n:
            run = (run * i + tr[i]) / (i + 1)
        else:
            run += (tr[i] - run) / n
        out[i] = run
    return out


def _roll(x, n, fn):
    """`fn` over every window of `n`, aligned on the window's LAST bar."""
    out = np.full(len(x), np.nan)
    out[n - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, n), axis=1)
    return out


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
            max_gap_s=MAX_GAP_S, day=None, be_trigger=None, contracts=1,
            be_offset_ticks=0.0, day_target=0.0, day_loss=0.0,
            gov_day=None) -> list[Trade]:
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

    `be_trigger` is a price per trade: the first tick to reach it moves the stop
    to the entry fill, once, and an exit there is reported as `be` rather than
    `stop`. The engine holds the mechanism and the strategy chooses the price,
    the same split as `limit_px` -- and the separate reason matters because a
    stop-out ends at the worst price the trade ever saw while a breakeven exit
    does not, so several of the selfcheck's tightest invariants are true of one
    and false of the other.

    A breakeven exit still LOSES: it gives back the exit slippage and the
    commission. Modelling it as flat would be the same kind of free money the
    rest of this function exists to refuse. `be_offset_ticks` moves the stop to
    the entry plus that many ticks IN THE TRADE'S FAVOUR, as NinjaTrader does;
    at 0 it is the entry fill and nothing changes.

    `contracts` scales every currency figure -- P&L, both excursions, the fall
    from the running peak, and the commission, which is charged per contract
    because a broker charges per contract. It is here rather than at the account
    layer for one reason: the daily governor below is a DOLLAR threshold, and
    $3,000 on four contracts stops a trade that $3,000 on one would let run. A
    strategy that sizes itself must therefore be simulated at its own size.

    `day_target` / `day_loss` are that governor, and they are modelled the way
    NinjaTrader's is: tested on EVERY TICK against realized plus UNREALIZED P&L,
    flattening the open trade where it breaches and refusing every later entry
    that day. Testing it only on closed trades would be a different, kinder
    strategy -- the real one exits mid-trade, which is why a $3,000 target books
    $4,245 and a $2,000 limit books -$3,514. Days are counted on `gov_day`, the
    session-boundary day, not the calendar day.
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
    nc = float(contracts)
    governed = day_target > 0 or day_loss > 0
    if governed and gov_day is None:
        gov_day = day
    gov_d, day_pnl, locked = None, 0.0, False

    for k in range(len(entry_idx)):
        i0 = int(entry_idx[k])
        if i0 >= n or ts[i0] < free_at:
            continue
        if governed:
            # A NEW SESSION CLEARS BOTH THE RUNNING TOTAL AND THE LOCKOUT, which
            # is the whole of NinjaTrader's `ResetSession`. Reading it off the
            # entry tick means a day with no entries never needs a reset at all.
            gd = int(gov_day[i0])
            if gd != gov_d:
                gov_d, day_pnl, locked = gd, 0.0, False
            if locked:
                continue           # locked out: NT8 does not arm the next window
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
        be = None if be_trigger is None else float(be_trigger[k])
        if be is not None and (be <= fill if d > 0 else be >= fill):
            be = None               # a trigger the entry is already past is not one
        moved = False
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
            # Move the stop BEFORE the exit tests. On the tick that reaches the
            # trigger the old stop is far behind and cannot fire, and the target
            # -- if this same tick reached it too -- still wins below.
            if be is not None and (p >= be if d > 0 else p <= be):
                st, be, moved = fill + d * be_offset_ticks * costs.tick_size, None, True
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
                exit_i, why = i, ("be" if moved else "stop"); break
            if hit_targ:
                exit_i, exit_px, why = i, tg, "target"; break
            # THE GOVERNOR IS TESTED LAST ON THE TICK, so a tick that also
            # breached a bracket books the bracket. The brackets are resting
            # orders sitting at the exchange; the governor is a market order
            # this program sends after seeing the print, and it cannot beat an
            # order that was already there.
            if governed:
                total = day_pnl + (p - fill) * d * costs.point_value * nc
                if ((day_target > 0 and total >= day_target)
                        or (day_loss > 0 and total <= -day_loss)):
                    exit_i, exit_px, why = i, p, "governor"; break
        if exit_i is None:
            exit_i = min(stop_end, n) - 1
            exit_px = float(px[exit_i])
        exit_px -= d * slip                       # and again on exit
        gross = (exit_px - fill) * d * costs.point_value * nc
        adverse = (lo if d > 0 else hi)
        favorable = (hi if d > 0 else lo)
        pnl = gross - costs.commission * nc       # a broker bills per contract
        out.append(Trade(
            entry_time=tp.to_datetime(ts[i0]), exit_time=tp.to_datetime(ts[exit_i]),
            direction=d, entry_price=fill, exit_price=exit_px,
            pnl=pnl,
            mae=min((adverse - fill) * d * costs.point_value * nc, 0.0),
            mfe=max((favorable - fill) * d * costs.point_value * nc, 0.0),
            reason=why, intra_mdd=mdd * costs.point_value * nc))
        free_at = ts[exit_i] + cool
        if governed:
            day_pnl += pnl
            # Either the governor fired inside the trade, or a bracket exit left
            # the day past a threshold -- NinjaTrader's next tick sees the same
            # total and locks out just the same. Both end the trading day.
            locked = (why == "governor"
                      or (day_target > 0 and day_pnl >= day_target)
                      or (day_loss > 0 and day_pnl <= -day_loss))
    return out


def prepare(contract, tf_secs=300, start=None, end=None, rth_only=True,
            strategy=None) -> dict:
    """Load, slice and bar the tape once, for reuse across many runs.

    Measured on 28 days of NQ: loading the cache, slicing to RTH and building
    bars costs 0.66 s, while the strategy and the fill loop cost 0.15 s. A sweep
    that calls `backtest` naively therefore spends 80% of its time re-reading
    data that did not change -- hoisting this out makes a 200-combination sweep
    roughly five times faster.
    """
    # The strategy decides the session, not the caller: an overnight setup on an
    # RTH tape returns no trades and no error, and every screen would then report
    # a flat, healthy-looking nothing.
    if strategy is not None and LIBRARY[strategy].full_session:
        rth_only = False
    # Narrow at load time (Task 5's ranged `load_cache`) so the full tape is
    # never held alive once the request only wants a slice of it -- and drop
    # the narrowed-but-still-whole-tape reference the moment slice_range has
    # produced its own copy, so RTH-only trims (the common case) actually free
    # the overnight ticks instead of keeping them pinned for the rest of prepare().
    raw = tp.load_cache(contract, start, end)
    t = tp.slice_range(raw, start, end, rth_only=rth_only)
    del raw
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
    return dict(contract=contract, tape=t, bars=tp.build_bars(t, tf_secs),
                tf_secs=tf_secs, rth_only=rth_only, days=len(days),
                start=tp.date_str(days[0]), end=tp.date_str(days[-1]),
                dayi=dayi, n_holes=len(holes),
                hole_days=sorted({tp.date_str(int(dayi[i])) for i in holes}))


def backtest(contract, strategy_name, tf_secs=300, start=None, end=None,
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
        ctx = prepare(contract, tf_secs, start, end, rth_only, strategy_name)
    elif ctx["tf_secs"] != tf_secs:
        raise SystemExit(f"prepared tape is {tp.tf_label(ctx['tf_secs'])}, asked for "
                         f"{tp.tf_label(tf_secs)}")
    elif strat.full_session and ctx["rth_only"]:
        raise SystemExit(f"{strategy_name} trades outside RTH — prepare the tape "
                         f"with strategy={strategy_name!r} (or rth_only=False)")
    rth_only = ctx["rth_only"]              # the tape that ran, not the request
    t, bars = ctx["tape"], ctx["bars"]

    res = strat.entries(bars, t, p)
    ei, dr, st, tg = res[:4]
    # A retrace strategy rests a limit at its level; a breakout crosses the
    # spread. A strategy that trades on a limit says so by returning a fifth
    # array, so the fill model is a property of the setup rather than an accident
    # of who wrote the entry function.
    lim = res[4] if len(res) > 4 else None
    # And a sixth for a breakeven trigger, on the same principle: the engine owns
    # the mechanism, the strategy owns the price, and a setup that wants neither
    # returns four arrays and is unaffected.
    bet = res[5] if len(res) > 5 else None
    cd = p.get("cooldown_min", 0.0)
    # `.get` with the neutral default throughout: a strategy that declares none
    # of these is resolved exactly as it was before they existed.
    tgt = float(p.get("daily_profit_target", 0.0))
    dl = float(p.get("daily_loss_limit", 0.0))
    gd = None
    if (tgt > 0 or dl > 0) and strat.session_offset_s:
        # The governor's day, not the calendar's. Shifting the tape back by the
        # session offset and reusing `day_index` keeps one definition of a day.
        gd = tp.day_index(t["ts"] - strat.session_offset_s * tp.TPS)
    trades = resolve(t, ei, dr, st, tg, costs, cooldown_min=cd, limit_px=lim,
                     day=ctx["dayi"], be_trigger=bet,
                     contracts=p.get("contracts", 1),
                     be_offset_ticks=p.get("breakeven_offset_ticks", 0.0),
                     day_target=tgt, day_loss=dl, gov_day=gd)

    # Data quality, reported rather than assumed: a silence longer than MAX_GAP_S
    # inside a session means hourly files are missing, and every bar spanning one
    # is fiction. The user needs to see that before believing a P&L built on it.
    # Measured once per tape in `prepare`, not once per parameter set.
    meta = dict(contract=contract, strategy=strategy_name, label=strat.label,
                timeframe=tp.tf_label(tf_secs), tf_secs=tf_secs, start=ctx["start"],
                end=ctx["end"], days=ctx["days"], rth_only=rth_only,
                params=p, bars=len(bars["t"]), ticks=len(t["ts"]),
                signals=len(ei), trades=len(trades),
                data_holes=ctx["n_holes"], hole_days=ctx["hole_days"],
                gap_exits=sum(1 for x in trades if x.reason == "gap"),
                # THE EFFECTIVE CHARGE PER TRADE, not the per-contract rate. The
                # report reconstructs gross by adding this back to a net that was
                # already booked at size, so a four-lot run reporting the one-lot
                # rate understates both its gross and its commission while still
                # balancing -- self-consistent and wrong, the worst kind.
                commission=costs.commission * float(p.get("contracts", 1)),
                slippage_ticks=costs.slippage_ticks)

    # THE LEDGER MUST SHOW THE PRICE THAT ACTUALLY TRADED. The ALL tape is
    # back-adjusted so indicators do not see a multi-hundred-point step at each
    # roll, but a user checks these fills against NinjaTrader, and an entry
    # printed hundreds of points below what the screen said that day reads as
    # a bug. Only entry_price/exit_price are mapped -- pnl/mae/mfe/intra_mdd
    # are DIFFERENCES and a constant offset cancels in a subtraction; mapping
    # them too would corrupt them. Imported here rather than at module scope:
    # `continuous` imports `tape`, and engine importing it at the top either
    # way makes that circular (see tape.py's own local imports of it).
    if contract == "ALL" and trades:
        import continuous as _c
        cmeta = _c.load_meta()
        if cmeta and cmeta.get("roll_ts"):
            entry_ts = np.array([tp.to_net(t.entry_time) for t in trades], np.int64)
            exit_ts = np.array([tp.to_net(t.exit_time) for t in trades], np.int64)
            entry_px = np.array([t.entry_price for t in trades], np.float64)
            exit_px = np.array([t.exit_price for t in trades], np.float64)
            raw_entry = _c.to_raw(entry_px, entry_ts, cmeta)
            raw_exit = _c.to_raw(exit_px, exit_ts, cmeta)
            for t, ep, xp in zip(trades, raw_entry, raw_exit):
                t.entry_price, t.exit_price = float(ep), float(xp)

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
    c = tp.sample_contract()
    if c is None:
        print("no tape cached — run: python3 tape.py --build 'NQ 09-26'")
        return

    # 1. LOOKAHEAD. ORB's opening range is a TIME window, so its high/low is the
    #    same whether drawn as 1m or 15m bars, and the entry is the tick that
    #    crosses it. Identical results across timeframes is therefore the
    #    CORRECT behaviour -- and the first implementation, which entered at the
    #    bar's first tick using end-of-bar information, scored t=4.78 instead.
    #    The 30-second bar is in the list for a second reason: sub-minute sizes go
    #    through the same code, and a run that silently re-barred to something else
    #    would break this invariant rather than pass it quietly.
    res = {tf: summarise(*backtest(c, "orb", tf)) for tf in (30, 60, 300, 900)}
    pnls = {tf: round(r["pnl"], 2) for tf, r in res.items()}
    assert len(set(pnls.values())) == 1, f"ORB must be timeframe-invariant: {pnls}"

    # 2. A bar-counting strategy must NOT be timeframe-invariant, or bars are
    #    not really being rebuilt. 30s vs 2m spans both units.
    ma = {tf: summarise(*backtest(c, "ma_cross", tf)) for tf in (30, 120, 900)}
    assert ma[30]["trades"] != ma[900]["trades"] and ma[30]["trades"] != ma[120]["trades"], \
        f"MA cross should differ by timeframe: {[m['trades'] for m in ma.values()]}"

    # 3. Pessimism must be monotonic: more slippage can never help.
    a = summarise(*backtest(c, "orb", 300, costs=Costs(slippage_ticks=0)))
    b = summarise(*backtest(c, "orb", 300, costs=Costs(slippage_ticks=5)))
    assert b["pnl"] < a["pnl"], f"slippage did not hurt: {a['pnl']} -> {b['pnl']}"

    # 4. A date range must actually restrict the data. The cut is derived from what
    #    this contract actually holds: a hardcoded date silently stopped restricting
    #    anything the day a contract whose tape ends earlier became `contracts[0]`,
    #    and the test failed for the data rather than for the code.
    lo, hi, _ = tp.available_range(c)
    cut = tp.date_str(np.unique(tp.day_index(tp.load_cache(c)["ts"]))[-2])
    full = summarise(*backtest(c, "orb", 300))
    half = summarise(*backtest(c, "orb", 300, start=None, end=cut))
    assert half["days"] < full["days"], (
        f"date range had no effect: {c} spans {lo}..{hi}, cut at {cut}")

    # 5. Every trade must resolve to a real exit reason.
    tr, _ = backtest(c, "orb", 300)
    assert all(t.reason in ("stop", "be", "target", "timeout", "close", "gap")
               for t in tr)
    assert all(t.mae <= 0 <= t.mfe for t in tr), "MAE/MFE signs"

    # 6. THE INVARIANT THAT CAUGHT A REAL BUG. A stop-out cannot make money and a
    #    target cannot lose it. Both were violated when a strategy measured its
    #    stop from a signal LEVEL while the engine filled at a later tick: one
    #    stop-out booked +$17,635 and the strategy showed +$34K over 23 days.
    #    Checked across every strategy, because the mistake is not local to one.
    costs = Costs()
    slip_cost = costs.slippage_ticks * costs.tick_size * costs.point_value
    for name in LIBRARY:
        trades, meta = backtest(c, name, 300, costs=costs)
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
            elif t.reason == "be":
                # A BREAKEVEN EXIT IS NOT FLAT. The stop moved to the entry
                # fill, so the exit gives back the exit slippage and the
                # commission and can only be worse than that -- a stop is a
                # market order and may still slip through its level. Anything
                # at or above zero here means breakeven was modelled as free.
                assert t.pnl <= -(slip_cost + costs.commission) + 0.01, (
                    f"{name}: breakeven exit booked {t.pnl:+.2f}, at best it "
                    f"costs {-(slip_cost + costs.commission):.2f}")
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
        nominal = (LIBRARY[name]().risk_ticks(meta["params"])
                   * costs.tick_size * costs.point_value)
        bound = -(nominal + 100 * costs.tick_size * costs.point_value)
        for t in trades:
            if t.reason == "stop":
                assert t.pnl >= bound, (
                    f"{name}: stop lost {t.pnl:,.0f}, bound {bound:,.0f} "
                    f"({meta['data_holes']} data holes in range)")

    # 9. THE TIME WINDOW IS A CONSTRAINT, NOT A LABEL. It is tested on the entry
    #    tick rather than on the signal bar, so the check is on entry times: a
    #    window read off the bar would let a 14:59 bar enter at 15:00:20.
    rb, rbm = backtest(c, "range_break", 60, params={"session_start": 600,
                                                    "last_entry": 720})
    for t in rb:
        s = t.entry_time.hour * 60 + t.entry_time.minute
        assert 600 <= s < 720, f"range_break entered at {t.entry_time}"
    wide = summarise(*backtest(c, "range_break", 60))
    assert wide["signals"] >= len(rb), "narrowing the window added signals"
    # 9b. BREAKEVEN MUST DO SOMETHING, AND ONLY WHAT IT SAYS. Arming it can only
    #     turn winners-that-came-back into small losses and losers into smaller
    #     ones, so it must produce `be` exits that the same run without it does
    #     not have -- and it must never touch a trade that went straight to its
    #     target, which is the mistake a trigger tested on the wrong side makes.
    off, om = backtest(c, "range_break", 60, params={"be_trigger_pct": 0})
    on, _ = backtest(c, "range_break", 60, params={"be_trigger_pct": 50})
    assert not any(t.reason == "be" for t in off), "be_trigger_pct=0 still moved a stop"
    assert any(t.reason == "be" for t in on), "breakeven never triggered"
    assert (sum(1 for t in on if t.reason == "target")
            <= sum(1 for t in off if t.reason == "target")), (
        "breakeven created target exits, which it cannot do")
    # And a consolidation is defined by its width: nothing wider than
    # max_range_ticks may arm. Checked by squeezing the ceiling to the floor,
    # which must leave strictly fewer signals than the default band.
    tight = summarise(*backtest(c, "range_break", 60,
                                params={"min_range_ticks": 1, "max_range_ticks": 20}))
    assert tight["signals"] < wide["signals"], (
        f"range width had no effect: {tight['signals']} vs {wide['signals']}")

    # 10. THE WINDOWS ARE THE STRATEGY. LatigoBreak's three windows are offsets
    #     from the 18:00 ET session begin, and two of them live outside RTH -- so
    #     the two ways this port can be silently wrong are an RTH tape (finds
    #     nothing, reads as "no edge") and an offset that drifts (trades the
    #     wrong minutes and reads as an edge). Both are checked on entry TIMES.
    lb, lbm = backtest(c, "latigo_break", 300)
    assert not lbm["rth_only"], "latigo_break was handed an RTH tape"
    win_s = [(18 * 3600 + o) % 86400 for o in LatigoBreak._OFFSETS]
    span = max(30 * 60, 30 + 60)                  # default window, floored
    for t in lb:
        s = (t.entry_time.hour * 3600 + t.entry_time.minute * 60
             + t.entry_time.second)
        assert any((s - w0) % 86400 <= span for w0 in win_s), (
            f"latigo_break entered at {t.entry_time}, outside every window")
    # A flag that does nothing is the classic silent bug, and so is a degenerate-
    # candle gate that never gates.
    off = summarise(*backtest(c, "latigo_break", 300, params={
        "trade_globex_reopen": 0, "trade_evening": 0, "trade_us_open": 0}))
    assert off["trades"] == 0, f"windows all off still traded {off['trades']} times"
    # The degenerate-candle gate has to BITE, and "a 400-tick threshold silences
    # it" is not the way to say so: this tape really does hold 30-second opening
    # ranges of 400-729 ticks (2026-06-14 and 2026-07-26 at the reopen, 2026-06-12
    # at 09:30). Asserting zero there passed by luck and would have failed the day
    # a news reopen landed in the range. Monotonic instead, plus one threshold no
    # opening candle can reach.
    tight = summarise(*backtest(c, "latigo_break", 300, params={"min_r30_ticks": 100}))
    assert tight["signals"] < lbm["signals"], "min_r30_ticks never skipped a window"
    none = summarise(*backtest(c, "latigo_break", 300,
                               params={"min_r30_ticks": 100_000}))
    assert none["signals"] == 0, "an unreachable opening range still traded"
    # 10b. THE CONFIRMATION MUST DEGENERATE TO THE NAIVE CHASE. With no hold and
    #      no extension required, NT8's `ConfirmReady` is true on the break print
    #      itself, so the filtered path must produce the SAME entries as the
    #      unfiltered one, tick for tick. This is the check that catches an
    #      off-by-one in the confirmation index -- the filter is otherwise free to
    #      be quietly wrong, because a different entry still looks like a trade.
    naive, _ = backtest(c, "latigo_break", 300, params={"use_whipsaw_filter": 0})
    degen, _ = backtest(c, "latigo_break", 300, params={"hold_seconds": 0,
                                                      "extension_r30": 0})
    assert ([t.entry_time for t in naive] == [t.entry_time for t in degen]), (
        "hold=0, extension=0 did not reproduce the naive chase")
    assert not set(t.entry_time for t in naive) & set(t.entry_time for t in lb), (
        "the whipsaw filter changed nothing: same entries as the naive chase")
    # 10c. SIZE IS EXACTLY LINEAR WHILE NOTHING GOVERNS IT. Four contracts must
    #      be four times one contract, trade for trade, commission included --
    #      any drift means a currency figure was left un-scaled, and a partly
    #      scaled trade is worse than an unscaled one because it looks right.
    #      Governed OFF here on purpose: linearity is a claim about the fill
    #      model, and a dollar threshold is exactly what makes size non-linear.
    ungov = {"daily_profit_target": 0, "daily_loss_limit": 0}
    one, _ = backtest(c, "latigo_break", 300, params={**ungov, "contracts": 1})
    four, _ = backtest(c, "latigo_break", 300, params={**ungov, "contracts": 4})
    assert len(one) == len(four), "size changed the trade set with no governor on"
    for a4, a1 in zip(four, one):
        for f in ("pnl", "mae", "mfe", "intra_mdd"):
            assert abs(getattr(a4, f) - 4 * getattr(a1, f)) < 0.01, (
                f"contracts did not scale {f}: {getattr(a4, f):.2f} vs "
                f"4 x {getattr(a1, f):.2f}")
    # 10d. THE DAILY GOVERNOR ENDS THE DAY, and it is checked on the running
    #      total rather than on the exit reason: a lockout that leaks lets a
    #      later window trade a day NinjaTrader had already closed, which is the
    #      failure that makes a PropSim day look better than a Replay day.
    #      Thresholds sized off this tape so both sides actually fire.
    gov, gm = backtest(c, "latigo_break", 300, params={
        "contracts": 4, "daily_profit_target": 3000, "daily_loss_limit": 2000})
    assert any(t.reason == "governor" for t in gov), "the governor never fired"
    day_of, run = {}, {}
    for t in gov:
        # Same definition the engine used: the day starts at the 18:00 reopen.
        k = (t.entry_time - __import__("datetime").timedelta(
            seconds=LatigoBreak.session_offset_s)).date()
        assert not day_of.get(k), (
            f"traded {t.entry_time} after {k} was already locked out")
        run[k] = run.get(k, 0.0) + t.pnl
        if run[k] >= 3000 or run[k] <= -2000:
            day_of[k] = True
    assert len(gov) < len(four), "the governor removed no trades at all"
    # 10e. A BREAKEVEN OFFSET IS IN THE TRADE'S FAVOUR. Moving the stop past the
    #      entry can only improve a `be` exit; if it does not, the offset is
    #      being applied on the wrong side, which silently costs money.
    be0 = [t for t in backtest(c, "latigo_break", 300, params={
        **ungov, "use_breakeven": 1, "breakeven_offset_ticks": 0})[0]
        if t.reason == "be"]
    be5 = [t for t in backtest(c, "latigo_break", 300, params={
        **ungov, "use_breakeven": 1, "breakeven_offset_ticks": 5})[0]
        if t.reason == "be"]
    assert be0, "breakeven never triggered on latigo_break"
    assert sum(t.pnl for t in be5) > sum(t.pnl for t in be0), (
        "a breakeven offset did not help: applied on the wrong side")
    # 10f. THE BAR IN FORMATION MUST NOT SEE ITS OWN FUTURE. Folding the forming
    #      bar into the ATR is what matches NinjaTrader, and it is also the one
    #      change here that could quietly become lookahead -- reading the bar's
    #      finished high and low instead of the part that had printed. So the
    #      test is the definition: cut the tape one tick after an entry, so the
    #      rest of that bar does not exist, and demand the SAME stop and target.
    #      An off-by-one in the slice or a previous close read off the wrong bar
    #      moves them; nothing else does.
    lbs = LatigoBreak()
    pl = {k: v.default for k, v in lbs.params.items()}
    lctx = prepare(c, 300, strategy="latigo_break")
    ei, _, lst, ltg = lbs.entries(lctx["bars"], lctx["tape"], pl)[:4]
    assert len(ei) > 2, "latigo_break emitted nothing to test for lookahead"
    k = len(ei) // 2
    e = int(ei[k])
    cut = {kk: vv[:e + 1] for kk, vv in lctx["tape"].items()}
    ei2, _, lst2, ltg2 = lbs.entries(tp.build_bars(cut, 300), cut, pl)[:4]
    hit = np.flatnonzero(ei2 == e)
    assert len(hit), f"the entry at tick {e} vanished when its bar was truncated"
    j = int(hit[0])
    assert abs(lst2[j] - lst[k]) < 1e-9 and abs(ltg2[j] - ltg[k]) < 1e-9, (
        f"ATR read the future: stop/target {lst[k]:.2f}/{ltg[k]:.2f} on the full "
        f"tape, {lst2[j]:.2f}/{ltg2[j]:.2f} with the bar cut at the entry")

    print(f"selfcheck OK: ORB timeframe-invariant (${pnls[300]:,.0f} at 30s/1m/5m/15m); "
          f"MA cross varies ({ma[30]['trades']} vs {ma[900]['trades']} trades); "
          f"slippage monotonic (${a['pnl']:,.0f} -> ${b['pnl']:,.0f}); "
          f"date range {full['days']} -> {half['days']} days")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--strategy"); ap.add_argument("--contract")
    ap.add_argument("--tf", default="5m", help="bar size: 30s, 2m, 15m (bare "
                                               "numbers are minutes)")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--slippage", type=float, default=2.0)
    ap.add_argument("--sweep-tf", action="store_true",
                    help="run 30s/1m/3m/5m/15m on identical ticks")
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
    tfs = [30, 60, 180, 300, 900] if args.sweep_tf else [tp.tf_secs(args.tf)]
    print(f"{'tf':>5}{'bars':>9}{'signals':>9}{'trades':>8}{'/day':>7}"
          f"{'P&L':>12}{'WR':>7}{'t(daily)':>10}")
    for tf in tfs:
        tr, meta = backtest(args.contract, args.strategy, tf,
                            args.start, args.end, costs=costs)
        s = summarise(tr, meta)
        print(f"{tp.tf_label(tf):>5}{s['bars']:>9,}{s['signals']:>9,}{s['trades']:>8,}"
              f"{s['per_day']:>7.1f}{s['pnl']:>12,.0f}{s['wr']:>7.1%}"
              f"{s['t_daily']:>10.2f}")
    print(f"\n{meta['contract']} {meta['start']}..{meta['end']} "
          f"({meta['days']} days), slippage {args.slippage} ticks/side")


if __name__ == "__main__":
    main()
