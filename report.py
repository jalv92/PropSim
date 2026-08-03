#!/usr/bin/env python3
"""The Analyzer report: one backtest, read two ways.

`grid` is NinjaTrader's summary table over PropSim's own trades. `prop` is the
same run seen through a prop firm's rules. `build` composes them.

NOTHING HERE RE-DERIVES A DRAWDOWN RULE. The prop-firm numbers are `sim.replay`
and `sim.sim_eval` verbatim; this module arranges them. This file has already
grown one bug from having two definitions of one rule (see `sim._day_draw`), and
a third definition would be worse because it would be the one on screen.

    python3 report.py --selfcheck
"""
from __future__ import annotations

import argparse

import numpy as np

import nttrades
import sim
from prop_rules import INTRA, NONE, RuleSet


def _column(trades, commission, days, contracts):
    """One column of the summary table for a subset of trades.

    `pnl` on a trade is ALREADY net of commission (engine.py books it as
    `gross - costs.commission`), so the gross figures are reconstructed by adding
    the commission back on. Winners and losers are partitioned on the NET number
    because that is the one the trader keeps, and the identity
    `gross_profit + gross_loss - commission == net_profit` holds for any
    partition that covers every trade.
    """
    n = len(trades)
    if not n:
        return dict(net_profit=0.0, gross_profit=0.0, gross_loss=0.0,
                    commission=0.0, profit_factor=None, max_drawdown=0.0,
                    trades=0, pct_profitable=None, wins=0, losses=0, evens=0,
                    avg_trade=None, avg_win=None, avg_loss=None,
                    win_loss_ratio=None, max_consec_wins=0, max_consec_losses=0,
                    largest_win=None, largest_loss=None, per_day=0.0,
                    avg_mae=None, avg_mfe=None, avg_fall=None, max_fall=None)

    k = float(contracts)
    net = np.array([t.pnl for t in trades]) * k
    comm = commission * n * k
    gross = net + commission * k                  # per trade, before commission
    wins = net > 0
    losses = net < 0

    run = peak = mdd = 0.0
    for v in net:                                  # equity curve of THIS subset
        run += float(v)
        peak = max(peak, run)
        mdd = min(mdd, run - peak)

    best = cur = 0
    worst = curl = 0
    for w, l in zip(wins, losses):
        cur = cur + 1 if w else 0
        curl = curl + 1 if l else 0
        best, worst = max(best, cur), max(worst, curl)

    def avg(a):
        return float(a.mean()) if len(a) else None

    return dict(
        net_profit=float(net.sum()),
        gross_profit=float(gross[wins].sum()),
        gross_loss=float(gross[~wins].sum()),
        commission=comm,
        profit_factor=(None if not len(net) else
                       (float("inf") if gross[~wins].sum() == 0 else
                        abs(float(gross[wins].sum() / gross[~wins].sum())))),
        max_drawdown=mdd,
        trades=n,
        pct_profitable=float(wins.mean()),
        wins=int(wins.sum()), losses=int(losses.sum()),
        evens=int(n - wins.sum() - losses.sum()),
        avg_trade=float(net.mean()),
        avg_win=avg(net[wins]), avg_loss=avg(net[losses]),
        win_loss_ratio=(None if not losses.any() or not wins.any() else
                        abs(float(net[wins].mean() / net[losses].mean()))),
        max_consec_wins=best, max_consec_losses=worst,
        largest_win=(float(net[wins].max()) if wins.any() else None),
        largest_loss=(float(net[losses].min()) if losses.any() else None),
        per_day=n / max(days, 1),
        avg_mae=float(np.mean([t.mae for t in trades]) * k),
        avg_mfe=float(np.mean([t.mfe for t in trades]) * k),
        # PropSim's own two rows: the fall from the running peak is what an
        # intraday trailing floor tests, and the maximum of it is the single
        # number that decides whether this strategy can be sized up at all.
        avg_fall=float(np.mean([t.intra_mdd for t in trades]) * k),
        max_fall=float(max(t.intra_mdd for t in trades) * k))


def grid(trades, meta, contracts=1):
    """NinjaTrader's summary table, split All / Long / Short.

    Rows are limited to what the trade list alone supports. Sharpe, Sortino,
    Ulcer index, R squared, Probability, ETD, bars in trade, time to recover and
    flat period are ABSENT rather than blank: they need machinery that does not
    exist here and none of them changes a decision these rows do not.
    """
    commission = float(meta.get("commission", 0.0))
    days = int(meta.get("days", 1))
    ordered = sorted(trades, key=lambda t: t.entry_time)
    return dict(
        all=_column(ordered, commission, days, contracts),
        long=_column([t for t in ordered if t.direction > 0],
                     commission, days, contracts),
        short=_column([t for t in ordered if t.direction < 0],
                      commission, days, contracts))


def prop(trades, rules, contracts=1, sims=20_000, rng=None):
    """The same run seen through one prop firm's rules.

    Two answers, deliberately kept apart. `run` is what happened to THIS
    sequence -- a fact. `projection` is the Monte Carlo over resampled futures --
    a projection. Presenting either as the other is how a sample of size one
    turns into a probability, or a probability into a promise.
    """
    ordered = sorted(trades, key=lambda t: t.entry_time)
    total = sum(t.pnl for t in ordered) * contracts

    rep = sim.replay(ordered, rules, contracts=contracts)
    after = 0.0
    if rep["outcome"] == "busted" and rep["date"] is not None:
        # Walk to the trade the replay named and sum from there. Deriving this
        # from the trade list rather than from `total - rep["profit"]` is the
        # point: the two only agree if the replay's indices are right.
        same_day = [t for t in ordered if t.date == rep["date"]]
        killer = same_day[rep["trade_of_day"] - 1]
        idx = ordered.index(killer)
        # WHY the account died decides whether the killing trade ever booked.
        # `sim.replay` breaches inside a trade BEFORE `bal += pnl` -- the trade
        # never finished, so its P&L is fiction. The close test and the daily
        # loss limit both fire AFTER it: that money really moved, and calling it
        # fiction tells the trader the opposite of what happened.
        if rep["reason"] != "drawdown floor, inside the trade":
            idx += 1
        after = sum(t.pnl for t in ordered[idx:]) * contracts

    pool = nttrades.to_pool(ordered)
    prof = dict(p=0.5, rr=1.0, risk=200.0, tpd=pool["width"],
                friction=0.0, pool=pool)
    mc = sim.sim_eval(prof, sim.policy_fixed(contracts), sims,
                      rng if rng is not None else np.random.default_rng(7),
                      rules)

    return dict(
        # A firm whose breach basis was never verified against a primary source
        # gets no verdict. Defaulting it to what the other four do would be
        # inventing a rule and printing it as a measurement.
        applicable=rules.breach_basis is not None,
        # MEASURED means every trade carried a tick-measured intra-trade path;
        # BOUNDED means at least one fell back to `mfe - mae`, its upper bound,
        # which at size is not a near miss (0.000 against 0.226 on Apex 25K at
        # three contracts). The reader must be able to tell which they got.
        fidelity="measured" if pool["have_path"] else "bounded",
        run=dict(
            outcome=rep["outcome"], reason=rep["reason"],
            date=rep["date"], day=rep["day"],
            trade_of_day=rep["trade_of_day"],
            deepest_fall=(max(t.intra_mdd for t in ordered) * contracts
                          if ordered else 0.0),
            allowed_fall=rules.max_dd,
            margin=rep["margin"], low_water=rep["low_water"],
            best_day=rep["best_day"], days=rep["days"],
            pnl_until=total - after, pnl_after=after,
            warnings=rep["warnings"], account=rep["account"]),
        projection=dict(
            p_pass=float(mc["p_pass"]), p_bust=float(mc["p_bust"]),
            p_timeout=float(mc["p_timeout"]),
            days_to_pass=(None if mc["days"] != mc["days"] else float(mc["days"])),
            sims=sims, pool_days=pool["n_days"],
            distinct_days=pool.get("distinct_days"),
            trades_per_day=pool.get("trades_per_day")))


def selfcheck():
    from types import SimpleNamespace

    def _t(day, i, pnl, mae, mfe, mdd, direction=1):
        return SimpleNamespace(date=f"2026-01-{day:02d}", entry_time=(day, i),
                               pnl=pnl, mae=mae, mfe=mfe, intra_mdd=mdd,
                               direction=direction, reason="target")

    meta = dict(commission=5.0, days=2)
    trades = [_t(1, 1, 100.0, -40.0, 120.0, 40.0, +1),
              _t(1, 2, -60.0, -80.0, 20.0, 100.0, -1),
              _t(2, 1, 250.0, -30.0, 300.0, 30.0, +1),
              _t(2, 2, -110.0, -150.0, 10.0, 160.0, -1)]

    g = grid(trades, meta)
    # THE IDENTITY THAT CATCHES A MISCLASSIFIED TRADE. Every trade lands in
    # exactly one of winners/losers/evens, and gross is net plus the commission
    # that was already taken out of it, so this holds whatever the partition.
    for col in ("all", "long", "short"):
        c = g[col]
        assert abs(c["gross_profit"] + c["gross_loss"] - c["commission"]
                   - c["net_profit"]) < 0.01, (col, c)
    # ...and the columns are a partition of the trades, not three filters that
    # happen to look right.
    for k in ("net_profit", "gross_profit", "gross_loss", "trades", "commission"):
        assert abs(g["long"][k] + g["short"][k] - g["all"][k]) < 0.01, (k, g)

    # ---- the prop-firm block ------------------------------------------------
    rules = RuleSet(
        firm="test", variant="test", phase="evaluation", size=50_000,
        start_balance=50_000.0, profit_target=3_000.0, profit_target_basis=None,
        max_dd=2_000.0, hwm_basis=NONE, breach_basis=INTRA, dd_lock_offset=None,
        daily_loss_limit=None, dll_soft=False, consistency_pct=None,
        consistency_effect=None, min_days=0, qualifying_day_min_profit=None,
        max_contracts=5, micro_ratio=10.0, profit_split=0.9,
        buffer_required=None, min_payout=None, max_withdrawal=None,
        automation_allowed=None, account_cost=None, reset_cost=None,
        unmodeled_rules=[], verified=True, retrieved="test", source_url=None)

    # Day 2's second trade digs $2,600 below a floor $2,000 under a static
    # $50,000 start, so the account dies there and the trades after it are
    # fiction. Written to kill on a KNOWN trade so the split can be checked
    # against the trade list rather than against itself.
    fatal = [_t(1, 1, 300.0, -100.0, 400.0, 100.0),
             _t(2, 1, 200.0, -100.0, 300.0, 100.0),
             _t(2, 2, -900.0, -2600.0, 100.0, 2700.0),
             _t(3, 1, 5000.0, -50.0, 5200.0, 50.0)]
    p = prop(fatal, rules, contracts=1, sims=500,
             rng=np.random.default_rng(4))
    assert p["run"]["outcome"] == "busted", p
    assert (p["run"]["day"], p["run"]["trade_of_day"]) == (2, 2), p

    # THE SPLIT MUST ADD UP. `until` comes from the replay's balance and `after`
    # from walking the trade list to the trade the replay named, so the two are
    # computed by different routes and only agree if the replay's day/trade
    # indices actually point at the trade that killed the account.
    total = sum(t.pnl for t in fatal)
    assert abs(p["run"]["pnl_until"] + p["run"]["pnl_after"] - total) < 0.01, p
    assert p["run"]["pnl_after"] == 4100.0, p     # -900 + 5000, the fiction

    # An unverified breach basis must disable the verdict, not default it to the
    # majority. Guessing here is how a simulator quietly invents a rule.
    unver = RuleSet(**dict(rules.__dict__, breach_basis=None))
    assert prop(fatal, unver, sims=200,
                rng=np.random.default_rng(4))["applicable"] is False

    # A daily loss limit fires AFTER the trade booked its loss, unlike an
    # intra-trade breach: that money really moved. The split adding up does NOT
    # catch a misattribution here, so the assert names the two halves rather
    # than their sum.
    dll_rules = RuleSet(**dict(rules.__dict__, daily_loss_limit=1000.0))
    hit = [_t(1, 1, -1200.0, -1300.0, 50.0, 1350.0),
           _t(2, 1, 5000.0, -50.0, 5200.0, 50.0)]
    d = prop(hit, dll_rules, contracts=1, sims=200, rng=np.random.default_rng(4))
    assert d["run"]["reason"] == "daily loss limit", d
    assert d["run"]["pnl_until"] == -1200.0, d
    assert d["run"]["pnl_after"] == 5000.0, d

    print(f"selfcheck OK: grid identity holds on {g['all']['trades']} trades "
          f"(net {g['all']['net_profit']:+,.2f} = gross "
          f"{g['all']['gross_profit']:,.2f} {g['all']['gross_loss']:+,.2f} "
          f"- commission {g['all']['commission']:,.2f}); "
          f"long+short == all"
          f"; the account dies on day {p['run']['day']} trade "
          f"{p['run']['trade_of_day']} and the split adds up "
          f"({p['run']['pnl_until']:+,.0f} real {p['run']['pnl_after']:+,.0f} "
          f"fiction); an unverified breach basis disables the verdict"
          f"; a daily loss limit still credits its booked loss to the real "
          f"half ({d['run']['pnl_until']:+,.0f} real {d['run']['pnl_after']:+,.0f} "
          f"fiction)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()


if __name__ == "__main__":
    main()
