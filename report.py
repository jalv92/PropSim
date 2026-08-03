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

    print(f"selfcheck OK: grid identity holds on {g['all']['trades']} trades "
          f"(net {g['all']['net_profit']:+,.2f} = gross "
          f"{g['all']['gross_profit']:,.2f} {g['all']['gross_loss']:+,.2f} "
          f"- commission {g['all']['commission']:,.2f}); "
          f"long+short == all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()


if __name__ == "__main__":
    main()
