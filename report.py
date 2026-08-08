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

import ledger
import nttrades
import sim
from prop_rules import INTRA, NONE, RuleSet


def _fall(t):
    """The fall from a running peak inside a trade, or its upper bound.

    `None` means the source recorded no path. NinjaTrader stores MinPrice and
    MaxPrice and nothing between them, so for a trade read out of its database
    this is not recoverable, and `mfe - mae` is the widest the fall could have
    been. It is the same substitution `sim.trade_path` makes, and it is
    deliberately the pessimistic one: a shallower guess is the one that reports
    an account survived a day it did not.

    Without this the whole report crashed on any NinjaTrader trade -- `np.mean`
    over a list of None raises -- which is why real account trades could only
    ever be scored on the Monte Carlo side.
    """
    v = getattr(t, "intra_mdd", None)
    return (t.mfe - t.mae) if v is None else v


def path_measured(trades) -> bool:
    """True when every trade carried a real intra-trade path.

    The reader has to be able to tell a measured fall from a bounded one: on a
    trailing-drawdown account the two are not a rounding difference.
    """
    return all(getattr(t, "intra_mdd", None) is not None for t in trades)


def _column(trades, commission, days, contracts):
    """One column of the summary table for a subset of trades.

    `pnl` on a trade is ALREADY net of commission (engine.py books it as
    `gross - costs.commission`), so the gross figures are reconstructed by adding
    the commission back on. The identity
    `gross_profit + gross_loss - commission == net_profit` holds for any
    partition that covers every trade.

    TWO PARTITIONS, ON PURPOSE. The gross rows split on `gross > 0`, because a
    trade that lost less than the commission is a gross winner and summing it
    into gross loss makes a POSITIVE gross loss and an inflated profit factor.
    The net rows -- avg win/loss, win-loss ratio, consecutives, largest, percent
    profitable -- split on `net`, because that is the money the trader kept.
    """
    n = len(trades)
    if not n:
        return dict(net_profit=0.0, gross_profit=0.0, gross_loss=0.0,
                    commission=0.0, profit_factor=None, no_losses=False,
                    max_drawdown=0.0,
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
    gross_profit = float(gross[gross > 0].sum())
    gross_loss = float(gross[gross <= 0].sum())

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
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        commission=comm,
        profit_factor=(None if gross_profit == 0 and gross_loss == 0 else
                       (float("inf") if gross_loss == 0 else
                        abs(gross_profit / gross_loss))),
        # Infinity does not survive JSON (`dashboard._jsonable` nulls it), and a
        # null renders as an em dash -- the same thing "not computable" renders
        # as. A run with no losing trade is a real, reachable result and has to
        # be told apart from one the report could not measure.
        no_losses=bool(gross_loss == 0),
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
        avg_fall=float(np.mean([_fall(t) for t in trades]) * k),
        max_fall=float(max(_fall(t) for t in trades) * k))


def daily(trades, commission, contracts=1):
    """NinjaTrader's Analysis view at Period=Daily: one row per session.

    Every per-day figure comes from `_column`, the same function the summary
    table is built from, so a day cannot be measured one way and the run
    another. Only the two cumulative columns are computed here, and they are
    walked over TRADES rather than over daily closes: a day that ran +$900 and
    closed at +$100 drew down $800, and a curve sampled once a day cannot see it.

    ETD is NinjaTrader's "end trade drawdown" -- how much of the best unrealised
    profit was handed back before the exit. It is the gap between the MFE this
    project already measures and what the trade actually booked.
    """
    by_day: dict[str, list] = {}
    for t in trades:
        by_day.setdefault(t.date, []).append(t)
    k = float(contracts)
    total = len(trades)
    run = peak = cum_mdd = 0.0
    out = []
    for d in sorted(by_day):
        day = by_day[d]
        c = _column(day, commission, 1, contracts)
        for t in day:                       # inside the day, trade by trade
            run += t.pnl * k
            peak = max(peak, run)
            cum_mdd = min(cum_mdd, run - peak)
        out.append(dict(
            date=d, trades=c["trades"],
            cum_net=run, net_profit=c["net_profit"],
            gross_profit=c["gross_profit"], gross_loss=c["gross_loss"],
            commission=c["commission"],
            cum_max_drawdown=cum_mdd, max_drawdown=c["max_drawdown"],
            pct_profitable=c["pct_profitable"], avg_trade=c["avg_trade"],
            avg_win=c["avg_win"], avg_loss=c["avg_loss"],
            largest_win=c["largest_win"], largest_loss=c["largest_loss"],
            avg_mae=c["avg_mae"], avg_mfe=c["avg_mfe"],
            avg_etd=float(np.mean([t.mfe - t.pnl for t in day]) * k),
            # PropSim's own column, and the one no other daily table has: the
            # deepest fall from a running peak INSIDE a trade that day. It is
            # what an intraday trailing floor is tested against, so it is the
            # column that says whether a good-looking day was actually close to
            # ending the account.
            max_fall=c["max_fall"],
            pct_of_trades=(c["trades"] / total if total else 0.0)))
    return out


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
    lived = len(ordered)          # trades that were actually taken
    if rep["outcome"] in ("busted", "passed") and rep["date"] is not None:
        # Walk to the trade the replay named and sum from there. Deriving this
        # from the trade list rather than from `total - rep["profit"]` is the
        # point: the two only agree if the replay's indices are right.
        on_day = [i for i, t in enumerate(ordered) if t.date == rep["date"]]
        if rep["outcome"] == "passed":
            # A PASS IS DETECTED AT THE END OF A DAY, so `trade_of_day` is None
            # and every trade on that date really happened. The evaluation ended
            # there: the next day's trades are as fictional as post-bust ones.
            idx = lived = on_day[-1] + 1
        else:
            idx = lived = on_day[rep["trade_of_day"] - 1] + 1
            # WHY the account died decides whether the killing trade ever
            # booked. `sim.replay` breaches inside a trade BEFORE `bal += pnl`
            # -- the trade never finished, so its P&L is fiction. The close test
            # and the daily loss limit both fire AFTER it: that money really
            # moved, and calling it fiction tells the trader the opposite of
            # what happened. Its FALL is real either way -- an intra-trade
            # breach IS that fall -- so `lived` includes it regardless.
            if rep["reason"] == "drawdown floor, inside the trade":
                idx -= 1
        after = sum(t.pnl for t in ordered[idx:]) * contracts

    # A per-trade excursion is comparable to `max_dd` only when the floor is
    # what the trade was measured against -- i.e. when the account died on it.
    # On a static floor with a cushion the two are unrelated, and a bare
    # "(allowed $2,000)" next to $2,700 reads as a breach on a healthy account.
    floor_breach = (rep["reason"] or "").startswith("drawdown floor")

    pool = nttrades.to_pool(ordered)
    prof = dict(p=0.5, rr=1.0, risk=200.0, tpd=pool["width"],
                friction=0.0, pool=pool)
    mc = sim.sim_eval(prof, sim.policy_fixed(contracts), sims,
                      rng if rng is not None else np.random.default_rng(7),
                      rules)

    # ---- the chart -----------------------------------------------------------
    # Daily closing balance for EVERY day the strategy traded, split at the same
    # boundary the P&L split uses. `lived` carries the floor straight from
    # `sim.replay`; `after` is the tail that never happened and is drawn as
    # such. Summing P&L by day is arithmetic, not a rule, so it is done here --
    # but the FLOOR is never recomputed, only passed through.
    by_day: dict[str, float] = {}
    for t in ordered:
        by_day[t.date] = by_day.get(t.date, 0.0) + t.pnl * contracts
    running, full = rules.start_balance, []
    for d, p in by_day.items():
        running += p
        full.append(dict(date=d, balance=round(running, 2)))
    real_days = [dict(date=d, balance=round(b, 2), floor=round(f, 2))
                 for d, b, f in rep["curve"]]
    fiction_days = full[len(real_days) - 1:] if real_days else full

    return dict(
        chart=dict(lived=real_days, after=fiction_days,
                   start=rules.start_balance,
                   target=(rules.start_balance + rules.profit_target
                           if rules.profit_target else None),
                   ended=rep["outcome"] != "open"),
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
            # Over the trades that HAPPENED only. On a busted account the
            # deepest fall of the whole list can come from a trade after the
            # breach -- printed one row above the split whose entire job is
            # keeping those two worlds apart.
            deepest_fall=(max(_fall(t) for t in ordered[:lived]) * contracts
                          if ordered else 0.0),
            allowed_fall=rules.max_dd if floor_breach else None,
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
            # THE SAME FLOOR THE MONTE CARLO SCREEN HAS ALWAYS APPLIED. It is the
            # same statistic out of the same `sim_eval`, and it was gated on one
            # tab and published bare on the other: three sessions whose daily
            # totals are identical give the bootstrap no freedom at all, so all
            # 20,000 "resampled futures" are one day repeated and P(bust) comes
            # out exactly 0.000 or 1.000 by construction. `distinct_days` was
            # already computed and shipped here -- as a caption, which a number
            # on the same screen outranks every time.
            insufficient=(None if min(int(pool["n_days"]),
                                      int(pool.get("distinct_days")
                                          or pool["n_days"])) >= sim.MIN_DAYS
                          else dict(n_days=int(pool["n_days"]),
                                    distinct=int(pool.get("distinct_days")
                                                 or pool["n_days"]),
                                    needed=sim.MIN_DAYS,
                                    meaningful=sim.MEANINGFUL_DAYS)),
            # `p_pass` is 0.000 by construction on a phase with no profit target
            # (see `sim.sim_eval`), which reads as a verdict and is not one.
            has_target=bool(mc.get("has_target", True)),
            trades_per_day=pool.get("trades_per_day")))


def build(trades, meta, rules, contracts=1, sims=20_000, rng=None):
    """The whole report for one run: the table, the rules, and the price of the look.

    `prop` is None when there are no trades -- there is nothing to judge, and a
    verdict of "survived" on an empty range would be the most flattering lie the
    product could tell.
    """
    return dict(
        has_trades=bool(trades),
        grid=grid(trades, meta, contracts),
        daily=daily(trades, meta.get("commission", 0.0), contracts),
        # HOW MANY trades were judged on a bound rather than a measurement.
        # "At least one" is the right warning when it is one of fifty; when it
        # is fifty of fifty -- every trade read out of NinjaTrader, which records
        # no path at all -- the verdict itself is an upper bound on how badly it
        # went, and saying so is the difference between a finding and a libel.
        path_bounded=sum(1 for t in trades
                         if getattr(t, "intra_mdd", None) is None),
        prop=(prop(trades, rules, contracts, sims, rng) if trades else None),
        # Carried INSIDE the report, not on another tab: a t-statistic means
        # something different on trial 1 than on trial 43, and the two numbers
        # have to be read together or the first one wins by default.
        noise=ledger.stats(),
        meta=dict(meta, contracts=contracts))


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

    # THE DAILY TABLE MUST ADD UP TO THE SUMMARY TABLE. Two views of one run
    # that disagree is worse than having only one: the reader cannot tell which
    # is wrong, and both carry the same authority on screen.
    dy = daily(trades, meta["commission"])
    assert [r["date"] for r in dy] == ["2026-01-01", "2026-01-02"], dy
    assert abs(sum(r["net_profit"] for r in dy) - g["all"]["net_profit"]) < 0.01, dy
    assert abs(dy[-1]["cum_net"] - g["all"]["net_profit"]) < 0.01, dy
    assert sum(r["trades"] for r in dy) == g["all"]["trades"], dy
    assert abs(sum(r["pct_of_trades"] for r in dy) - 1.0) < 1e-9, dy
    # The cumulative drawdown is walked over TRADES, not over daily closes. Day
    # two runs +250 then -110 and CLOSES at +140, so a curve sampled once a day
    # never sees the 110 it gave back. Here the running peak is 40+250=290 and
    # the trough 180, so the deepest fall is 110 -- and it must not be the -100
    # that end-of-day sampling reports.
    assert abs(dy[-1]["cum_max_drawdown"] - (-110.0)) < 0.01, dy[-1]

    # A TRADE THAT LOST LESS THAN THE COMMISSION IS A GROSS WINNER. Splitting
    # the gross rows on the net sign sums it into gross loss -- which makes a
    # POSITIVE gross loss and a profit factor biased upward, every time. The
    # identity above cannot see it: it holds for any partition.
    scalp = grid([_t(1, i, -2.0, -20.0, 5.0, 22.0) for i in (1, 2, 3)]
                 + [_t(2, 1, 40.0, -10.0, 60.0, 12.0)], meta)["all"]
    assert scalp["gross_profit"] == 54.0 and scalp["gross_loss"] == 0.0, scalp
    assert abs(scalp["gross_profit"] + scalp["gross_loss"]
               - scalp["commission"] - scalp["net_profit"]) < 0.01, scalp
    # ...and "no losing trades" is a reachable answer, not the em dash that
    # "not computable" renders as, so it gets its own flag rather than an
    # infinity JSON cannot carry.
    assert scalp["no_losses"] is True and scalp["profit_factor"] == float("inf")
    assert scalp["losses"] == 3, scalp   # the NET rows still say what was kept

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
             _t(3, 1, 5000.0, -50.0, 5200.0, 9000.0)]
    p = prop(fatal, rules, contracts=1, sims=500,
             rng=np.random.default_rng(4))
    assert p["run"]["outcome"] == "busted", p
    assert (p["run"]["day"], p["run"]["trade_of_day"]) == (2, 2), p

    # The deepest fall must come from the trades that HAPPENED. Day 3 falls
    # $9,000 on an account that died on day 2, and printing that as this run's
    # deepest fall contradicts the split two rows below it.
    assert p["run"]["deepest_fall"] == 2700.0, p
    # The account died ON the floor, so the allowance is the number it was
    # measured against and the comparison is honest.
    assert p["run"]["allowed_fall"] == 2000.0, p

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
    # A daily loss limit is not a drawdown floor, so there is nothing to compare
    # the fall against and the allowance is withheld rather than printed beside
    # an unrelated number.
    assert d["run"]["allowed_fall"] is None, d

    # A PASSED ACCOUNT ALSO ENDS. The evaluation stops the day the target is
    # met, so the days after it are exactly as fictional as post-bust trades.
    # The pass is detected at the END of the day, so every trade on that date is
    # real. The sum identity cannot catch a wrong boundary here -- it holds
    # wherever the boundary is put -- so both halves are named.
    won = [_t(1, 1, 3000.0, -100.0, 3100.0, 100.0),
           _t(2, 1, 900.0, -50.0, 1000.0, 5000.0),
           _t(3, 1, 900.0, -50.0, 1000.0, 60.0),
           _t(4, 1, 900.0, -50.0, 1000.0, 60.0),
           _t(5, 1, 900.0, -50.0, 1000.0, 60.0)]
    w = prop(won, rules, contracts=1, sims=200, rng=np.random.default_rng(4))
    assert w["run"]["outcome"] == "passed", w
    assert w["run"]["trade_of_day"] is None, w
    assert w["run"]["pnl_until"] == 3000.0, w
    assert w["run"]["pnl_after"] == 3600.0, w
    assert w["run"]["deepest_fall"] == 100.0, w      # not day 2's $5,000
    assert w["run"]["allowed_fall"] is None, w

    # TWO TRADES THAT COMPARE EQUAL. `engine.Trade` is a plain dataclass, so
    # `list.index()` matches by VALUE and hands back the FIRST of an identical
    # pair -- putting the boundary a trade early whenever the SECOND one is what
    # breached, and moving real money into the fiction half. The lookup is
    # positional for exactly that reason; this is what holds it there. Value
    # matching would report 0 banked and -1,800 fictional.
    twin = [_t(1, 1, -900.0, -1100.0, 50.0, 1150.0),
            _t(1, 1, -900.0, -1100.0, 50.0, 1150.0)]
    tw = prop(twin, rules, contracts=1, sims=200, rng=np.random.default_rng(4))
    assert tw["run"]["trade_of_day"] == 2, tw
    assert tw["run"]["pnl_until"] == -900.0, tw
    assert tw["run"]["pnl_after"] == -900.0, tw

    # The chart splits at the SAME boundary as the P&L, and its floor comes from
    # the replay rather than a second calculation. A chart drawn from its own
    # floor would be the third definition of the rule this module exists to
    # avoid, and it would drift silently -- the picture and the verdict beside
    # it would stop agreeing.
    ch = p["chart"]
    assert len(ch["lived"]) == p["run"]["days"], (len(ch["lived"]), p["run"]["days"])
    assert all(set(pt) == {"date", "balance", "floor"} for pt in ch["lived"]), ch["lived"][:1]
    assert ch["after"][0]["date"] == ch["lived"][-1]["date"], (ch["after"][0], ch["lived"][-1])
    assert ch["ended"] is True, ch

    # ---- report composes sim, it does not re-derive it ----------------------
    # THE ONE PROPERTY THIS MODULE EXISTS TO PRESERVE. Asserted against `sim`
    # directly so that inlining a drawdown comparison here fails loudly instead
    # of quietly becoming a third definition of the rule.
    pool = nttrades.to_pool(fatal)
    prof = dict(p=0.5, rr=1.0, risk=200.0, tpd=pool["width"],
                friction=0.0, pool=pool)
    mc = sim.sim_eval(prof, sim.policy_fixed(1), 500,
                      np.random.default_rng(4), rules)
    assert abs(p["projection"]["p_bust"] - float(mc["p_bust"])) < 1e-12, (p, mc)
    rep = sim.replay(fatal, rules, contracts=1)
    assert (p["run"]["outcome"], p["run"]["reason"], p["run"]["day"]) == (
        rep["outcome"], rep["reason"], rep["day"]), (p, rep)

    # ---- build --------------------------------------------------------------
    b = build(fatal, dict(meta, days=3), rules, contracts=1, sims=200,
              rng=np.random.default_rng(4))
    assert b["has_trades"] and b["grid"]["all"]["trades"] == 4, b
    assert b["prop"]["run"]["outcome"] == "busted", b
    assert set(b["noise"]) >= {"trials", "expected_max_t", "threshold_t"}, b

    # A range with no trades is a legitimate answer, not a crash. Every earlier
    # version of a report in this project has fallen over on the empty case at
    # least once, always in front of the user.
    empty = build([], dict(meta, days=0), rules, sims=200,
                  rng=np.random.default_rng(4))
    assert empty["has_trades"] is False, empty
    assert empty["grid"]["all"]["trades"] == 0, empty
    assert empty["prop"] is None, empty

    # A long-only strategy is a normal result, not an edge case, and it empties
    # a whole column. The zero-trade branch has to return the SAME key set as
    # the populated one, or the day someone adds a metric to one branch and not
    # the other the report renders a hole and nothing here notices.
    only_long = grid([t for t in trades if t.direction > 0], meta)
    assert only_long["short"]["trades"] == 0, only_long
    assert set(only_long["short"]) == set(only_long["long"]), only_long

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
          f"fiction)"
          f"; a PASSED account splits at the end of its target day "
          f"({w['run']['pnl_until']:+,.0f} real {w['run']['pnl_after']:+,.0f} "
          f"fiction) and its deepest fall "
          f"({w['run']['deepest_fall']:,.0f}) ignores the days that never "
          f"happened; the allowance is shown only on a floor breach"
          f"; gross rows split on gross, so a sub-commission loser is a gross "
          f"winner (PF {scalp['profit_factor']}, no_losses "
          f"{scalp['no_losses']})"
          f"; p_bust equals sim.sim_eval ({mc['p_bust']:.4f}) and the verdict "
          f"equals sim.replay ({rep['outcome']}/{rep['reason']})"
          f"; a value-identical pair breaches on its SECOND trade "
          f"({tw['run']['pnl_until']:+,.0f} real {tw['run']['pnl_after']:+,.0f} "
          f"fiction), so the boundary is positional"
          f"; the chart splits where the P&L does ({len(p['chart']['lived'])} real "
          f"day(s), {len(p['chart']['after'])} drawn as fiction) and takes its "
          f"floor from the replay"
          f"; an empty range returns zeros instead of raising, and an emptied "
          f"column keeps every key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()


if __name__ == "__main__":
    main()
