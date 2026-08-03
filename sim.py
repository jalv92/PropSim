#!/usr/bin/env python3
"""Monte Carlo simulator for prop-firm accounts, driven by `prop_rules.py`.

Rules are NOT hardcoded here any more. Every account parameter comes from a
`RuleSet` selected out of the canonical table:

    from prop_rules import select
    rules = select("my_funded_futures", "rapid", "evaluation", 50_000)
    sim_eval(profile, policy, sims, rng, rules)

Adding a firm is adding a data entry in `docs/research/prop-rules/` and
re-running `python3 prop_rules.py --build`. No logic change.

    python3 sim.py --selfcheck
    python3 sim.py --firm topstep --variant standard_path --size 50000

TWO BUGS FIXED IN THE 2026-07-25 REFACTOR, both found by the rules research and
both of which made every previous pass-rate figure optimistic:

1. BREACH WAS TESTED ONLY AT THE CLOSE. The old model checked the drawdown
   floor once per day on the closing balance. Four of five firms state the
   floor is breached in REAL TIME on equity including unrealized P&L -- MFF
   verbatim: "open equity losses are taken into consideration when calculating
   whether or not the account failed on this rule." A strategy that dips below
   the floor intraday and recovers into the close was surviving in the sim and
   dead in reality.

2. THE FUNDED HIGH-WATER MARK IGNORED UNREALIZED GAINS. The pre-refactor `sim_funded`
   carried a `ponytail:` comment admitting this. MFF confirmed the ratchet is
   also unrealized: "the trailing drawdown is calculated intraday based on your
   peak balance, which includes both realized and unrealized gains." So the
   floor ratcheted up more slowly in the sim than in reality.

3. THE RATCHET USED THE CLOSED P&L, NOT THE UNREALIZED PEAK -- fix 2 only half
   done. `hwm_basis == INTRA` ratcheted on `max(pnl, 0)`, so a trade that ran
   +$800 in the money and closed at +$200 raised the floor by $200 in the sim
   and by $800 in reality. The MFE was measured by the backtest engine and
   thrown away here. Worse, two scalars have no order: peak-then-dip ratchets
   the floor UP and then drops the account against it, while dip-then-peak
   tests the low against a floor that has not moved. Trades now carry
   `mae_first`, measured from the tick indices where the engine produced them
   and assumed pessimistically (peak first) where the source cannot time them.
   This only ever LOWERS survival on an intraday-trailing account, which is the
   hardest rule any of these firms sell.

All three are now driven by `rules.hwm_basis` / `rules.breach_basis`.

A strategy is modeled as a per-trade Bernoulli: win prob `p`, win pays
`rr * risk` per contract, loss costs `risk` per contract, and every contract
pays `friction` per round trip. Sizing policies decide contracts from the live
account state.
"""
from __future__ import annotations

import argparse

import numpy as np

from prop_rules import EOD, INTRA, NONE, RuleSet, firms, select, variants

EVAL_HORIZON_DAYS = 90      # count as fail if not passed by then
FUNDED_HORIZON_DAYS = 250

# ponytail: a Bernoulli trade has no price path, so intra-trade excursion is
# modeled, not simulated. A LOSER touches its stop by construction (MAE = full
# risk). A WINNER never touched its stop -- it would have been stopped out --
# so its MAE lies in [0, risk); MAE_FRAC_WIN is where in that range we put it.
# 0.5 is a deliberate midpoint, not a measurement. It only matters when
# breach_basis is intraday_equity, and it is the single largest modeling
# assumption in this file. Feed real per-trade MAE from a backtest to remove it.
MAE_FRAC_WIN = 0.5

# The mirror of the above for the favourable side. A WINNER's peak is exact, not
# assumed: it exits AT its target, so the peak is the full `rr * risk`. A LOSER
# never reached the target, so its MFE lies in [0, rr*risk); this is where in
# that range we put it. Same status as MAE_FRAC_WIN -- a midpoint, not a
# measurement -- and it only bites when hwm_basis is intraday_equity.
MFE_FRAC_LOSS = 0.5

# Defaults for callers that pass no rule set -- notably the frozen round*.py
# research scripts, which reach this module through the `mff_sim` shim left in
# MFF-Sim. Loaded from the canonical table, so they get the corrected MFF rules
# rather than the old hardcoded constants.
DEFAULT_EVAL = select("my_funded_futures", "rapid", "evaluation", 50_000)
DEFAULT_FUNDED = select("my_funded_futures", "rapid", "funded_sim", 50_000)


# ---- Strategy profiles ------------------------------------------------------
# risk = $ lost per contract on a stop-out (e.g. 10 NQ pts = $200).
# friction = $ per contract per round trip (commission + slippage).
PROFILES = {
    "zero-edge 1:1":      dict(p=0.50, rr=1.0, risk=200.0, tpd=4, friction=5.0),
    "high-WR low-R":      dict(p=0.62, rr=0.6, risk=200.0, tpd=4, friction=5.0),
    "modest edge 1:1":    dict(p=0.53, rr=1.0, risk=200.0, tpd=4, friction=5.0),
    # Zarattini ORB reports avg trade 0.13R at 24% WR -> avg win ~3.7R
    "ORB-like 24% 3.7R":  dict(p=0.24, rr=3.7, risk=100.0, tpd=1, friction=5.0),
    # arXiv positive controls: ~+$25/mini net -> p=.55, rr=1.2 on $150 risk
    "regime 60min hold":  dict(p=0.55, rr=1.2, risk=150.0, tpd=2, friction=5.0),
}


# ---- Sizing policies --------------------------------------------------------

def policy_fixed(n):
    def f(buffer, remaining, prof, rules):
        return np.full_like(buffer, float(n))
    f.__name__ = f"fixed-{n}"
    return f


def policy_bold(buffer, remaining, prof, rules):
    """Dubins-Savage bold play, capped: enough contracts to reach the target in
    one winning trade, but never more than the buffer survives."""
    win_per_con = prof["rr"] * prof["risk"] - prof["friction"]
    loss_per_con = prof["risk"] + prof["friction"]
    need = np.ceil(remaining / max(win_per_con, 1.0))
    survivable = np.floor(buffer / loss_per_con)
    return np.clip(np.minimum(need, np.maximum(survivable, 1.0)), 1, _cap(rules))


def policy_governor(frac):
    """Risk `frac` of the live buffer per trade."""
    def f(buffer, remaining, prof, rules):
        loss_per_con = prof["risk"] + prof["friction"]
        return np.clip(np.floor(buffer * frac / loss_per_con), 1, _cap(rules))
    f.__name__ = f"governor-{int(frac * 100)}%"
    return f


POLICIES = [policy_fixed(1), policy_fixed(3), policy_fixed(5),
            policy_bold, policy_governor(0.25), policy_governor(0.50)]


def _cap(rules: RuleSet) -> int:
    return int(rules.max_contracts) if rules.max_contracts else 5


def _trade_pnl(rng, prof, n, sims):
    """Returns (realized_pnl, intra_trade_low, intra_trade_high, mae_first)."""
    win = rng.random(sims) < prof["p"]
    pnl = np.where(win,
                   prof["rr"] * prof["risk"] - prof["friction"],
                   -prof["risk"] - prof["friction"]) * n
    # worst and best equity excursions reached *during* the trade
    mae = np.where(win, -prof["risk"] * MAE_FRAC_WIN, -prof["risk"]) * n
    target = prof["rr"] * prof["risk"]
    mfe = np.where(win, target, target * MFE_FRAC_LOSS) * n
    # Not an assumption, a consequence of the model: a Bernoulli winner exits at
    # its target, so its peak is the LAST thing that happened and the dip came
    # first. A loser exits at its stop, so its low is last. Same structure the
    # tick engine measures, which is why its self-check asserts exactly this.
    return pnl, np.minimum(mae, pnl), np.maximum(mfe, pnl), win


def load_trades(path, friction=0.0):
    """Read a real trade list into a DAY-BLOCK pool for bootstrap resampling.

    Accepts any CSV with a per-trade profit column (profit/pnl/net/...) in
    account currency for ONE contract. Optional columns: a date (enables day
    blocks) and an MAE column (removes the MAE_FRAC_WIN assumption).

    Day blocks rather than iid trades because every prop-firm rule that matters
    -- daily loss limits, consistency, minimum days, end-of-day ratcheting --
    is a function of WITHIN-DAY structure. Resampling individual trades
    destroys exactly the thing being simulated.
    """
    import csv as _csv
    rows = list(_csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    cols = {k.lower().strip(): k for k in rows[0]}

    def pick(*names):
        for n in names:
            for c, orig in cols.items():
                if n in c:
                    return orig
        return None

    pcol = pick("profit", "pnl", "p&l", "net")
    if pcol is None:
        raise SystemExit(f"{path}: no profit column in {list(cols)}")
    dcol = pick("date", "time", "entry")
    mcol = pick("mae", "adverse", "drawdown")
    fcol = pick("mfe", "favorable", "favourable", "runup", "run-up")

    def num(v):
        v = str(v).replace("$", "").replace(",", "").replace("(", "-").replace(")", "")
        try:
            return float(v)
        except ValueError:
            return None

    days, cur, key = [], [], None
    for r in rows:
        v = num(r[pcol])
        if v is None:
            continue
        m = num(r[mcol]) if mcol else None
        f = num(r[fcol]) if fcol else None
        k = str(r[dcol])[:10] if dcol else None
        if dcol and k != key and cur:
            days.append(cur); cur = []
        key = k
        cur.append((v - friction, -abs(m) if m is not None else None,
                    abs(f) if f is not None else None))
    if cur:
        days.append(cur)
    if not dcol:                      # no date column -> one trade per "day"
        days = [[t] for d in days for t in d]

    width = max(len(d) for d in days)
    pnl = np.zeros((len(days), width))
    mae = np.zeros((len(days), width))
    mfe = np.zeros((len(days), width))
    mask = np.zeros((len(days), width), bool)
    have_mae = mcol is not None
    for i, d in enumerate(days):
        for j, (v, m, f) in enumerate(d):
            pnl[i, j] = v
            mae[i, j] = m if m is not None else min(v, 0.0)
            mfe[i, j] = f if f is not None else max(v, 0.0)
            mask[i, j] = True
    # A CSV carries two magnitudes and no timing, so the order is unknown and
    # resolves to the pessimistic one: the peak ratchets an intraday floor up
    # before the low is tested against it. All-False, not a guess per row.
    return dict(pnl=pnl, mae=mae, mfe=mfe, mask=mask,
                mae_first=np.zeros((len(days), width), bool),
                n_days=len(days), width=width,
                n_trades=int(mask.sum()), have_mae=have_mae, have_order=False,
                mean=float(pnl[mask].mean()), source=str(path))


def _day_draw(rng, prof, sims):
    """Pick one real trading day per sim, or None when running parametric.

    Shared by sim_eval and sim_funded: they drifted apart once already -- the
    pool was wired into the evaluation only, so the funded phase silently fell
    back to a Bernoulli whose p=0 lost every trade and reported P(payout)=0.
    """
    pool = prof.get("pool")
    if pool is None:
        return None, prof["tpd"]
    return rng.integers(0, pool["n_days"], sims), pool["width"]


def _slot_pnl(rng, prof, pool, pick, slot, n, sims):
    """P&L, both intra-trade extremes and their order, for one trade slot."""
    if pool is None:
        return _trade_pnl(rng, prof, n, sims)
    pnl = pool["pnl"][pick, slot] * n
    return (pnl,
            np.minimum(pool["mae"][pick, slot] * n, pnl),
            np.maximum(pool["mfe"][pick, slot] * n, pnl),
            pool["mae_first"][pick, slot])


def _walk_trade(hwm, bal, busted, active, low, high, mae_first, rules):
    """Breach test and high-water ratchet for ONE trade, walked IN ORDER.

    Returns the updated (hwm, busted). Split out of the two simulation loops
    because they had already drifted apart once (see `_day_draw`), and this is
    the block where a divergence is worth real money.

    The order only matters when the floor moves during the trade -- an intraday
    trailing account. There, a peak that arrives FIRST ratchets the floor up and
    the account then falls the full peak-to-trough distance against it, while a
    peak that arrives LAST is tested against a floor that has not moved yet.
    Same two numbers, different outcome.
    """
    if rules.hwm_basis != INTRA:
        # Static or end-of-day floor: it cannot move mid-trade, so the sequence
        # inside the trade is unobservable and only the low is tested.
        if rules.breach_basis == INTRA:
            busted = busted | (active & (bal + low <= _floor(hwm, rules)))
        return hwm, busted

    up = np.maximum(hwm, bal + high)          # the unrealized peak, not max(pnl,0)
    if rules.breach_basis != INTRA:
        return np.where(active & ~busted, up, hwm), busted

    floor_at_low = np.where(mae_first, _floor(hwm, rules), _floor(up, rules))
    busted = busted | (active & (bal + low <= floor_at_low))
    return np.where(active & ~busted, up, hwm), busted


def _floor(hwm, rules: RuleSet):
    """Drawdown floor given the current high-water mark."""
    if rules.max_dd is None:
        return np.full_like(hwm, -np.inf)
    f = hwm - rules.max_dd
    lock = rules.dd_floor_lock
    if lock is not None:
        f = np.minimum(f, lock)          # floor stops trailing at the lock point
    return f


# ---- Eval simulation --------------------------------------------------------

def sim_eval(prof, policy, sims, rng, rules: RuleSet | None = None,
             max_days=EVAL_HORIZON_DAYS, record_paths=0, on_day=None):
    """Pass/bust/timeout rates for the evaluation phase of `rules`.

    `rules` defaults to the MFF 50K Rapid evaluation so that the frozen
    pre-registered research scripts (round2.py and friends) keep running
    unedited -- editing a pre-registered file would invalidate it. They now get
    the corrected drawdown semantics for free.
    """
    rules = rules or DEFAULT_EVAL
    start = rules.start_balance
    target = rules.profit_target or 0.0
    bal = np.full(sims, start)
    hwm = np.full(sims, start)
    best_day = np.zeros(sims)
    total_days = np.zeros(sims)
    passed = np.zeros(sims, bool)
    busted = np.zeros(sims, bool)
    pass_day = np.zeros(sims)
    bust_day = np.zeros(sims)

    # equity paths for the dashboard's spaghetti chart: EOD balance per day for
    # the first `record_paths` sims. Kept as a preallocated array written once
    # per day -- the cost is one row-copy per simulated day, not per trade.
    rec = min(record_paths, sims)
    paths = np.full((rec, max_days + 1), np.nan) if rec else None
    prev_done = np.zeros(rec, bool)
    if rec:
        paths[:, 0] = start

    cons = rules.consistency_pct
    # a consistency rule that only blocks payouts does not constrain the eval
    cons_binds = cons is not None and rules.consistency_effect in (
        "blocks_pass", "inflates_target")
    day_cap = (cons / 100.0) * target * 0.95 if cons_binds else np.inf
    min_days = rules.min_days or 0

    for day in range(max_days):
        active = ~passed & ~busted
        if not active.any():
            if rec:
                paths[:, day + 1:] = np.nan
            break
        day_pnl = np.zeros(sims)
        day_open = bal.copy()
        pool = prof.get("pool")
        pick, slots = _day_draw(rng, prof, sims)
        was_bust = busted.copy()   # before the trade loop: intraday breaches flip
                                   # `busted` in there, so capturing it after would
                                   # make every intraday bust look like day zero

        for slot in range(slots):
            trade = active & (day_pnl < day_cap)
            if pool is not None:
                trade = trade & pool["mask"][pick, slot]
            if not trade.any():
                if pool is None:
                    break          # parametric: every sim trades every slot
                continue           # pooled: this slot is empty for all sims,
                                   # but a later one may not be
            floor = _floor(hwm, rules)
            n = policy(bal - floor,
                       np.maximum(start + target - bal, 0.0), prof, rules)
            pnl, low, high, mae_first = _slot_pnl(
                rng, prof, pool, pick, slot, n, sims)

            # BUG FIXES 1-3: breach on the intra-trade low rather than the
            # close, ratchet on the unrealized PEAK rather than the closed P&L,
            # and do both in the order the price path actually walked.
            hwm, busted = _walk_trade(hwm, bal, busted, trade,
                                      low, high, mae_first, rules)

            bal = np.where(trade & ~busted, bal + pnl, bal)
            day_pnl = np.where(trade & ~busted, day_pnl + pnl, day_pnl)
            if rules.breach_basis != INTRA:
                busted |= trade & (bal <= _floor(hwm, rules))

            # daily loss limit (hard breach unless the firm calls it a pause)
            if rules.daily_loss_limit:
                hit = trade & ((bal - day_open) <= -rules.daily_loss_limit)
                if rules.dll_soft:
                    day_pnl = np.where(hit, np.inf, day_pnl)   # stop trading today
                else:
                    busted |= hit

        total_days = np.where(active, total_days + 1, total_days)
        busted |= active & (bal <= _floor(hwm, rules))
        bust_day = np.where(busted & ~was_bust, total_days, bust_day)
        alive = active & ~busted
        if rules.hwm_basis == EOD:
            hwm = np.where(alive, np.maximum(hwm, bal), hwm)
        best_day = np.where(alive, np.maximum(best_day, np.where(
            np.isfinite(day_pnl), day_pnl, 0.0)), best_day)

        profit = bal - start
        need = target
        if cons_binds and rules.consistency_effect == "inflates_target":
            # Topstep/TPT: a big day raises the bar instead of failing you
            need = np.maximum(target, best_day / (cons / 100.0))
        ok = alive & (profit >= need) & (total_days >= min_days)
        if cons_binds and rules.consistency_effect == "blocks_pass":
            ok &= best_day <= (cons / 100.0) * profit
        pass_day = np.where(ok & ~passed, total_days, pass_day)
        passed |= ok

        if rec:
            # A path draws through the day it finishes -- that day's balance is
            # its final one -- and stops after. NaN breaks the polyline cleanly.
            # Writing on `prev_done` (state BEFORE this day) rather than `done`
            # is what makes the last point land; testing `done` would either
            # drop the final balance or repeat it forever.
            paths[:, day + 1] = np.where(prev_done, np.nan, bal[:rec])
            prev_done = (passed | busted)[:rec].copy()
        if on_day is not None and (day % 2 == 0 or day == max_days - 1):
            on_day(day, passed.mean(), busted.mean(), bal.copy())

    out = dict(p_pass=passed.mean(), p_bust=busted.mean(),
               p_timeout=(~passed & ~busted).mean(),
               days=pass_day[passed].mean() if passed.any() else float("nan"),
               bust_days=float(bust_day[busted].mean()) if busted.any() else float("nan"),
               final=bal - start, outcome=np.where(passed, 1, np.where(busted, 2, 0)),
               pass_day=pass_day, bust_day=bust_day)
    if rec:
        out["paths"] = paths
    return out


# ---- Funded simulation ------------------------------------------------------

def sim_funded(prof, policy, sims, rng, rules: RuleSet | None = None,
               max_days=FUNDED_HORIZON_DAYS, record_paths=0,
               start_day=None, record_idx=None):
    """Survival, payout stats and expected trader income for a funded account.

    `start_day` chains this phase to an evaluation: it is the number of trading
    days each path already spent passing, so the funded leg gets what is LEFT of
    the `max_days` planning horizon rather than a fresh one. A path that took 60
    days to pass trades 190, not 250, and `first_payout_day` comes back measured
    from the day the evaluation was bought. Give a path `max_days` to freeze it
    out entirely, which is how paths that never passed are excluded.

    `record_idx` picks WHICH sims are drawn on the chart. Under chaining the
    interesting ones are the paths that actually reached this phase; recording
    "the first N" would fill the funded chart with flat lines belonging to
    accounts that never existed.
    """
    rules = rules or DEFAULT_FUNDED
    start = rules.start_balance
    bal = np.full(sims, start)
    hwm = np.full(sims, start)
    busted = np.zeros(sims, bool)
    income = np.zeros(sims)
    n_payouts = np.zeros(sims)
    survive_days = np.zeros(sims)
    first_payout_day = np.full(sims, np.nan)

    # How many trading days this path still has. Without chaining every path
    # gets the whole horizon, which is the old behaviour.
    budget = (np.full(sims, float(max_days)) if start_day is None
              else np.maximum(float(max_days) - np.asarray(start_day, float), 0.0))
    offset = (np.zeros(sims) if start_day is None
              else np.asarray(start_day, float))

    rec = min(record_paths, sims)
    rows = (np.arange(rec) if record_idx is None
            else np.asarray(record_idx, int)[:rec])
    rec = len(rows)
    paths = np.full((rec, max_days + 1), np.nan) if rec else None
    prev_done = np.zeros(rec, bool)
    if rec:
        paths[:, 0] = start

    buffer_req = rules.buffer_required if rules.buffer_required is not None else start
    min_payout = rules.min_payout or 0.0
    split = rules.profit_split if rules.profit_split is not None else 0.9

    for day in range(max_days):
        active = ~busted & (day < budget)
        if not active.any():
            break
        pool = prof.get("pool")
        pick, slots = _day_draw(rng, prof, sims)
        for slot in range(slots):
            if pool is not None and not pool["mask"][pick, slot].any():
                continue
            floor = _floor(hwm, rules)
            n = policy(bal - floor,
                       np.maximum(buffer_req + min_payout - bal, min_payout),
                       prof, rules)
            pnl, low, high, mae_first = _slot_pnl(
                rng, prof, pool, pick, slot, n, sims)
            if pool is not None:                 # sims whose day has no trade here
                traded = pool["mask"][pick, slot]
                pnl = np.where(traded, pnl, 0.0)
                low = np.where(traded, low, 0.0)
                high = np.where(traded, high, 0.0)

            hwm, busted = _walk_trade(hwm, bal, busted, active,
                                      low, high, mae_first, rules)

            bal = np.where(active & ~busted, bal + pnl, bal)
            if rules.hwm_basis == INTRA:
                hwm = np.maximum(hwm, bal)
            busted |= active & (bal <= _floor(hwm, rules))
            active &= ~busted          # `= ~busted` here would resurrect a path
                                       # whose horizon has already run out

        if rules.hwm_basis == EOD:
            hwm = np.where(active, np.maximum(hwm, bal), hwm)

        withdrawable = bal - buffer_req
        pay = active & (withdrawable >= max(min_payout, 0.01))
        if rules.max_withdrawal is not None:
            withdrawable = np.minimum(withdrawable, rules.max_withdrawal)
        income += np.where(pay, withdrawable * split, 0.0)
        # Measured from the day the ATTEMPT started, not from day one of the
        # funded account: "you get paid on day 40" is a different claim from
        # "you get paid 40 days after passing an evaluation that took 60".
        first_payout_day = np.where(pay & np.isnan(first_payout_day),
                                    offset + day + 1, first_payout_day)
        n_payouts += pay
        bal = np.where(pay, bal - withdrawable, bal)
        # A withdrawal lowers the balance, so the high-water mark must come down
        # with it or the floor would sit ABOVE the post-payout balance and bust
        # the account instantly. Only bites for firms whose floor never locks
        # (dd_lock_offset is None) -- with a lock the lock hides it, which is
        # exactly why it is worth an explicit line rather than luck.
        hwm = np.where(pay, hwm - withdrawable, hwm)
        survive_days = np.where(active, survive_days + 1, survive_days)
        if rec:
            # cumulative withdrawn + live balance: what the trader has actually
            # taken out is the number that matters in the funded phase, not the
            # account balance, which resets to the buffer after every payout
            paths[:, day + 1] = np.where(prev_done, np.nan,
                                         bal[rows] + income[rows])
            prev_done = (busted | (day + 1 >= budget))[rows].copy()

    out = dict(p_payout=(n_payouts > 0).mean(), income=income.mean(),
               payouts=n_payouts.mean(), days=survive_days.mean(),
               p_bust=busted.mean(), survive_days=survive_days,
               income_path=income, first_payout_day=first_payout_day,
               # A clean partition, because "paid" and "busted" overlap: you can
               # withdraw and still lose the account later. The question that
               # matters is whether money came out before it died.
               #   0 = still alive at the horizon
               #   1 = busted, but took at least one payout first
               #   2 = busted with nothing withdrawn
               outcome=np.where(~busted, 0, np.where(n_payouts > 0, 1, 2)))
    if rec:
        out["paths"] = paths
        # which sims those rows belong to -- the caller needs it to line the
        # per-path outcome colours up with the drawn curves
        out["path_idx"] = rows
    return out


def sim_attempt(prof, policy, sims, rng, ev_rules, fu_rules, fee=100.0,
                record_paths=0, on_day=None, chain=True):
    """One end-to-end attempt: buy an evaluation, and if it passes, trade the
    funded account. Returns the per-path net EV distribution the dashboard's
    KPI tiles are built from.

    THE TWO LEGS ARE CHAINED (`chain=True`). Each path carries its own pass day
    into the funded phase, so:

      * a path that needed 60 trading days to pass gets the REMAINING planning
        horizon, not a fresh one -- a slow pass earns less, which is true and
        used to be invisible;
      * days-to-first-payout is measured from the day the evaluation was BOUGHT,
        which is the number a trader is actually deciding on;
      * paths that never passed consume no funded days at all, rather than being
        simulated in full and masked afterwards.

    What is deliberately NOT carried across is the balance: every firm in the
    table starts a funded account at its own nominal balance, so there is nothing
    to inherit. `chain=False` restores the old independent draw, and the
    self-check pins the direction of the difference between them.
    """
    ev = sim_eval(prof, policy, sims, rng, ev_rules,
                  record_paths=record_paths, on_day=on_day)
    passed = ev["outcome"] == 1

    if chain:
        # A path that never passed gets a zero-day funded horizon: same effect as
        # masking it out afterwards, without inventing the days it never traded.
        start_day = np.where(passed, ev["pass_day"], float(FUNDED_HORIZON_DAYS))
        # Draw the funded chart from paths that reached the funded phase. "The
        # first N sims" would fill it with accounts that never existed.
        idx = np.flatnonzero(passed)
        if idx.size == 0:
            idx = np.arange(min(record_paths, sims))
        fu = sim_funded(prof, policy, sims, rng, fu_rules,
                        record_paths=record_paths, start_day=start_day,
                        record_idx=idx)
    else:
        fu = sim_funded(prof, policy, sims, rng, fu_rules,
                        record_paths=record_paths)

    net = -fee + np.where(passed, fu["income_path"], 0.0)
    paid = passed & (fu["income_path"] > 0)
    fpd = fu["first_payout_day"][paid]
    return dict(
        ev=ev, fu=fu, net=net, passed=passed, chained=bool(chain),
        net_mean=float(net.mean()),
        net_p5=float(np.percentile(net, 5)), net_p95=float(np.percentile(net, 95)),
        p_payout=float(paid.mean()),
        mean_payout=float(fu["income_path"][paid].mean()) if paid.any() else 0.0,
        days_to_first_payout=float(np.nanmean(fpd)) if fpd.size else float("nan"),
    )


# ---- Self-check -------------------------------------------------------------

def _synthetic(**kw) -> RuleSet:
    base = dict(firm="test", variant="test", phase="evaluation", size=50_000,
                start_balance=50_000.0, profit_target=3000.0,
                profit_target_basis=None, max_dd=2000.0, hwm_basis=NONE,
                breach_basis=INTRA, dd_lock_offset=None, daily_loss_limit=None,
                dll_soft=False, consistency_pct=None, consistency_effect=None,
                min_days=0, qualifying_day_min_profit=None, max_contracts=5,
                micro_ratio=10.0, profit_split=0.9, buffer_required=None,
                min_payout=None, max_withdrawal=None,
                automation_allowed=None, account_cost=None,
                reset_cost=None, unmodeled_rules=[], verified=True,
                retrieved="test", source_url=None)
    base.update(kw)
    return RuleSet(**base)


def selfcheck(rng):
    # Gambler's ruin: symmetric zero-cost 1:1 walk, 1 contract, STATIC floor,
    # per-trade barrier. P(hit +3000 before -2000) with $200 steps = 10/25 = 0.4
    prof = dict(p=0.50, rr=1.0, risk=200.0, tpd=1, friction=0.0)
    flat = _synthetic(hwm_basis=NONE, breach_basis=INTRA, min_days=0)
    r = sim_eval(prof, policy_fixed(1), 40_000, rng, flat, max_days=2000)
    assert abs(r["p_pass"] - 0.40) < 0.02, f"gambler's ruin check failed: {r}"

    # BUG FIX 1 regression: an intraday breach basis must never pass MORE often
    # than a close-only one -- it can only kill paths the close-only test missed.
    trail = _synthetic(hwm_basis=EOD, breach_basis=INTRA)
    close_only = _synthetic(hwm_basis=EOD, breach_basis="realized")
    a = sim_eval(prof, policy_fixed(3), 20_000, rng, trail)
    b = sim_eval(prof, policy_fixed(3), 20_000, rng, close_only)
    assert a["p_pass"] <= b["p_pass"] + 0.005, (a, b)

    # BUG FIX 2 regression: ratcheting the HWM on unrealized peaks raises the
    # floor faster, so it must not increase survival.
    intra = _synthetic(hwm_basis=INTRA, breach_basis=INTRA)
    c = sim_eval(prof, policy_fixed(3), 20_000, rng, intra)
    assert c["p_pass"] <= a["p_pass"] + 0.005, (c, a)

    # BUG FIX 3 regression, in two parts. A pool lets the excursions be stated
    # exactly instead of drawn, so the two effects can be isolated from each
    # other and from the noise of a Bernoulli.
    def _pool(pnl_row, mae_row, mfe_row, first):
        arr = np.array(pnl_row, float).reshape(-1, 1)
        n = len(pnl_row)
        return dict(pnl=arr, mae=np.array(mae_row, float).reshape(-1, 1),
                    mfe=np.array(mfe_row, float).reshape(-1, 1),
                    mae_first=np.full((n, 1), first),
                    mask=np.ones((n, 1), bool), n_days=n, width=1, n_trades=n,
                    have_mae=True, have_order=True,
                    mean=float(arr.mean()), source="selfcheck")

    def _with(pool):
        return dict(p=0.5, rr=1.0, risk=200.0, tpd=1, friction=0.0, pool=pool)

    # One trade a day: +$600 booked, a $400 dip, and a peak that varies by day.
    # Sized so the account NEVER falls $2,000 below a floor that has not already
    # ratcheted -- which leaves the order of the two excursions as the only
    # thing that can kill it.
    pl, ma = [600.0] * 6, [-400.0] * 6
    mf = [1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2100.0]
    peak_first = sim_eval(_with(_pool(pl, ma, mf, False)), policy_fixed(1),
                          5_000, np.random.default_rng(3), intra)
    dip_first = sim_eval(_with(_pool(pl, ma, mf, True)), policy_fixed(1),
                         5_000, np.random.default_rng(3), intra)
    assert peak_first["p_pass"] < dip_first["p_pass"] - 0.20, (peak_first, dip_first)

    # ...and the peak has to be READ. Collapsing it to the closed P&L -- the bug
    # -- ratchets the floor by $600 instead of up to $2,100 and the same account
    # sails through. This is the assertion that fails if `mfe` is ever dropped
    # from the pool contract again.
    flat = sim_eval(_with(_pool(pl, ma, [max(v, 0.0) for v in pl], False)),
                    policy_fixed(1), 5_000, np.random.default_rng(3), intra)
    assert peak_first["p_pass"] < flat["p_pass"] - 0.20, (peak_first, flat)

    # Behind a floor that cannot move mid-trade there is nothing for the order
    # to change. Asserting EXACT equality on the same seed -- not "close" -- is
    # what proves the ordering branch is never consulted there.
    mixed = ([600.0, 600.0, -800.0, 600.0, -800.0, 600.0],
             [-400.0, -400.0, -900.0, -400.0, -900.0, -400.0],
             [1200.0, 900.0, 300.0, 1500.0, 200.0, 1100.0])
    eod_rules = _synthetic(hwm_basis=EOD, breach_basis=INTRA)
    ord_a = sim_eval(_with(_pool(*mixed, False)), policy_fixed(1), 5_000,
                     np.random.default_rng(5), eod_rules)
    ord_b = sim_eval(_with(_pool(*mixed, True)), policy_fixed(1), 5_000,
                     np.random.default_rng(5), eod_rules)
    assert (ord_a["p_pass"], ord_a["p_bust"]) == (ord_b["p_pass"], ord_b["p_bust"]), \
        (ord_a, ord_b)

    # consistency that only blocks payouts must not constrain the eval at all
    payout_only = _synthetic(consistency_pct=50, consistency_effect="blocks_payout")
    none_ = _synthetic(consistency_pct=None)
    d = sim_eval(prof, policy_fixed(3), 20_000, rng, payout_only)
    e = sim_eval(prof, policy_fixed(3), 20_000, rng, none_)
    assert abs(d["p_pass"] - e["p_pass"]) < 0.02, (d, e)

    # payout must not bust a no-lock account (hwm has to fall with the balance)
    nolock = _synthetic(phase="funded_sim", start_balance=0.0, profit_target=None,
                        dd_lock_offset=None, hwm_basis=EOD, buffer_required=2100.0,
                        min_payout=500.0, max_contracts=5)
    g = sim_funded(dict(p=0.60, rr=1.0, risk=200.0, tpd=4, friction=5.0),
                   policy_fixed(2), 5_000, rng, nolock)
    assert g["payouts"] > 1.0, f"no-lock account cannot repeat payouts: {g}"

    # the real MFF 50K Rapid eval must be loadable and simulate
    real = select("my_funded_futures", "rapid", "evaluation", 50_000)
    f = sim_eval(prof, policy_fixed(3), 5_000, rng, real)
    assert 0.0 <= f["p_pass"] <= 1.0

    # CHAINING regression. Spending days on the evaluation must cost something:
    # the funded leg starts later, so within one planning horizon a chained
    # attempt can never out-earn an independently drawn one, and its first payout
    # can never arrive sooner. Both directions are asserted because getting the
    # sign wrong here would make every slow-passing strategy look better.
    edge = dict(p=0.56, rr=1.0, risk=200.0, tpd=4, friction=5.0)
    ev_r = select("my_funded_futures", "rapid", "evaluation", 50_000)
    fu_r = select("my_funded_futures", "rapid", "funded_sim", 50_000)
    ch = sim_attempt(edge, policy_fixed(2), 8_000, np.random.default_rng(11),
                     ev_r, fu_r, fee=0.0, chain=True)
    ind = sim_attempt(edge, policy_fixed(2), 8_000, np.random.default_rng(11),
                      ev_r, fu_r, fee=0.0, chain=False)
    assert ch["net_mean"] <= ind["net_mean"] + 1e-6, (ch["net_mean"], ind["net_mean"])
    assert (ch["days_to_first_payout"] >= ind["days_to_first_payout"]
            or np.isnan(ind["days_to_first_payout"])), (ch, ind)

    print(f"selfcheck OK: gambler's ruin {r['p_pass']:.3f} ~ 0.400; "
          f"intraday breach {a['p_pass']:.3f} <= close-only {b['p_pass']:.3f}; "
          f"unrealized ratchet {c['p_pass']:.3f} <= realized {a['p_pass']:.3f}; "
          f"peak-first {peak_first['p_pass']:.3f} < dip-first "
          f"{dip_first['p_pass']:.3f} and < closed-P&L ratchet "
          f"{flat['p_pass']:.3f}; order is a no-op behind an EOD floor "
          f"({ord_a['p_pass']:.3f} == {ord_b['p_pass']:.3f}, "
          f"bust {ord_a['p_bust']:.3f}); "
          f"chained E[net] {ch['net_mean']:,.0f} <= independent "
          f"{ind['net_mean']:,.0f} and first payout day "
          f"{ch['days_to_first_payout']:.0f} >= {ind['days_to_first_payout']:.0f}")


# ---- Report -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--fee", type=float, default=100.0)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--firm", default="my_funded_futures")
    ap.add_argument("--variant", default="rapid")
    ap.add_argument("--size", type=int, default=50_000)
    ap.add_argument("--list-firms", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    if args.selfcheck:
        selfcheck(rng)
        return
    if args.list_firms:
        for f in firms():
            print(f"\n{f}:")
            for v, ph, sz in variants(f):
                print(f"  --variant {v:<28} --size {sz:<8} [{ph}]")
        return

    ev_rules = select(args.firm, args.variant, "evaluation", args.size)
    try:
        fu_rules = select(args.firm, args.variant, "funded_sim", args.size)
    except KeyError:
        fu_rules = ev_rules

    print(f"=== {args.firm} / {args.variant} / ${args.size:,} ===")
    print(f"eval: target ${ev_rules.profit_target:,.0f} dd ${ev_rules.max_dd:,.0f} "
          f"ratchet={ev_rules.hwm_basis} breach={ev_rules.breach_basis or 'UNVERIFIED'} "
          f"lock={ev_rules.dd_floor_lock}")
    for w in ev_rules.warnings():
        print(f"  !! {w}")

    for pname, prof in PROFILES.items():
        gross = prof["p"] * prof["rr"] * prof["risk"] - (1 - prof["p"]) * prof["risk"]
        print(f"\n--- {pname}  (net edge ${gross - prof['friction']:+.1f}/con/trade)")
        print(f"{'policy':<14}{'P(pass)':>8}{'P(bust)':>8}{'days':>6}"
              f"{'| P(payout)':>12}{'E[income]':>10}{'E[attempt]':>11}")
        for pol in POLICIES:
            ev = sim_eval(prof, pol, args.sims, rng, ev_rules)
            fu = sim_funded(prof, pol, args.sims, rng, fu_rules)
            print(f"{pol.__name__:<14}{ev['p_pass']:>8.1%}{ev['p_bust']:>8.1%}"
                  f"{ev['days']:>6.1f}{fu['p_payout']:>12.1%}"
                  f"{fu['income']:>10.0f}"
                  f"{-args.fee + ev['p_pass'] * fu['income']:>11.0f}")


if __name__ == "__main__":
    main()
