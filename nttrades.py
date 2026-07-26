#!/usr/bin/env python3
"""Read the user's own trades straight out of NinjaTrader's database.

    python3 nttrades.py                       # summarise every account
    python3 nttrades.py --account 9           # one account, trade by trade

NinjaTrader keeps executions in `<NT8>/db/NinjaTrader.sqlite`. Reading them
directly means the user never exports anything: point at the NinjaTrader folder
and their live, sim and Playback trades are all there, with real per-trade
excursions and exact commissions.

WHAT THIS DOES NOT SEE. The `Backtest` account is always empty -- NinjaTrader
does not persist Strategy Analyzer results to this database. Strategy Analyzer
runs are captured by the companion NinjaScript AddOn instead. What IS here:
live accounts, `Sim101`, and **Playback**, which is how order-flow strategies
get evaluated correctly (NinjaTrader cannot backtest them in the Strategy
Analyzer at all -- Tick Replay and High Order Fill Resolution are mutually
exclusive, and those strategies need both).

TWO THINGS THAT WERE MEASURED, NOT ASSUMED, because both are silent killers:

1. `Executions.OrderId` joins to `Orders.OrderId` -- a STRING -- not to
   `Orders.Id`. The `Orders.Id` join looks right and returns zero rows; the
   string join matches 375 of 375.

2. Direction comes from `Orders.OrderAction`, never from price movement.
   Assuming every trade was long turned 28 real trades into 28 losses during
   development. The enum was calibrated against 142 paired round trips: exits
   are always the opposite side of their entry (0->2 x91, 2->0 x43, 2->1 x8),
   and entries with action 0 sit above their protective stop. So actions
   {0, 1} are the BUY side and {2, 3} the SELL side. `_check_pairing` asserts
   that invariant on every reconstruction rather than trusting this comment.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import ntdata

NET_EPOCH = datetime(1, 1, 1)
SENTINEL = 1.7976931348623157e308      # NT8 writes ±double.Max for "not set"
BUY_SIDE = (0, 1)                      # see module docstring -- calibrated
SELL_SIDE = (2, 3)


@dataclass
class Trade:
    account: str
    instrument: str
    direction: int                     # +1 long, -1 short
    qty: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    point_value: float
    commission: float
    pnl: float                         # net of commission and fees, in currency
    mae: float                         # <= 0, currency, real intra-trade low
    mfe: float                         # >= 0, currency

    @property
    def date(self) -> str:
        return self.entry_time.date().isoformat()


def _dt(net_ticks: int) -> datetime:
    return NET_EPOCH + timedelta(microseconds=net_ticks / 10)


def _open_readonly(db_path: Path):
    """Copy first, then open read-only.

    NinjaTrader holds this database open while it runs; querying the live file
    risks lock contention and a partially-written page. A copy costs a few MB
    and removes any chance of touching the user's data.
    """
    tmp = Path(tempfile.gettempdir()) / "propsim_nt8_snapshot.sqlite"
    shutil.copy2(db_path, tmp)
    con = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _instruments(con):
    """Instrument id -> (name, point value, tick size)."""
    return {r["Id"]: (r["Name"], r["PointValue"], r["TickSize"])
            for r in con.execute("""SELECT i.Id, m.Name, m.PointValue, m.TickSize
                                    FROM Instruments i
                                    JOIN MasterInstruments m ON m.Id=i.MasterInstrument""")}


def _accounts(con):
    return {r["Id"]: r["Name"] for r in con.execute("SELECT Id, Name FROM Accounts")}


def _check_pairing(entry_act, exit_act):
    """An exit must be the opposite side of its entry, or the pairing is wrong."""
    entry_buy = entry_act in BUY_SIDE
    exit_buy = exit_act in BUY_SIDE
    return entry_buy != exit_buy


def read_trades(nt_root=None, account=None) -> list[Trade]:
    """Reconstruct round-trip trades. `account` filters by name or id."""
    root = ntdata.resolve_root(nt_root)
    if root is None:
        raise SystemExit("No NinjaTrader 8 folder found — pass one explicitly.")
    db = Path(root) / "db" / "NinjaTrader.sqlite"
    if not db.exists():
        raise SystemExit(f"{db} not found")

    con = _open_readonly(db)
    instr = _instruments(con)
    accts = _accounts(con)

    rows = con.execute("""
        SELECT e.Account, e.Instrument, e.Time, e.Price, e.Quantity,
               e.Commission, e.Fee, e.IsEntry, e.IsExit,
               e.MaxPrice, e.MinPrice, o.OrderAction
        FROM Executions e
        JOIN Orders o ON o.OrderId = e.OrderId     -- string key; Orders.Id does NOT match
        ORDER BY e.Account, e.Instrument, e.Time""").fetchall()

    trades, mismatched = [], 0
    open_: dict[tuple, sqlite3.Row] = {}
    for r in rows:
        key = (r["Account"], r["Instrument"])
        if r["IsEntry"]:
            open_[key] = r
            continue
        if not r["IsExit"]:
            continue
        en = open_.pop(key, None)
        if en is None:
            continue                    # exit with no recorded entry (pre-existing position)

        if not _check_pairing(en["OrderAction"], r["OrderAction"]):
            mismatched += 1
            continue                    # do not guess -- drop and report

        direction = 1 if en["OrderAction"] in BUY_SIDE else -1
        name, pv, _tick = instr.get(r["Instrument"], ("?", 1.0, 0.01))
        qty = r["Quantity"] or 1
        gross = (r["Price"] - en["Price"]) * direction * pv * qty
        comm = (en["Commission"] or 0) + (r["Commission"] or 0) \
             + (en["Fee"] or 0) + (r["Fee"] or 0)

        # Excursions live on the ENTRY row; exit rows carry ±double.Max.
        # Adverse is the LOW for a long and the HIGH for a short.
        lo, hi = en["MinPrice"], en["MaxPrice"]
        have = abs(lo) < SENTINEL and abs(hi) < SENTINEL
        if have:
            adverse = lo if direction > 0 else hi
            favorable = hi if direction > 0 else lo
            mae = min((adverse - en["Price"]) * direction * pv * qty, 0.0)
            mfe = max((favorable - en["Price"]) * direction * pv * qty, 0.0)
        else:
            mae, mfe = min(gross - comm, 0.0), max(gross - comm, 0.0)

        trades.append(Trade(
            account=accts.get(r["Account"], str(r["Account"])), instrument=name,
            direction=direction, qty=qty,
            entry_time=_dt(en["Time"]), exit_time=_dt(r["Time"]),
            entry_price=en["Price"], exit_price=r["Price"], point_value=pv,
            commission=comm, pnl=gross - comm, mae=mae, mfe=mfe))

    if mismatched:
        print(f"  warning: {mismatched} execution pair(s) had a same-side "
              f"entry/exit and were dropped rather than signed by guesswork")
    if account is not None:
        want = str(account)
        trades = [t for t in trades
                  if t.account == want or want in t.account]
    return trades


def to_pool(trades: list[Trade], friction: float = 0.0) -> dict:
    """Day-block pool in the shape `sim.sim_eval` consumes.

    Day blocks, not individual trades: every prop-firm rule that matters --
    daily loss limits, consistency, minimum days, end-of-day ratcheting -- is a
    function of within-day structure, and resampling loose trades destroys it.
    """
    import numpy as np

    by_day = collections.defaultdict(list)
    for t in sorted(trades, key=lambda t: t.entry_time):
        by_day[t.date].append(t)
    days = [by_day[d] for d in sorted(by_day)]
    if not days:
        raise SystemExit("no trades to build a pool from")

    width = max(len(d) for d in days)
    pnl = np.zeros((len(days), width))
    mae = np.zeros((len(days), width))
    mask = np.zeros((len(days), width), bool)
    for i, day in enumerate(days):
        for j, t in enumerate(day):
            pnl[i, j] = t.pnl - friction
            mae[i, j] = t.mae - friction
            mask[i, j] = True
    # How many DISTINCT daily totals the bootstrap can draw from. This, not the
    # number of days, is what decides whether the simulated equity paths look
    # like a spread of futures or a lattice: a strategy taking one trade a day
    # with three possible outcomes gives four distinct days no matter how many
    # weeks it ran, and 1,000 resampled paths then collapse onto a handful of
    # trajectories. The chart is honest; without this number it reads as broken.
    daily = (pnl * mask).sum(axis=1)
    distinct = int(len(np.unique(np.round(daily, 2))))
    return dict(pnl=pnl, mae=mae, mask=mask, n_days=len(days), width=width,
                n_trades=int(mask.sum()), have_mae=True,
                distinct_days=distinct,
                trades_per_day=float(mask.sum(axis=1).mean()),
                mean=float(pnl[mask].mean()), source="NinjaTrader.sqlite")


def summarise(trades: list[Trade]) -> dict:
    n = len(trades)
    if not n:
        return dict(n=0)
    wins = [t for t in trades if t.pnl > 0]
    days = sorted({t.date for t in trades})
    longs = sum(1 for t in trades if t.direction > 0)
    return dict(n=n, days=len(days), first=days[0], last=days[-1],
                pnl=sum(t.pnl for t in trades), wr=len(wins) / n,
                longs=longs, shorts=n - longs,
                commission=sum(t.commission for t in trades),
                instruments=sorted({t.instrument for t in trades}))


def by_account(nt_root=None) -> dict:
    out = {}
    for t in read_trades(nt_root):
        out.setdefault(t.account, []).append(t)
    return {k: summarise(v) for k, v in sorted(out.items())}


# --------------------------------------------------------------------------
# account -> firm/phase detection
# --------------------------------------------------------------------------

def _patterns():
    import json
    import prop_rules
    try:
        return json.loads(prop_rules._res("account_patterns.json").read_text())
    except (OSError, ValueError):
        return {"patterns": []}


def detect_firm(account_name: str):
    """Best-effort (firm, phase, confidence) from an account name, or None.

    Longest prefix wins. A name identifies at most the firm and the phase --
    size and variant are NOT derivable, so the caller must still ask. Returned
    as a suggestion the UI labels as such, never as a settled fact: putting the
    wrong rule set in front of a funding decision is the failure mode this
    whole project exists to avoid.
    """
    up = (account_name or "").upper()
    best = None
    for pat in _patterns().get("patterns", []):
        pre = pat["prefix"].upper()
        if up.startswith(pre) and (best is None or len(pre) > len(best["prefix"])):
            best = pat
    if best is None:
        return None
    return dict(firm=best["firm"], phase=best.get("phase"),
                confidence=best.get("confidence", "unknown"),
                note=best.get("note", ""), matched=best["prefix"])


def detected_accounts(nt_root=None) -> list[dict]:
    """Every account with trades, its summary, and any firm match."""
    out = []
    for acct, s in by_account(nt_root).items():
        if not s.get("n"):
            continue
        out.append(dict(account=acct, **s, match=detect_firm(acct)))
    return sorted(out, key=lambda d: -d["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="NinjaTrader 8 folder")
    ap.add_argument("--account", help="account name or fragment")
    ap.add_argument("--list", action="store_true", help="per-account summary")
    args = ap.parse_args()

    if args.list or not args.account:
        print(f"{'account':<24}{'trades':>7}{'days':>6}{'P&L':>12}{'WR':>7}"
              f"   detected firm")
        for d in detected_accounts(args.root):
            m = d["match"]
            tag = "— unmatched, pick manually —" if not m else (
                f"{m['firm']}"
                + (f" / {m['phase']}" if m["phase"] else " / phase unknown")
                + ("" if m["confidence"] == "high" else f"  [{m['confidence']}]"))
            print(f"{d['account']:<24}{d['n']:>7}{d['days']:>6}{d['pnl']:>12,.2f}"
                  f"{d['wr']:>7.1%}   {tag}")
        return

    trades = read_trades(args.root, args.account)
    s = summarise(trades)
    if not s["n"]:
        raise SystemExit(f"no trades for account matching {args.account!r}")
    print(f"{s['n']} trades over {s['days']} days ({s['first']} .. {s['last']})  "
          f"P&L {s['pnl']:+,.2f}  WR {s['wr']:.1%}  "
          f"{s['longs']} long / {s['shorts']} short")
    print(f"\n{'date':<12}{'inst':<8}{'dir':>4}{'entry':>11}{'exit':>11}"
          f"{'P&L':>10}{'MAE':>10}")
    for t in trades[:25]:
        print(f"{t.date:<12}{t.instrument:<8}{'L' if t.direction > 0 else 'S':>4}"
              f"{t.entry_price:>11.2f}{t.exit_price:>11.2f}"
              f"{t.pnl:>10.2f}{t.mae:>10.2f}")
    if len(trades) > 25:
        print(f"  ... {len(trades) - 25} more")


if __name__ == "__main__":
    main()
