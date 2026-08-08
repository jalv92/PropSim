#!/usr/bin/env python3
"""Read the user's own trades straight out of NinjaTrader's database.

    python3 nttrades.py                       # summarise every account
    python3 nttrades.py --strategies          # summarise every NinjaScript strategy
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

AND WHICH STRATEGY MADE EACH TRADE IS IN HERE TOO. `Strategy2Execution` joins an
execution to a row in `Strategies`, which carries the NinjaScript `Classname` and
an `IsReplay` flag. That turns "the trades on this account" into "the trades THIS
strategy made", which is the only useful question on an account that also holds
discretionary trades. Measured on this database: 386 executions, 56 attributed --
all of them to ATM templates, because no NinjaScript strategy has completed a
Playback session yet.

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
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import ntdata
import sim            # for `trade_path` only; sim never imports this module

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
    # Which NinjaScript strategy produced this trade, when NinjaTrader recorded
    # it (see `_strategies`). None for discretionary trades. Defaults last so the
    # dataclass keeps working for every existing caller.
    strategy: str | None = None
    strategy_id: str | None = None
    is_replay: bool = False
    # Always None from this reader: NinjaTrader stores MinPrice/MaxPrice and
    # nothing about the path between them, so the fall from the running peak is
    # not recoverable here. Left unknown on purpose -- the simulator substitutes
    # its upper bound `mfe - mae` rather than inventing a shallower one.
    # See `engine.Trade.intra_mdd`.
    intra_mdd: float | None = None

    @property
    def date(self) -> str:
        return self.entry_time.date().isoformat()


def _dt(net_ticks: int) -> datetime:
    return NET_EPOCH + timedelta(microseconds=net_ticks / 10)


def _open_readonly(db_path: Path):
    """Snapshot the live database into memory, then read that.

    NinjaTrader holds this file open and writes to it while it runs, so the
    snapshot has to be taken in a way that survives a concurrent writer.

    THIS WAS A BYTE COPY AND THAT IS NOT SAFE. `shutil.copy2` reads six
    megabytes while NinjaTrader is rewriting pages inside it, so the result can
    hold pages from two different transactions -- a file that opens fine and
    then fails somewhere else entirely with "database disk image is malformed",
    three frames deep in whatever query happened to touch a torn page. The same
    race, caught a moment earlier, surfaced instead as `PermissionError`
    [WinError 32] when Windows had the file locked and `CopyFile2` could not
    start at all. Both were observed in the wild, and the database itself was
    healthy each time.

    SQLite's own backup API is the fix: it reads under a shared lock and
    restarts if the source changes underneath it, so what comes out is a
    consistent snapshot rather than a smear of one. `timeout` covers the case
    where NinjaTrader is mid-transaction and the read lock has to wait.

    The destination is memory, not a temp file, for a second reason: the old
    code reused one fixed path, and this server is threaded, so two requests
    arriving together wrote the same snapshot on top of each other. Six
    megabytes in memory has no such race and leaves nothing behind on disk.
    """
    try:
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            con = sqlite3.connect(":memory:")
            src.backup(con)
        finally:
            src.close()
        # Prove the snapshot before handing it out. A snapshot that is going to
        # fail should fail HERE, where the message can say what happened and the
        # caller already knows how to report it -- not later, inside an
        # unrelated query that has no idea the data came from a live file.
        con.execute("SELECT count(*) FROM Instruments").fetchone()
    except sqlite3.Error as exc:
        raise SystemExit(
            f"Could not read NinjaTrader's database: {exc}. The usual cause is "
            f"NinjaTrader writing to it at that moment — try again, or close "
            f"NinjaTrader if it keeps happening.") from exc
    con.row_factory = sqlite3.Row
    return con


def _instruments(con):
    """Instrument id -> (name, point value, tick size)."""
    return {r["Id"]: (r["Name"], r["PointValue"], r["TickSize"])
            for r in con.execute("""SELECT i.Id, m.Name, m.PointValue, m.TickSize
                                    FROM Instruments i
                                    JOIN MasterInstruments m ON m.Id=i.MasterInstrument""")}


# Filled on first successful read and never re-read. Only successes are cached:
# a database that was locked once must not pin an empty answer for the life of
# the dashboard process.
_MASTER: dict[str, tuple[float, float]] = {}


def master_instruments(nt_root=None) -> dict[str, tuple[float, float]]:
    """Root symbol -> (point value, tick size), e.g. 'MNQ' -> (2.0, 0.25).

    The multiplier is a property of the INSTRUMENT, and NinjaTrader already
    knows it for everything it can hand this project a tape of, so it is read
    rather than tabulated. Empty dict if the database cannot be read -- the
    caller falls back to its own table (see `engine.instrument`) rather than
    failing a backtest over a transient lock.

    FUTURES ONLY, AND THAT IS NOT A TIDINESS FILTER. `Name` is unique per asset
    class, not per table: this database holds "ES" the E-mini S&P future
    ($50/point) and "ES" Eversource Energy the NYSE stock ($1/point), plus an
    options row at $0. Keyed by name alone the stock wins on row order and a
    futures backtest prices 50 points of S&P as one dollar. Measured here:
    ES and CL both collided, GC and NQ did not. `InstrumentType = 0` is
    futures -- 358 of the 1,748 rows on this machine.
    """
    if _MASTER:
        return _MASTER
    root = ntdata.resolve_root(nt_root)
    db = (Path(root) / "db" / "NinjaTrader.sqlite") if root else None
    if db is None or not db.exists():
        return _MASTER
    try:
        con = _open_readonly(db)
    except SystemExit:
        return _MASTER
    _MASTER.update(
        (r["Name"], (float(r["PointValue"]), float(r["TickSize"])))
        for r in con.execute("SELECT Name, PointValue, TickSize "
                             "FROM MasterInstruments WHERE InstrumentType = 0")
        if r["Name"] and r["PointValue"] and r["TickSize"])
    return _MASTER


# NinjaTrader's connection provider id for the Playback connection. Measured on a
# real database: Playback101 = 13, while Sim101 and Backtest are 15 and live
# brokers are 38/50. This is how a replay run is identified, because
# `Strategies.IsReplay` is NOT set for one -- a strategy that traded a full
# Playback session came back with IsReplay = 0.
PLAYBACK_PROVIDER = 13


def _accounts(con):
    return {r["Id"]: (r["Name"], r["Provider"])
            for r in con.execute("SELECT Id, Name, Provider FROM Accounts")}


def _strategies(con):
    """Execution id -> the strategy that produced it.

    NinjaTrader keeps this in tables nothing else in this project used:
    `Strategies` (Classname, Name, Template, IsReplay) joined to executions
    through `Strategy2Execution`. It is the difference between "these are the
    trades on this account" and "these are the trades THIS STRATEGY made", which
    is the whole question when the account also holds discretionary trades.

    `IsReplay` is what marks a Market Replay / Playback run, and that matters:
    a Playback session is the only way an order-flow strategy can be evaluated
    at all, because the Strategy Analyzer cannot feed it depth or tick data with
    intrabar fills at the same time.
    """
    try:
        strat = {r["Id"]: dict(
            # Classname is the NinjaScript type, e.g.
            # NinjaTrader.NinjaScript.Strategies.BigPrintsStrategy. The bare
            # class name is what the user recognises.
            classname=r["Classname"] or "",
            name=(r["Classname"] or "").rsplit(".", 1)[-1] or (r["Name"] or "?"),
            label=r["Name"] or "",
            template=r["Template"] or "",
            is_replay=bool(r["IsReplay"]))
            for r in con.execute("SELECT Id, Classname, Name, Template, IsReplay "
                                 "FROM Strategies")}
        by_exec = {}
        for r in con.execute("SELECT Execution, Strategy FROM Strategy2Execution"):
            s = strat.get(r["Strategy"])
            if s:
                by_exec[r["Execution"]] = dict(s, id=str(r["Strategy"]))
        return by_exec
    except sqlite3.Error:
        # An older NinjaTrader schema without these tables must not break the
        # account-level reading that has always worked.
        return {}


def _check_pairing(entry_act, exit_act):
    """An exit must be the opposite side of its entry, or the pairing is wrong."""
    entry_buy = entry_act in BUY_SIDE
    exit_buy = exit_act in BUY_SIDE
    return entry_buy != exit_buy


def read_trades(nt_root=None, account=None, strategy=None) -> list[Trade]:
    """Reconstruct round-trip trades.

    `account` filters by account name or id; `strategy` filters by the recorded
    NinjaScript strategy (its id, or a substring of its class name).
    """
    root = ntdata.resolve_root(nt_root)
    if root is None:
        raise SystemExit("No NinjaTrader 8 folder found — pass one explicitly.")
    db = Path(root) / "db" / "NinjaTrader.sqlite"
    if not db.exists():
        raise SystemExit(f"{db} not found")

    con = _open_readonly(db)
    instr = _instruments(con)
    accts = _accounts(con)
    strat_of = _strategies(con)

    rows = con.execute("""
        SELECT e.Id, e.Account, e.Instrument, e.Time, e.Price, e.Quantity,
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

        # Attribution comes from the ENTRY: that is the execution that opened the
        # position, and a protective exit is sometimes recorded without it.
        s = strat_of.get(en["Id"]) or strat_of.get(r["Id"]) or {}
        acct_name, provider = accts.get(r["Account"], (str(r["Account"]), None))
        trades.append(Trade(
            account=acct_name, instrument=name,
            direction=direction, qty=qty,
            entry_time=_dt(en["Time"]), exit_time=_dt(r["Time"]),
            entry_price=en["Price"], exit_price=r["Price"], point_value=pv,
            commission=comm, pnl=gross - comm, mae=mae, mfe=mfe,
            strategy=s.get("name"), strategy_id=s.get("id"),
            is_replay=(provider == PLAYBACK_PROVIDER) or bool(s.get("is_replay"))))

    if mismatched:
        print(f"  warning: {mismatched} execution pair(s) had a same-side "
              f"entry/exit and were dropped rather than signed by guesswork")
    if account is not None:
        want = str(account)
        trades = [t for t in trades
                  if t.account == want or want in t.account]
    if strategy is not None:
        want = str(strategy)
        trades = [t for t in trades
                  if t.strategy_id == want
                  or (t.strategy and want.lower() in t.strategy.lower())]
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
    mfe = np.zeros((len(days), width))
    mdd = np.zeros((len(days), width))
    mask = np.zeros((len(days), width), bool)
    known_path = True
    for i, day in enumerate(days):
        for j, t in enumerate(day):
            # One shared definition with `sim.replay` -- see `sim.trade_path`.
            pnl[i, j], mae[i, j], mfe[i, j], mdd[i, j] = sim.trade_path(t, friction)
            if getattr(t, "intra_mdd", None) is None:
                known_path = False
            mask[i, j] = True
    # How many DISTINCT daily totals the bootstrap can draw from. This, not the
    # number of days, is what decides whether the simulated equity paths look
    # like a spread of futures or a lattice: a strategy taking one trade a day
    # with three possible outcomes gives four distinct days no matter how many
    # weeks it ran, and 1,000 resampled paths then collapse onto a handful of
    # trajectories. The chart is honest; without this number it reads as broken.
    daily = (pnl * mask).sum(axis=1)
    distinct = int(len(np.unique(np.round(daily, 2))))
    return dict(pnl=pnl, mae=mae, mfe=mfe, intra_mdd=mdd, mask=mask,
                n_days=len(days), width=width,
                n_trades=int(mask.sum()), have_mae=True,
                have_path=known_path,
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


def detected_strategies(nt_root=None) -> list[dict]:
    """Every NinjaScript strategy NinjaTrader has trades for, with its own stats.

    This is what makes "evaluate the strategies I actually have" answerable from
    the database alone, with no add-on: any strategy that has ever traded -- live,
    on Sim101, or in a Playback session -- left attributed executions behind.

    ATM template instances are excluded. NinjaTrader records each one as a
    separate `Strategies` row named AtmStrategy, so they are discretionary trades
    wearing a strategy's clothes: dozens of one-trade "strategies" that would
    drown the real ones in the picker.
    """
    groups: dict[tuple, list[Trade]] = {}
    for t in read_trades(nt_root):
        if not t.strategy_id or t.strategy == "AtmStrategy":
            continue
        groups.setdefault((t.strategy_id, t.strategy, t.account, t.is_replay),
                          []).append(t)
    out = []
    for (sid, name, acct, replay), tl in groups.items():
        s = summarise(tl)
        out.append(dict(strategy_id=sid, strategy=name, account=acct,
                        is_replay=replay, **s))
    return sorted(out, key=lambda d: -d["n"])


def detected_accounts(nt_root=None) -> list[dict]:
    """Every account with trades, its summary, and any firm match."""
    out = []
    for acct, s in by_account(nt_root).items():
        if not s.get("n"):
            continue
        out.append(dict(account=acct, **s, match=detect_firm(acct)))
    return sorted(out, key=lambda d: -d["n"])


def selfcheck():
    """The snapshot, and the two ways reading a live database goes wrong.

    Scoped to `_open_readonly` on purpose: the rest of this module was covered
    by the calibration work in the docstring, and this is the part that reads a
    file another process is writing.
    """
    import tempfile, threading

    root = ntdata.resolve_root()
    if root is None:
        print("no NinjaTrader folder found — nothing to check")
        return
    db = Path(root) / "db" / "NinjaTrader.sqlite"
    if not db.exists():
        print(f"{db} not found — nothing to check")
        return

    con = _open_readonly(db)
    assert con.execute("PRAGMA quick_check").fetchone()[0] == "ok", \
        "the snapshot must be a consistent database"
    n_instr = con.execute("SELECT count(*) FROM Instruments").fetchone()[0]
    assert n_instr > 0, "a snapshot with no instruments is not a NinjaTrader database"
    con.close()

    # A TORN SNAPSHOT MUST FAIL HERE, NOT THREE FRAMES LATER. This is the whole
    # point of the change: a byte copy of a live database yields a file that
    # opens fine and then raises "database disk image is malformed" inside some
    # unrelated query, where nothing knows to blame the copy.
    torn = Path(tempfile.gettempdir()) / "propsim_selfcheck_torn.sqlite"
    raw = db.read_bytes()
    torn.write_bytes(raw[:len(raw) // 2] + b"\x00" * (len(raw) // 2))
    try:
        _open_readonly(torn)
        raise AssertionError("a torn database must not be accepted")
    except SystemExit as exc:
        assert "NinjaTrader" in str(exc), f"the message must name the cause: {exc}"
    finally:
        torn.unlink(missing_ok=True)

    # And two requests arriving together must both get their own snapshot. The
    # previous implementation wrote one fixed temp path from a threaded server.
    counts, errs = [], []
    def one():
        try:
            c = _open_readonly(db)
            counts.append(c.execute("SELECT count(*) FROM Instruments").fetchone()[0])
            c.close()
        except Exception as e:                       # noqa: BLE001 -- reported below
            errs.append(f"{type(e).__name__}: {e}")
    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs, f"concurrent snapshots failed: {errs[:3]}"
    assert len(set(counts)) == 1, f"concurrent snapshots disagreed: {set(counts)}"

    trades = read_trades()
    print(f"selfcheck OK: snapshot of a live database is consistent "
          f"({n_instr:,} instruments, {len(trades)} trades); a torn one is "
          f"refused by name; 8 concurrent snapshots agree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--root", help="NinjaTrader 8 folder")
    ap.add_argument("--account", help="account name or fragment")
    ap.add_argument("--list", action="store_true", help="per-account summary")
    ap.add_argument("--strategies", action="store_true",
                    help="per-NinjaScript-strategy summary")
    ap.add_argument("--replay", action="store_true",
                    help="walk these trades in their real order against a "
                         "firm's rules and report whether they broke the account")
    ap.add_argument("--firm", default="my_funded_futures")
    ap.add_argument("--variant", default="rapid")
    ap.add_argument("--phase", default="evaluation")
    ap.add_argument("--size", type=int, default=50_000)
    ap.add_argument("--strategy", help="restrict to one NinjaScript strategy")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.strategies:
        rows = detected_strategies(args.root)
        if not rows:
            print("No NinjaScript strategy has recorded trades in this database.\n"
                  "Run one in Market Replay / Playback (or live on Sim101) and its\n"
                  "trades will appear here attributed to it -- NinjaTrader records\n"
                  "the link in Strategy2Execution. Strategy Analyzer runs do NOT\n"
                  "land here; those need the add-on in nt8/.")
            return
        print(f"{'strategy':<26}{'account':<22}{'where':<10}{'trades':>7}{'days':>6}"
              f"{'P&L':>12}{'WR':>7}")
        for d in rows:
            print(f"{d['strategy'][:25]:<26}{d['account'][:21]:<22}"
                  f"{('Playback' if d['is_replay'] else 'live/sim'):<10}"
                  f"{d['n']:>7}{d['days']:>6}{d['pnl']:>12,.2f}{d['wr']:>7.1%}")
        return

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

    trades = read_trades(args.root, args.account, strategy=args.strategy)
    s = summarise(trades)
    if not s["n"]:
        raise SystemExit(f"no trades for account matching {args.account!r}")

    if args.replay:
        from prop_rules import select
        r = sim.replay(trades, select(args.firm, args.variant,
                                      args.phase, args.size))
        verdict = {"busted": "BROKE THE ACCOUNT", "passed": "PASSED",
                   "open": "still open"}[r["outcome"]]
        print(f"{s['n']} trades, {s['days']} days ({s['first']} .. {s['last']}) "
              f"vs {r['account']}\n")
        print(f"  {verdict}" + (f" -- {r['reason']}" if r["reason"] else ""))
        if r["date"]:
            where = f"  on {r['date']} (day {r['day']}"
            if r["trade_of_day"]:
                where += f", trade {r['trade_of_day']} of that day"
            print(where + ")")
        print(f"  P&L {r['profit']:+,.2f}   lowest equity {r['low_water']:,.2f}"
              f"   best day {r['best_day']:+,.2f}")
        if r["margin"] is not None:
            print(f"  closest approach to the drawdown floor: "
                  f"${r['margin']:,.2f}")
        # The excursions decide an intraday floor, and NinjaTrader's database
        # does not record the path between them. Saying so is the difference
        # between a verdict and a guess dressed as one.
        pool = to_pool(trades)
        if not pool["have_path"]:
            print("\n  note: this database stores each trade's best and worst "
                  "price but not\n  the path between them, so the deepest fall "
                  "is taken at its upper bound.\n  A verdict of BROKE THE "
                  "ACCOUNT on a trailing-intraday firm may be the\n  bound, not "
                  "the trade. Re-run it from Playback through the add-on for\n"
                  "  a measured path.")
        for w in r["warnings"]:
            print(f"  !! {w}")
        return
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
