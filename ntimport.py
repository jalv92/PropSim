#!/usr/bin/env python3
"""Import Strategy Analyzer runs captured by the PropSim NinjaScript add-on.

    python3 ntimport.py                  # every captured run
    python3 ntimport.py --show <run_id>  # one run, trade by trade
    python3 ntimport.py --selfcheck

The add-on (`nt8/`) writes one JSON file per run to `~/.prop-sim/backtests/`.
This module turns those files into the same `engine.Trade` objects the tape
engine produces, so a run of the user's OWN `.cs` strategy scores through
exactly the same path as everything else: trades -> day-block pool -> rules ->
P(pass).

TWO THINGS THIS MODULE REFUSES TO GUESS, because both are silent money errors:

1. WHETHER A RUN'S FILLS CAN BE TRUSTED. NinjaTrader cannot do Tick Replay and
   intrabar fill resolution at the same time -- they are mutually exclusive --
   so a strategy needing both cannot be backtested correctly there at all. Every
   file records the configuration that produced it and `fidelity()` turns that
   into a verdict the UI shows. A run captured by the performance metric carries
   no configuration at all and is labelled `unknown`, never assumed good.

2. WHETHER `ProfitCurrency` IS QUOTED BEFORE OR AFTER COMMISSION. At $5 a round
   trip this decides the answer on any high-frequency strategy, and it is not
   worth settling by belief: the add-on also exports NinjaTrader's own totals for
   the run, and `cost_basis()` reconciles the per-trade sum against them to find
   out which basis this build uses. When the totals are absent it assumes profit
   is GROSS and subtracts costs -- the pessimistic branch -- and says so.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import engine

SCHEMA = "propsim.trades/1"
RUNS_DIR = Path.home() / ".prop-sim" / "backtests"


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def read_file(path: Path) -> dict:
    """Parse one captured run. Raises SystemExit on anything unusable."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{Path(path).name}: unreadable ({exc})")
    got = d.get("schema")
    if got != SCHEMA:
        # Three producers write this format and one consumer reads it. An
        # unknown version is refused rather than half-understood.
        raise SystemExit(f"{Path(path).name}: schema {got!r}, expected {SCHEMA!r}"
                         " — update PropSim or re-export from NinjaTrader")
    if not isinstance(d.get("trades"), list):
        raise SystemExit(f"{Path(path).name}: no trade list")
    return d


def fidelity(meta: dict) -> dict:
    """Can this run's fills be believed? Returns level/label/detail.

    level is one of:
      exact           fills resolved tick by tick
      fills-guessed   stops resolved on the primary bar -- optimistic
      unknown         the run's configuration was not recorded
    """
    if meta.get("source") == "nt8-performance-metric" or meta.get("fill_resolution") is None:
        return dict(level="unknown", label="configuration unknown",
                    detail="Captured by the performance metric, which never sees "
                           "the strategy, so the bar type, Tick Replay setting and "
                           "order-fill resolution of this run were not recorded. "
                           "PropSim cannot certify these fills. Re-run it with the "
                           "PropSim fitness selected to get a labelled run.")
    high = str(meta.get("fill_resolution")) == "High"
    tick_bars = str(meta.get("bar_type")) == "Tick" and float(meta.get("bar_value") or 0) <= 1
    replay = bool(meta.get("tick_replay"))
    if high or tick_bars:
        d = ("Fills were resolved at tick granularity, so a stop and a target "
             "inside the same bar were ordered correctly.")
        if not replay:
            d += (" Tick Replay was off, which is the only way NinjaTrader allows "
                  "this — if the strategy reads the order flow (OnMarketData), its "
                  "SIGNALS were computed without ticks even though its fills are exact.")
        return dict(level="exact", label="fills resolved tick by tick", detail=d)
    return dict(level="fills-guessed", label="fills guessed on the bar",
                detail=("Order-fill resolution was Standard on "
                        f"{float(meta.get('bar_value') or 0):g}-"
                        f"{str(meta.get('bar_type')).lower()} bars, so "
                        "when a stop and a target both fell inside one bar NinjaTrader "
                        "picked an order rather than measuring it. Every intrabar stop "
                        "in this run is a guess, and the error is optimistic."
                        + (" Tick Replay was on, so the signals are right and only the "
                           "fills are not — that combination is unavoidable in "
                           "NinjaTrader, which forbids Tick Replay together with High "
                           "fill resolution." if replay else "")))


def cost_basis(meta: dict, rows: list[dict]) -> tuple[str, bool]:
    """(description, subtract_costs) — decided from the run's own totals."""
    profit = sum(float(r.get("profit") or 0.0) for r in rows)
    costs = sum(float(r.get("commission") or 0.0) + float(r.get("fee") or 0.0)
                for r in rows)
    perf = meta.get("perf") or {}
    gross = perf.get("gross_profit")
    loss = perf.get("gross_loss")
    cum = perf.get("cum_profit")
    tol = max(1.0, 0.005 * abs(profit))

    if cum is not None and abs(profit - float(cum)) <= tol and costs > tol:
        return (f"per-trade profit already nets commission (matches NinjaTrader's "
                f"cumulative {float(cum):,.2f})"), False
    if gross is not None and loss is not None and abs(profit - (float(gross) + float(loss))) <= tol:
        return (f"per-trade profit is gross (matches NinjaTrader's gross "
                f"{float(gross) + float(loss):,.2f}); ${costs:,.2f} of commission "
                f"and fees subtracted"), True
    if not perf:
        return ("NinjaTrader's own totals were not exported with this run, so "
                f"profit is assumed gross and ${costs:,.2f} of costs subtracted — "
                "cross-check the net against your Trade Performance tab"), True
    return (f"could not reconcile the per-trade sum {profit:,.2f} against "
            f"NinjaTrader's totals; assuming gross and subtracting ${costs:,.2f}"), True


def load_run(path: Path) -> tuple[list[engine.Trade], dict]:
    """One captured run as (trades, meta) in the engine's own shapes."""
    d = read_file(path)
    rows = [r for r in d["trades"] if r.get("entry") and r.get("exit")]
    basis, subtract = cost_basis(d, rows)

    trades: list[engine.Trade] = []
    for r in rows:
        cost = (float(r.get("commission") or 0.0) + float(r.get("fee") or 0.0)) if subtract else 0.0
        pnl = float(r.get("profit") or 0.0) - cost
        trades.append(engine.Trade(
            entry_time=_dt(r["entry"]), exit_time=_dt(r["exit"]),
            direction=int(r.get("dir") or 1),
            entry_price=float(r.get("entry_price") or 0.0),
            exit_price=float(r.get("exit_price") or 0.0),
            pnl=pnl,
            # NinjaTrader reports excursions as magnitudes; PropSim's convention
            # is signed (adverse <= 0 <= favourable), matching the SQLite reader
            # so both sources feed one pool builder.
            mae=-abs(float(r.get("mae") or 0.0)),
            mfe=abs(float(r.get("mfe") or 0.0)),
            reason=str(r.get("signal") or "nt8")))
    trades.sort(key=lambda t: t.entry_time)

    days = sorted({t.date for t in trades})
    meta = dict(
        run_id=Path(path).stem, path=str(path), source=d.get("source"),
        strategy=d.get("strategy") or "?", instrument=d.get("instrument") or "?",
        written=d.get("written"), params=d.get("params") or {},
        bar_type=d.get("bar_type"), bar_value=d.get("bar_value"),
        tick_replay=d.get("tick_replay"), fill_resolution=d.get("fill_resolution"),
        slippage_ticks=d.get("slippage_ticks"),
        n_trades=len(trades), days=len(days),
        start=days[0] if days else None, end=days[-1] if days else None,
        pnl=round(sum(t.pnl for t in trades), 2),
        cost_basis=basis, fidelity=fidelity(d))
    # The same three-input contract the tape engine reports, so the UI can label
    # an imported run and a native one identically.
    meta["label"] = f"{meta['strategy']} (NinjaTrader)"
    meta["contract"] = meta["instrument"]
    meta["timeframe"] = (f"{int(float(d.get('bar_value') or 0))}"
                         f"{str(d.get('bar_type') or '')[:1].lower()}"
                         if d.get("bar_type") else "?")
    return trades, meta


def list_runs(runs_dir: Path | None = None) -> list[dict]:
    """Every captured run, newest first, deduplicated.

    A user with both hooks enabled captures each optimisation iteration twice:
    once with its configuration (the fitness) and once without (the metric).
    Same run, two files -- so keep the labelled one. Deduplicating here rather
    than coordinating inside NinjaTrader is deliberate: optimisation iterations
    run in parallel across cores, and shared state between two NinjaScript
    objects is a race waiting to interleave two parameter sets into one file.
    """
    d = Path(runs_dir or RUNS_DIR)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            trades, meta = load_run(p)
        except SystemExit as exc:
            out.append(dict(run_id=p.stem, path=str(p), broken=str(exc),
                            n_trades=0, strategy="?", instrument="?",
                            fidelity=dict(level="unknown", label="unreadable",
                                          detail=str(exc))))
            continue
        if not trades:
            continue
        meta["key"] = (meta["instrument"], meta["n_trades"],
                       trades[0].entry_time.isoformat(), meta["pnl"])
        out.append(meta)

    best: dict = {}
    for m in out:
        k = m.get("key")
        if k is None:
            best[m["run_id"]] = m
            continue
        prev = best.get(k)
        if prev is None or (prev["fidelity"]["level"] == "unknown"
                           and m["fidelity"]["level"] != "unknown"):
            best[k] = m
    rows = list(best.values())
    rows.sort(key=lambda m: m.get("written") or "", reverse=True)
    for m in rows:
        m.pop("key", None)
    return rows


# --------------------------------------------------------------------------

def _fixture(dirpath: Path, source="nt8-optimization-fitness", perf=True) -> Path:
    """A synthetic captured run, in exactly the add-on's format."""
    trades, gross = [], 0.0
    for i in range(12):
        day = 2 + i // 3
        win = i % 3 != 0
        profit = 240.0 if win else -160.0
        gross += profit
        trades.append(dict(
            entry=f"2026-06-{day:02d}T09:{35 + (i % 3) * 8:02d}:00",
            exit=f"2026-06-{day:02d}T09:{40 + (i % 3) * 8:02d}:00",
            dir=1 if i % 2 else -1, qty=1,
            entry_price=20000.0 + i, exit_price=20012.0 + i,
            profit=profit, commission=4.0, fee=1.0,
            mae=80.0 if win else 160.0, mfe=260.0 if win else 20.0,
            signal="Long1"))
    doc = dict(schema=SCHEMA, source=source, written="2026-07-26T10:00:00",
               strategy="MyBarStrategy", instrument="NQ 09-26",
               point_value=20.0, tick_size=0.25,
               bar_type="Minute", bar_value=5.0, tick_replay=True,
               fill_resolution="Standard", fill_resolution_type="Minute",
               fill_resolution_value=1.0, slippage_ticks=0.0,
               calculate="OnBarClose", params=dict(Fast=20, Slow=60),
               trades=trades)
    if perf:
        wins = sum(t["profit"] for t in trades if t["profit"] > 0)
        losses = sum(t["profit"] for t in trades if t["profit"] < 0)
        doc["perf"] = dict(gross_profit=wins, gross_loss=losses,
                           cum_profit=gross - 12 * 5.0, trades=len(trades))
    else:
        doc["perf"] = None
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"fixture-{source[-6:]}-{'perf' if perf else 'noperf'}.json"
    p.write_text(json.dumps(doc))
    return p


def selfcheck():
    """Round-trip the add-on's format through to a simulator-ready pool."""
    import tempfile

    import nttrades
    tmp = Path(tempfile.mkdtemp(prefix="propsim-import-"))

    # 1. A labelled run: gross profit reconciles, costs get subtracted once.
    p = _fixture(tmp)
    trades, meta = load_run(p)
    assert len(trades) == 12, meta
    assert meta["fidelity"]["level"] == "fills-guessed", meta["fidelity"]
    assert "gross" in meta["cost_basis"], meta["cost_basis"]
    gross = 8 * 240.0 - 4 * 160.0
    assert abs(meta["pnl"] - (gross - 12 * 5.0)) < 0.01, meta["pnl"]

    # 2. MAE must be signed the way every other source signs it, or the
    #    intraday-breach test silently reads an adverse excursion as a gain.
    assert all(t.mae <= 0 <= t.mfe for t in trades)

    # 3. Totals absent -> assume gross, subtract, and SAY so.
    p2 = _fixture(tmp, perf=False)
    _, m2 = load_run(p2)
    assert "assumed gross" in m2["cost_basis"], m2["cost_basis"]

    # 4. A metric-captured run can never claim its fills are good.
    p3 = _fixture(tmp, source="nt8-performance-metric")
    _, m3 = load_run(p3)
    assert m3["fidelity"]["level"] == "unknown", m3["fidelity"]

    # 5. Dedup. All three fixtures describe the SAME run (same instrument, trade
    #    count, first entry and net), so the folder must collapse to one row --
    #    and the row that survives has to be the labelled one, or enabling both
    #    hooks would downgrade every run to "configuration unknown".
    rows = list_runs(tmp)
    assert len(rows) == 1, [r["run_id"] for r in rows]
    assert rows[0]["fidelity"]["level"] != "unknown", rows[0]

    # A genuinely different run is NOT merged away.
    only_metric = Path(tempfile.mkdtemp(prefix="propsim-import-metric-"))
    _fixture(only_metric, source="nt8-performance-metric")
    solo = list_runs(only_metric)
    assert len(solo) == 1 and solo[0]["fidelity"]["level"] == "unknown", solo

    # 6. An unknown schema is refused, not half-read.
    bad = tmp / "bad.json"
    bad.write_text(json.dumps(dict(schema="propsim.trades/99", trades=[])))
    try:
        read_file(bad)
        raise AssertionError("a future schema must be refused")
    except SystemExit:
        pass

    # 7. The whole point: it reaches the simulator as a day-block pool.
    pool = nttrades.to_pool(trades)
    assert pool["n_days"] == 4 and pool["n_trades"] == 12, pool
    print(f"selfcheck OK: {len(trades)} trades imported, net {meta['pnl']:+,.2f} "
          f"({meta['cost_basis']}); fidelity={meta['fidelity']['level']}; "
          f"pool {pool['n_days']} days x {pool['width']} slots; "
          f"schema guard and dedup hold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="where the add-on writes (default ~/.prop-sim/backtests)")
    ap.add_argument("--show", help="run id to print trade by trade")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    d = Path(args.dir) if args.dir else RUNS_DIR
    if args.show:
        p = next((q for q in d.glob("*.json") if q.stem == args.show), None)
        if p is None:
            raise SystemExit(f"no run {args.show!r} in {d}")
        trades, meta = load_run(p)
        print(f"{meta['strategy']} on {meta['instrument']} — "
              f"{meta['bar_value']:g}-{str(meta['bar_type']).lower()} bars, "
              f"{meta['n_trades']} trades over {meta['days']} days "
              f"({meta['start']} .. {meta['end']})")
        print(f"net {meta['pnl']:+,.2f}   [{meta['cost_basis']}]")
        print(f"fidelity: {meta['fidelity']['label']} — {meta['fidelity']['detail']}")
        if meta["params"]:
            print("params: " + ", ".join(f"{k}={v}" for k, v in meta["params"].items()))
        print(f"\n{'date':<12}{'dir':>4}{'entry':>11}{'exit':>11}{'P&L':>10}{'MAE':>10}")
        for t in trades[:25]:
            print(f"{t.date:<12}{'L' if t.direction > 0 else 'S':>4}"
                  f"{t.entry_price:>11.2f}{t.exit_price:>11.2f}"
                  f"{t.pnl:>10.2f}{t.mae:>10.2f}")
        if len(trades) > 25:
            print(f"  ... {len(trades) - 25} more")
        return

    rows = list_runs(d)
    if not rows:
        print(f"No captured runs in {d}.\n"
              "Install the add-on (see nt8/README.md), select 'PropSim export' as\n"
              "the Strategy Analyzer fitness, and run a backtest.")
        return
    print(f"{'run':<34}{'strategy':<20}{'trades':>7}{'days':>6}{'P&L':>12}   fidelity")
    for m in rows:
        if m.get("broken"):
            print(f"{m['run_id']:<34}{'— unreadable —':<20}{'':>7}{'':>6}{'':>12}   "
                  f"{m['fidelity']['label']}")
            continue
        print(f"{m['run_id']:<34}{m['strategy'][:19]:<20}{m['n_trades']:>7}"
              f"{m['days']:>6}{m['pnl']:>12,.2f}   {m['fidelity']['label']}")


if __name__ == "__main__":
    main()
