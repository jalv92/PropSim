#!/usr/bin/env python3
"""Parameter sweeps, scored against a pre-registered gate.

    python3 optimize.py --contract "NQ 09-26" --strategy orb --tf 5
    python3 optimize.py --contract "NQ 09-26" --strategy orb \\
        --range stop_ticks=20:60:10 --range rr=1:3:0.5
    python3 optimize.py --selfcheck

WHAT THIS IS FOR. Finding parameter values on your own tick data, fast, and then
running those values in NinjaTrader. 200 combinations over 28 days take about
half a minute here; the same search in Market Replay is days of wall clock.

WHAT IT REFUSES TO DO, and these are not preferences -- they were pre-registered
in the project's design before any of this code existed:

1. **P(pass) is never the objective.** It is a bounded, highly non-linear
   transform of the edge; near zero edge it is dominated by variance, so an
   optimiser pointed at it discovers lottery tickets rather than edge. The
   objective here is the cluster-robust t-statistic on DAILY aggregated P&L,
   among configurations that clear a minimum trade count. The prop-firm Monte
   Carlo is a pricing calculator that runs afterwards, on a result, never inside
   the search.

2. **The gate statistic is daily, not per trade.** Intraday trades share a
   session and are not independent; a per-trade t overstates significance by
   sqrt(1 + (k-1)rho), which at k=5, rho=0.4 is 1.6x. Aggregating to days
   absorbs whatever rho actually is, with no need to estimate it.

3. **Every combination is a trial, and the ledger is told so.** A sweep of 200
   raises the noise bar for everything that comes after it, permanently. The
   winner is compared against the expected maximum t of that many null trials --
   which is the number that decides whether "the best of 200" means anything.
   At 200 trials pure noise reaches t = 2.74.

4. **Nothing here is validated.** Everything on disk is exploration data. A
   configuration that clears the bar has earned ONE forward test on sessions
   recorded after today -- it has not been confirmed. The wording in the output
   says so, because "best" on a results table is read as "good".
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

import engine
import ledger

# The pre-registered bar: a hypothesis earns a forward test at train t >= 1.5,
# N >= 80 trades, and positive in at least 2 of 3 sub-periods.
MIN_T = 1.5
MIN_TRADES = 80
SUBPERIODS = 3
MIN_POSITIVE_SUBPERIODS = 2
MAX_COMBOS = 5000        # a hard stop; a sweep this size is already a red flag


def parse_range(spec: str):
    """`stop_ticks=20:60:10` -> ("stop_ticks", [20, 30, 40, 50, 60])."""
    key, _, rhs = spec.partition("=")
    if not rhs:
        raise SystemExit(f"bad range {spec!r}, expected key=lo:hi:step")
    if ":" not in rhs:
        return key.strip(), [float(v) for v in rhs.split(",")]
    lo, hi, step = (float(x) for x in rhs.split(":"))
    if step <= 0:
        raise SystemExit(f"bad step in {spec!r}")
    n = int(round((hi - lo) / step)) + 1
    return key.strip(), [round(lo + i * step, 10) for i in range(max(n, 1))]


def grid(ranges: dict[str, list], cap=MAX_COMBOS) -> list[dict]:
    keys = list(ranges)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(ranges[k] for k in keys))]
    if len(combos) > cap:
        raise SystemExit(f"{len(combos):,} combinations exceeds the cap of {cap:,} — "
                         f"narrow the ranges. A sweep this wide costs more in "
                         f"multiple-testing debt than any result it could find is worth.")
    return combos


def default_ranges(strategy: str, points=4) -> dict[str, list]:
    """A modest sweep from each parameter's declared bounds.

    Deliberately coarse. A fine grid over a wide range is how a sweep finds the
    one cell where noise happened to line up, and every cell it visits is charged
    to the ledger.
    """
    S = engine.LIBRARY[strategy]
    out = {}
    for k, p in S.params.items():
        if p.hi - p.lo <= 1:                      # a flag, not a dial
            continue
        lo = max(p.lo, p.default / 2)
        hi = min(p.hi, p.default * 2 if p.default else p.hi)
        if hi <= lo:
            lo, hi = p.lo, p.hi
        step = (hi - lo) / max(points - 1, 1)
        vals = [round(lo + i * step, 4) for i in range(points)]
        out[k] = sorted({(round(v) if float(p.default).is_integer() else v) for v in vals})
    return out


def subperiod_pnl(trades, n=SUBPERIODS) -> list[float]:
    """P&L per contiguous third of the TRADING DAYS in the run.

    Split by day rather than by trade so each block is a period of market, not a
    block of activity -- a strategy that traded twice in March and 400 times in
    July would otherwise get three "periods" that are all July.
    """
    if not trades:
        return [0.0] * n
    days = sorted({t.date for t in trades})
    if len(days) < n:
        return [float(sum(t.pnl for t in trades))] + [0.0] * (n - 1)
    edges = [days[int(round(i * len(days) / n))] for i in range(n)] + [None]
    out = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        out.append(float(sum(t.pnl for t in trades
                             if t.date >= lo and (hi is None or t.date < hi))))
    return out


def evaluate(row: dict, noise_t: float) -> tuple[str, str]:
    """(verdict, reason) for one configuration, against the pre-registered bar."""
    t = row["t_daily"]
    if t is None or t != t:
        return "no signal", "not enough trading days to compute a daily t-statistic"
    if row["trades"] < MIN_TRADES:
        return "below the bar", f"{row['trades']} trades, needs {MIN_TRADES}"
    if row["positive_subperiods"] < MIN_POSITIVE_SUBPERIODS:
        return "below the bar", (f"positive in {row['positive_subperiods']} of "
                                 f"{SUBPERIODS} sub-periods, needs {MIN_POSITIVE_SUBPERIODS}")
    if t < MIN_T:
        return "below the bar", f"t {t:.2f}, needs {MIN_T}"
    if t <= noise_t:
        return "indistinguishable from noise", (
            f"t {t:.2f} does not exceed the {noise_t:.2f} that the best of this many "
            f"null trials reaches by chance")
    return "earns a forward test", (
        f"t {t:.2f} over {row['days']} sessions, {row['trades']} trades, "
        f"positive in {row['positive_subperiods']}/{SUBPERIODS} sub-periods, "
        f"above the {noise_t:.2f} noise ceiling at this trial count")


def sweep(contract, strategy, timeframe=5, start=None, end=None, ranges=None,
          costs: engine.Costs | None = None, on_progress=None,
          log_to_ledger=True) -> dict:
    """Run every combination, rank by the daily t-statistic, charge the ledger."""
    ranges = ranges or default_ranges(strategy)
    combos = grid(ranges)
    costs = costs or engine.Costs()
    ctx = engine.prepare(contract, timeframe, start, end)

    t0 = time.time()
    rows = []
    for i, p in enumerate(combos):
        trades, meta = engine.backtest(contract, strategy, timeframe, start, end,
                                       params=p, costs=costs, ctx=ctx)
        s = engine.summarise(trades, meta)
        subs = subperiod_pnl(trades)
        t_daily = s["t_daily"]
        rows.append(dict(
            params={k: p[k] for k in ranges},
            trades=s["trades"], days=s["days"], pnl=round(s["pnl"], 2),
            mean=round(s.get("mean", 0.0), 2), wr=round(s["wr"], 4),
            per_day=round(s["per_day"], 2),
            t_daily=(None if t_daily != t_daily else round(float(t_daily), 3)),
            subperiods=[round(v, 2) for v in subs],
            positive_subperiods=sum(1 for v in subs if v > 0)))
        if on_progress and (i % 5 == 0 or i == len(combos) - 1):
            on_progress(i + 1, len(combos), time.time() - t0)

    # Rank by the gate statistic among configurations that could clear the bar;
    # everything else sorts below it, still visible.
    def key(r):
        t = r["t_daily"]
        eligible = (t is not None and r["trades"] >= MIN_TRADES
                    and r["positive_subperiods"] >= MIN_POSITIVE_SUBPERIODS)
        return (1 if eligible else 0, t if t is not None else -99)
    rows.sort(key=key, reverse=True)

    before = ledger.stats()
    # Charged BEFORE any verdict is computed, so the winner is judged against a
    # trial count that already includes this sweep. Judging it against the count
    # before the sweep is how "best of 200" gets to look like one lucky idea.
    if log_to_ledger:
        w = rows[0] if rows else {}
        ledger.append("sweep", strategy=strategy, contract=contract,
                      fp=ledger.fingerprint(kind="sweep", strategy=strategy,
                                            contract=contract, tf=timeframe,
                                            start=ctx["start"], end=ctx["end"],
                                            ranges=ranges,
                                            slip=costs.slippage_ticks,
                                            comm=costs.commission),
                      timeframe=f"{timeframe}m",
                      start=ctx["start"], end=ctx["end"], days=ctx["days"],
                      n_trials=len(combos), combos=len(combos),
                      ranges={k: [float(v) for v in vs] for k, vs in ranges.items()},
                      best_params=w.get("params"), t_daily=w.get("t_daily"),
                      pnl=w.get("pnl"), trades=w.get("trades"),
                      slippage_ticks=costs.slippage_ticks, commission=costs.commission)
    after = ledger.stats()
    noise_t = after["expected_max_t"] if log_to_ledger else ledger.expected_max_t(
        before["trials"] + len(combos))

    for r in rows:
        r["verdict"], r["reason"] = evaluate(r, noise_t)

    return dict(strategy=strategy, label=engine.LIBRARY[strategy].label,
                contract=contract, timeframe=timeframe,
                combos=len(combos), ranges={k: list(v) for k, v in ranges.items()},
                elapsed=round(time.time() - t0, 1),
                noise_t=round(noise_t, 2),
                trials_before=before["trials"], trials_after=after["trials"],
                threshold_t=after["threshold_t"],
                slippage_ticks=costs.slippage_ticks, commission=costs.commission,
                rows=rows)


def ninjascript_block(res: dict, row: dict) -> str:
    """The winning values, as a comment block to paste into NinjaScript.

    The caveat travels WITH the numbers. A parameter set copied out of a results
    table arrives in NinjaTrader stripped of every qualification unless the
    qualification is inside the thing you copied.
    """
    p = row["params"]
    name = lambda k: "".join(w.capitalize() for w in k.split("_"))
    lines = [
        f"// PropSim sweep — {res['label']} on {res['contract']}, "
        f"{res['timeframe']}-minute bars",
        f"// {row['trades']} trades over {row['days']} sessions, "
        f"net {row['pnl']:,.0f}, daily t = "
        + ("n/a" if row["t_daily"] is None else f"{row['t_daily']:.2f}"),
        f"// Best of {res['combos']} combinations. Ledger trial "
        f"{res['trials_after']}; the best of that many null trials reaches "
        f"t = {res['noise_t']:.2f} by chance.",
        f"// VERDICT: {row['verdict'].upper()} — {row['reason']}.",
        "// NOT VALIDATED. These values were fitted on data already searched;",
        "// they have earned one forward test, not a funded account.",
        f"// Costs assumed: {res['slippage_ticks']:g} ticks/side slippage, "
        f"${res['commission']:g} per round trip.",
    ]
    lines += [f"{name(k)} = {v:g};" for k, v in p.items()]
    return "\n".join(lines)


def selfcheck():
    contracts = engine.tp.cached_contracts()
    if not contracts:
        print("no tape cached — run: python3 tape.py --build 'NQ 09-26'")
        return
    c = contracts[0]

    # 1. Range parsing and grid size.
    k, vals = parse_range("stop_ticks=20:60:10")
    assert k == "stop_ticks" and vals == [20, 30, 40, 50, 60], vals
    g = grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(g) == 6, g

    # 2. A prepared tape must give byte-identical results to a cold call, or the
    #    whole speedup is silently changing the answer.
    ctx = engine.prepare(c, 5)
    a, ma = engine.backtest(c, "orb", 5, ctx=ctx)
    b, mb = engine.backtest(c, "orb", 5)
    assert len(a) == len(b) and abs(sum(t.pnl for t in a) - sum(t.pnl for t in b)) < 1e-9, \
        (len(a), len(b))

    # 3. Sub-periods split by DAY and account for the whole run.
    trades, meta = engine.backtest(c, "fvg", 5, ctx=engine.prepare(c, 5))
    subs = subperiod_pnl(trades)
    assert len(subs) == SUBPERIODS
    assert abs(sum(subs) - sum(t.pnl for t in trades)) < 0.01, (subs, sum(subs))

    # 4. THE GATE. A configuration whose t does not clear the noise ceiling at the
    #    current trial count is never called a winner, however well it ranks.
    row = dict(t_daily=2.0, trades=120, days=28, positive_subperiods=3)
    v, _ = evaluate(row, noise_t=2.5)
    assert v == "indistinguishable from noise", v
    v, _ = evaluate(row, noise_t=1.2)
    assert v == "earns a forward test", v
    v, _ = evaluate(dict(row, trades=10), noise_t=1.2)
    assert v == "below the bar", v
    v, _ = evaluate(dict(row, positive_subperiods=1), noise_t=1.2)
    assert v == "below the bar", v
    v, _ = evaluate(dict(row, t_daily=1.2), noise_t=0.5)
    assert v == "below the bar", v

    # 5. A sweep charges the ledger once per COMBINATION, not once per sweep.
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="propsim-opt-")) / "trials.jsonl"
    ledger.append("sweep", path=tmp, strategy="x", n_trials=200, t_daily=2.0)
    s = ledger.stats(tmp)
    assert s["trials"] == 200, s
    assert 2.7 < s["expected_max_t"] < 2.8, s   # noise ceiling at 200 trials

    # 6. End to end, small, without touching the real ledger.
    res = sweep(c, "orb", 5, ranges={"stop_ticks": [30, 40], "rr": [1.5, 2.0]},
                log_to_ledger=False)
    assert res["combos"] == 4 and len(res["rows"]) == 4, res["combos"]
    assert all("verdict" in r for r in res["rows"])
    ts = [r["t_daily"] for r in res["rows"] if r["t_daily"] is not None]
    assert ts == sorted(ts, reverse=True) or len(set(ts)) == 1, ts
    snippet = ninjascript_block(res, res["rows"][0])
    assert "NOT VALIDATED" in snippet and "StopTicks" in snippet, snippet

    print(f"selfcheck OK: prepared tape matches a cold run exactly; sub-periods sum "
          f"to the total; the noise ceiling vetoes a t=2.0 winner at 200 trials "
          f"(ceiling {ledger.expected_max_t(200):.2f}); 4-combination sweep ran in "
          f"{res['elapsed']}s and every row carries a verdict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract")
    ap.add_argument("--strategy")
    ap.add_argument("--tf", type=int, default=5)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--range", action="append", default=[],
                    help="key=lo:hi:step or key=v1,v2,v3 (repeatable)")
    ap.add_argument("--points", type=int, default=4,
                    help="grid points per parameter when no --range is given")
    ap.add_argument("--slippage", type=float, default=2.0)
    ap.add_argument("--commission", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--json", action="store_true")
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
    if not (args.contract and args.strategy):
        raise SystemExit('usage: optimize.py --contract "NQ 09-26" --strategy orb')

    ranges = dict(parse_range(r) for r in args.range) if args.range \
        else default_ranges(args.strategy, args.points)
    costs = engine.Costs(slippage_ticks=args.slippage, commission=args.commission)

    def prog(done, total, secs):
        print(f"\r  {done}/{total} combinations ({secs:.0f}s)", end="", flush=True)

    res = sweep(args.contract, args.strategy, args.tf, args.start, args.end,
                ranges, costs, on_progress=prog, log_to_ledger=not args.no_ledger)
    print()
    if args.json:
        print(json.dumps(res, indent=1))
        return

    print(f"\n{res['label']} on {res['contract']}, {res['timeframe']}-minute bars — "
          f"{res['combos']} combinations in {res['elapsed']}s")
    print("ranges: " + ", ".join(f"{k}={v[0]:g}..{v[-1]:g}" for k, v in res["ranges"].items()))
    print(f"ledger: trial {res['trials_before']} -> {res['trials_after']}. "
          f"The best of {res['trials_after']} null trials reaches t = {res['noise_t']:.2f} "
          f"by chance; a 5% claim needs {res['threshold_t']:.2f}.")

    keys = list(res["ranges"])
    head = "".join(f"{k[:9]:>10}" for k in keys)
    print(f"\n{head}{'trades':>8}{'P&L':>10}{'avg':>7}{'t':>7}{'sub+':>6}   verdict")
    for r in res["rows"][:args.top]:
        vals = "".join(f"{r['params'][k]:>10g}" for k in keys)
        t = "—" if r["t_daily"] is None else f"{r['t_daily']:.2f}"
        print(f"{vals}{r['trades']:>8}{r['pnl']:>10,.0f}{r['mean']:>7,.0f}"
              f"{t:>7}{r['positive_subperiods']:>6}   {r['verdict']}")

    best = res["rows"][0]
    print(f"\nBest: {best['reason']}")
    print("\n" + ninjascript_block(res, best))


if __name__ == "__main__":
    main()
