#!/usr/bin/env python3
"""Local dashboard for the prop-firm Monte Carlo.

    python3 dashboard.py            # http://127.0.0.1:8765
    python3 dashboard.py --port 9000 --open

Serves one page on localhost. Nothing leaves the machine: no auth, no upload,
no external requests, no CDN. That is a licensing constraint, not a preference
-- CME 3.2(ii) forbids supplying this data to another party or location, so the
compute goes to the data and the page is a local view of a local process.

Stdlib only (plus numpy, already a dependency of the simulator). Runs stream
over Server-Sent Events so the equity chart fills in while the simulation is
still going, the way MetaTrader's tester does it.

    GET /                 the page
    GET /api/config       firm -> variant -> phase -> size tree, with warnings
    GET /api/run?...      SSE: progress events, then one result event
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import prop_rules as pr
from sim import PROFILES, POLICIES, load_trades, sim_attempt
import nttrades
import ntdata
import ntimport
import ledger
import nt8gen
import aiauthor
import optimize
import plugins
import slippage
import tape as tp
import continuous
import engine
import report
import branding

HERE = Path(__file__).resolve().parent
PAGE = pr._res("dashboard.html")
PATHS_DRAWN = 1000          # spaghetti lines; the screenshot's reference count

# A pass rate computed from a handful of trading days is not "uncertain" -- it is
# structurally meaningless. The bootstrap resamples whole DAYS, so a pool holding
# one profitable session produces ten thousand identical winning paths and reports
# P(pass) = 100%. Measured, on a real two-trade Playback session: 100.0% pass,
# $105,111 mean payout. Below MIN_DAYS the headline figure is withheld rather than
# rendered with a warning next to it, because a number on a screen outranks a
# caption every time. MEANINGFUL_DAYS is the separate, larger threshold at which
# the figure stops being a statement about the fortnight you happened to replay.
MIN_DAYS = 10
MEANINGFUL_DAYS = 60
UPDATE_RESULT = {}          # filled once at startup, shown in the UI
PLUGIN_REPORT: list = []    # user strategy files: loaded, or why not


class ClientGone(Exception):
    """The browser closed mid-stream. Distinct from an application error.

    These used to share SystemExit, so a genuine user-facing failure -- "this
    contract has not been prepared yet" -- was swallowed as a disconnect and
    the page simply showed nothing. A silent failure is the worst kind: the
    user cannot tell a broken program from a slow one.
    """


# The most recent backtest, so Risk & Monte Carlo can score it without re-running.
LAST_BACKTEST: dict = {}

# Sweeps the browser has asked to hold. A sweep runs inside its own request, so
# "pause" is the progress callback declining to return -- `optimize.sweep` calls
# it every few combinations and does not care how long it takes. That is why
# pausing needs no change to the sweep itself.
SWEEP_PAUSED: dict[str, bool] = {}


def config_tree():
    """Everything the picker needs, plus the honesty fields for each account."""
    tree = {}
    for rs in pr.load():
        f = tree.setdefault(rs.firm, {})
        v = f.setdefault(rs.variant, {})
        p = v.setdefault(rs.phase, {})
        p[str(rs.size)] = dict(
            profit_target=rs.profit_target, max_dd=rs.max_dd,
            hwm_basis=rs.hwm_basis, breach_basis=rs.breach_basis,
            dd_lock=rs.dd_floor_lock, start_balance=rs.start_balance,
            min_days=rs.min_days, max_contracts=rs.max_contracts,
            consistency_pct=rs.consistency_pct,
            consistency_effect=rs.consistency_effect,
            account_cost=rs.account_cost, profit_split=rs.profit_split,
            verified=rs.verified, retrieved=rs.retrieved,
            warnings=rs.warnings(), unmodeled=rs.unmodeled_rules,
        )
    return dict(firms=tree, profiles=PROFILES,
                policies=[p.__name__ for p in POLICIES])


def _hist(x, bins=28):
    counts, edges = np.histogram(x, bins=bins)
    return dict(counts=counts.tolist(),
                edges=[round(float(e), 2) for e in edges])


def _fmt_day(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def prepare_stream(q, write):
    """Build a contract's tick cache, reporting progress as it goes."""
    contract = q.get("contract", [""])[0]
    write("meta", dict(contract=contract))

    def prog(done, total, ticks):
        write("progress", dict(done=done, total=total, ticks=ticks,
                               pct=round(100 * done / max(total, 1), 1)))

    # ALL is stitched from the other caches rather than parsed from .ncd, and
    # the expensive half is re-parsing the contracts whose files changed --
    # concatenating what is already cached is seconds. So: refresh only what
    # moved, then restitch whole. No incremental state to go quietly wrong.
    # continuous.build() raises SystemExit with the audit's findings if the
    # stitch fails (e.g. a wrong-contract session, an unmeasurable roll) --
    # that propagates out of here uncaught, and _sse() turns it into an
    # "error" event the page renders, rather than a stack trace nobody sees.
    if contract == continuous.ALL:
        inv = ntdata.inventory(ntdata.resolve_root() or Path("."), force=True)
        for name, v in (inv.get("tick") or {}).items():
            if name.split()[0] != continuous.ROOT:
                continue
            p = tp.cache_path(name)
            fresh = p.exists()
            if fresh:
                try:
                    _, hi, _ = tp.available_range(name)
                    fresh = _fmt_day(v["last"]) <= hi
                except Exception:
                    fresh = False
            if not fresh:
                write("stage", dict(text=f"parsing {name}"))
                tp.build_cache(name, force=True, on_progress=prog)
        write("stage", dict(text="stitching ALL"))
        m = continuous.build(on_progress=prog)
        write("result", dict(contract=contract, first=m["first"], last=m["last"],
                             days=m["sessions"], ticks=m["ticks"]))
        return

    tp.build_cache(contract, force=bool(q.get("force")), on_progress=prog)
    lo, hi, days = tp.available_range(contract)
    write("result", dict(contract=contract, first=lo, last=hi, days=days,
                         ticks=len(tp.load_cache(contract)["ts"]),
                         truncated=tp.truncated_hours(contract)))


def _combos(firm_key, combos):
    """Firm/variant/phase/size combinations, each carrying its display name.

    The label and its note travel WITH the combination so the dropdown can say
    "LucidDaily — no daily loss limit, end-of-day eval drawdown" instead of
    `luciddaily_dlloff_eod`, and so an account a trader cannot buy says so
    where they are choosing.
    """
    out = []
    for v, ph, size in combos:
        label, note = branding.plan(firm_key, v)
        out.append(dict(variant=v, phase=ph, size=size, plan=label,
                        plan_note=note, phase_label=branding.phase(firm_key, ph)))
    return out


def _jsonable(v):
    """NaN and numpy scalars are not JSON. Nulls are, and render as an em dash."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def analyzer_stream(q, write):
    """One run, scored against one prop-firm account, both readings in one payload.

    The toggle in the browser does not re-run anything: re-running on a toggle
    would charge the ledger twice for one search and could hand back two
    different numbers for the same question.
    """
    contract = q.get("contract", [""])[0]
    strategy = q.get("strategy", ["orb"])[0]
    # NinjaTrader's own two fields. They arrive as a pair and become seconds here,
    # once, so nothing downstream has to know which unit the user picked.
    tf = tp.tf_secs(q.get("tf", ["5"])[0], q.get("bartype", ["Minute"])[0])
    start = q.get("start", [""])[0] or None
    end = q.get("end", [""])[0] or None
    slip = float(q.get("slippage", ["2"])[0])
    comm = float(q.get("commission", ["5"])[0])
    contracts = max(1, int(float(q.get("contracts", ["1"])[0])))
    rules = pr.select(q.get("firm", [""])[0], q.get("variant", [""])[0],
                      q.get("phase", [""])[0], int(q.get("size", ["50000"])[0]))
    params = {}
    S = engine.LIBRARY[strategy]
    for k in S.params:
        v = q.get("p_" + k, [None])[0]
        if v not in (None, ""):
            params[k] = float(v)

    # A TRADE SOURCE THAT IS NOT A PROPSIM STRATEGY. Your own NinjaTrader
    # account, one NinjaScript strategy's trades, or an imported run: the rules
    # engine does not care where a trade list came from, so neither does this.
    # These used to reach the Monte Carlo and nothing else, which meant the one
    # question the product exists to answer -- "would MY trading have passed" --
    # could only ever be answered as a projection, never as a verdict on what
    # actually happened.
    account = (q.get("account", [""])[0] or "").strip()
    if account:
        tl, src, fidelity = trade_source(account)
        if not tl:
            raise SystemExit("that trade source has no trades on it")
        days = sorted({t.date for t in tl})
        # Per-trade commission varies on a real account, and `_column` takes a
        # scalar. The mean keeps the identity gross+loss-commission == net exact
        # for the column as a whole, which is the property that table is checked on.
        comms = [getattr(t, "commission", 0.0) or 0.0 for t in tl]
        meta = dict(contract=(getattr(tl[0], "instrument", "") or "—"),
                    strategy=account, label=src, timeframe="—",
                    start=days[0], end=days[-1], days=len(days),
                    params={}, trades=len(tl), source=src,
                    commission=(sum(comms) / len(comms) if comms else 0.0),
                    slippage_ticks=None, external=True,
                    fidelity=(fidelity or {}).get("level"))
        # SCORED, NOT SEARCHED. Re-pricing a trade list you already own tells you
        # about prop-firm rules, not about whether an edge is real, so it is
        # recorded without inflating the multiplicity correction -- exactly what
        # the Monte Carlo side already did for the same list.
        ledger.append("score", firm=q.get("firm", [""])[0], source=src,
                      size=int(q.get("size", ["50000"])[0]), sims=20_000,
                      variant=q.get("variant", [""])[0], policy="analyzer")
        rep = report.build(tl, meta, rules, contracts=contracts, sims=20_000,
                           rng=np.random.default_rng(7))
        rep["t_daily"] = None
        rep["over_cap"] = bool(rules.max_contracts
                               and contracts > rules.max_contracts)
        rep["path_measured"] = report.path_measured(tl)
        return write("result", _jsonable(rep))

    write("progress", dict(stage="loading the tape"))
    costs = engine.Costs(commission=comm, slippage_ticks=slip)
    trades, meta = engine.backtest(contract, strategy, tf, start, end,
                                   params=params, costs=costs)

    # BEFORE the report is built: a ledger written after the result is rendered
    # loses exactly the runs a user abandons on first glance. `kind="backtest"`
    # keeps this run in the same search budget as every earlier one, including
    # the ones the retired Backtest tab recorded -- a separate kind would let
    # the same look be laundered into a fresh trial count.
    write("progress", dict(stage="recording the trial"))
    summary = engine.summarise(trades, meta)
    ledger.append("backtest", strategy=strategy, contract=contract,
                  fp=ledger.fingerprint(kind="backtest", strategy=strategy,
                                        contract=contract, tf=tf,
                                        start=meta["start"], end=meta["end"],
                                        params=meta["params"],
                                        slip=slip, comm=comm),
                  timeframe=tp.tf_label(tf), start=meta["start"], end=meta["end"],
                  params=meta["params"], trades=len(trades),
                  pnl=round(summary["pnl"], 2),
                  t_daily=(None if summary["t_daily"] != summary["t_daily"]
                           else round(summary["t_daily"], 3)),
                  slippage_ticks=slip, commission=comm)

    # The "__backtest" trade source reads this, on either screen. Both resolve
    # their sources through `trade_source`, so what one can score the other can.
    global LAST_BACKTEST
    LAST_BACKTEST = dict(trades=trades, meta=meta, summary=summary)

    write("progress", dict(stage="scoring against the account"))
    # A STRATEGY THAT SIZES ITSELF IS NOT RESIZED HERE. `contracts` on this
    # screen scales a per-contract trade list; a strategy declaring its own
    # `contracts` has already been simulated at that size -- its daily governor
    # had to be, since the thresholds are dollars -- so multiplying again would
    # book four lots of a four-lot run. The strategy's number wins, and the
    # account cap is checked against it rather than against the dial.
    acct_contracts = contracts
    if "contracts" in engine.LIBRARY[strategy].params:
        acct_contracts = 1
        contracts = int(meta["params"].get("contracts", 1))
    rep = report.build(trades, meta, rules, contracts=acct_contracts, sims=20_000,
                       rng=np.random.default_rng(7))
    rep["t_daily"] = (None if summary["t_daily"] != summary["t_daily"]
                      else round(summary["t_daily"], 3))
    # Reported, not enforced: sizing past the account's cap is the user's call,
    # but the report has to say the run no longer describes that account.
    rep["over_cap"] = bool(rules.max_contracts
                           and contracts > rules.max_contracts)
    rep["path_measured"] = report.path_measured(trades)
    write("result", _jsonable(rep))


def _reload_plugins():
    """Re-scan the strategy folder so a file written just now shows up."""
    global PLUGIN_REPORT
    PLUGIN_REPORT = plugins.register_all()
    return PLUGIN_REPORT


def ai_write_stream(q, write):
    """Have Claude write a strategy, streaming each attempt's verdict.

    Streamed rather than a plain GET because each attempt is a model call plus a run
    over real ticks: a minute is normal, three attempts is not unusual, and a spinner
    that says nothing for that long is indistinguishable from a hang.

    Nothing here is charged to the trial ledger -- and no P&L is sent to the browser.
    See aiauthor's module docstring: reporting the smoke test's numbers would turn
    generation into a free search over the user's data.
    """
    description = q.get("description", [""])[0]
    name = (q.get("name", [""])[0] or "").strip() or None
    contract = (q.get("contract", [""])[0] or "").strip() or None
    tf = tp.tf_secs(q.get("tf", ["5"])[0], q.get("bartype", ["Minute"])[0])
    write("meta", dict(model=aiauthor.model_name(),
                       max_attempts=aiauthor.MAX_ATTEMPTS,
                       key_source=aiauthor.key_source()))

    def on_attempt(a):
        write("attempt", dict(attempt=a["attempt"], ok=a["ok"], name=a.get("name"),
                              problems=aiauthor.redact(a.get("problems") or ""),
                              tokens=a["usage"]))

    res = aiauthor.write_strategy(description, name, contract=contract, tf_secs=tf,
                                  on_attempt=on_attempt)
    installed = None
    if res["ok"] and q.get("install"):
        installed = str(aiauthor.install(res, overwrite=bool(q.get("overwrite"))))
        _reload_plugins()
    write("result", dict(
        ok=res["ok"], source=res["source"], header=res["header"],
        info=res["info"], tokens=res["tokens"], model=res["model"],
        problems=aiauthor.redact(res["problems"] or ""),
        attempts=len(res["attempts"]), installed=installed,
        dir=str(plugins.USER_DIR)))


def ai_translate_stream(q, write):
    """Port a strategy to NinjaScript, streaming each compile attempt."""
    strategy = (q.get("strategy", [""])[0] or "").strip()
    params = {}
    for k, v in q.items():
        if k.startswith("p_") and v and v[0] != "":
            try:
                params[k[2:]] = float(v[0])
            except ValueError:
                pass
    if not params:
        S = engine.LIBRARY.get(strategy)
        if S is None:
            raise SystemExit(f"no strategy called {strategy!r}")
        params = {k: p.default for k, p in S.params.items()}
    meta = {k: (q.get(k, [""])[0] or None) for k in ("contract", "start", "end",
                                                     "timeframe", "verdict", "reason")}
    for k in ("trades", "days", "trials", "slippage_ticks"):
        try:
            meta[k] = int(float(q.get(k, [""])[0]))
        except (TypeError, ValueError):
            meta[k] = None
    for k in ("pnl", "t_daily", "noise_t"):
        try:
            meta[k] = float(q.get(k, [""])[0])
        except (TypeError, ValueError):
            meta[k] = None

    write("meta", dict(model=aiauthor.model_name(), strategy=strategy, params=params,
                       max_attempts=aiauthor.MAX_ATTEMPTS))

    def on_attempt(a):
        write("attempt", dict(attempt=a["attempt"], ok=a["ok"],
                              errors=[aiauthor.redact(e) for e in a.get("errors") or []],
                              warnings=a.get("warnings", 0)))

    res = aiauthor.translate(strategy, params, meta=meta, on_attempt=on_attempt)
    path = nt8gen.write_out(res) if res["compiles"] is not False else None
    write("result", dict(compiles=res["compiles"], source=res["source"],
                         class_name=res["class_name"], notes=res["notes"],
                         attempts=res["attempts"], tokens=res["tokens"],
                         model=res["model"], note=res.get("note"),
                         path=str(path) if path else None, dir=str(nt8gen.OUT_DIR)))


def optimize_stream(q, write):
    """Sweep a parameter grid and rank it against the pre-registered gate."""
    run_id = (q.get("id", [""])[0] or "")[:64]
    contract = q.get("contract", [""])[0]
    strategy = q.get("strategy", ["orb"])[0]
    tf = tp.tf_secs(q.get("tf", ["5"])[0], q.get("bartype", ["Minute"])[0])
    start = q.get("start", [""])[0] or None
    end = q.get("end", [""])[0] or None
    costs = engine.Costs(slippage_ticks=float(q.get("slippage", ["2"])[0]),
                         commission=float(q.get("commission", ["5"])[0]))
    # The account is a CONSTRAINT on the search, not its objective -- see
    # `optimize.account_check`. Optional: with no firm chosen the sweep behaves
    # exactly as it did, so an existing bookmark keeps working.
    op_contracts = max(1, int(float(q.get("contracts", ["1"])[0])))
    rules = None
    if q.get("firm", [""])[0]:
        try:
            rules = pr.select(q.get("firm", [""])[0], q.get("variant", [""])[0],
                              q.get("phase", [""])[0],
                              int(q.get("size", ["50000"])[0]))
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"that account does not exist: {exc}")

    ranges = {}
    S = engine.LIBRARY[strategy]
    for k in S.params:
        spec = q.get("r_" + k, [""])[0]
        if not spec.strip():
            continue
        _, vals = optimize.parse_range(f"{k}={spec.strip()}")
        if vals:
            ranges[k] = vals
    if not ranges:
        raise SystemExit("give at least one parameter a range to sweep")

    SWEEP_PAUSED.setdefault(run_id, False)
    write("meta", dict(strategy=strategy, contract=contract,
                       timeframe=tp.tf_label(tf), tf_secs=tf,
                       combos=len(optimize.grid(ranges)),
                       account=(None if rules is None else dict(
                           label=f"{rules.firm}/{rules.variant}/{rules.phase}"
                                 f"/${rules.size:,}",
                           max_dd=rules.max_dd, contracts=op_contracts,
                           max_contracts=rules.max_contracts,
                           over_cap=bool(rules.max_contracts
                                         and op_contracts > rules.max_contracts))),
                       ranges={k: list(v) for k, v in ranges.items()}))

    def prog(done, total, secs):
        write("progress", dict(done=done, total=total, secs=round(secs, 1),
                               pct=round(100 * done / max(total, 1), 1)))
        # Holding here holds the sweep. The heartbeat is not decoration: it is
        # the only thing that notices a browser that went away while paused,
        # because a paused run makes no other writes and would otherwise sit on
        # a thread for as long as the process lives.
        while SWEEP_PAUSED.get(run_id):
            write("progress", dict(done=done, total=total, secs=round(secs, 1),
                                   pct=round(100 * done / max(total, 1), 1),
                                   paused=True))
            time.sleep(1.0)

    try:
        res = optimize.sweep(contract, strategy, tf, start, end, ranges, costs,
                             on_progress=prog, rules=rules,
                             contracts=op_contracts)
    finally:
        # However this ends -- finished, stopped, or the browser vanished --
        # the run stops being pausable.
        SWEEP_PAUSED.pop(run_id, None)
    best = res["rows"][0] if res["rows"] else None
    res["export"] = optimize.ninjascript_block(res, best) if best else ""
    # The bar can be structurally unreachable on this data -- a one-trade-a-day
    # strategy over 28 sessions can never show 80 trades, no matter the
    # parameters. Say that instead of letting every row read "below the bar" as
    # if the parameters were the problem.
    if best and all(r["trades"] < optimize.MIN_TRADES for r in res["rows"]):
        res["unreachable"] = (
            f"No configuration can clear the bar on this data: the best of them "
            f"takes {max(r['trades'] for r in res['rows'])} trades and the gate "
            f"needs {optimize.MIN_TRADES}. That is a sample-size problem, not a "
            f"parameter problem — widen the date range or use a strategy that "
            f"trades more often.")
    res["rows"] = res["rows"][:200]
    write("result", res)


def trade_source(account: str):
    """Resolve what `#account` names into a real trade LIST.

    ONE resolver, because the two halves of this app used to disagree about what
    a trade source even is: the Monte Carlo had four of them and the Analyzer had
    none, which is why a real NinjaTrader account could be *simulated* but never
    *scored* against the rules it actually traded under. Both go through here now,
    so a source that works on one screen works on the other by construction.

    Returns (trades, label, fidelity). `trades` is None when `account` names
    nothing -- an empty picker, or the parametric profile the Monte Carlo falls
    back to.
    """
    account = (account or "").strip()
    if not account:
        return None, None, None

    if account.startswith("__import:"):
        run_id = account.split(":", 1)[1]
        path = next((p for p in ntimport.RUNS_DIR.glob("*.json")
                     if p.stem == run_id), None)
        if path is None:
            raise SystemExit(f"imported run {run_id!r} is no longer on disk")
        tl, im = ntimport.load_run(path)
        # A NinjaTrader run WAS a search: someone chose a strategy, a period and
        # a parameter set and kept the result. It counts as a trial the first time
        # it is scored here, and only once -- re-reading the same run is not a new
        # look at the market. Charged in the resolver so it cannot depend on which
        # screen the user happened to open it from.
        if not any(r.get("run_id") == run_id for r in ledger.read()):
            ledger.append("import", run_id=run_id, strategy=im["strategy"],
                          contract=im["instrument"], timeframe=im["timeframe"],
                          start=im["start"], end=im["end"], params=im["params"],
                          trades=im["n_trades"], pnl=im["pnl"],
                          fidelity=im["fidelity"]["level"])
        return tl, (f"{im['n_trades']} trades from a NinjaTrader run: "
                    f"{im['strategy']} · {im['instrument']} · "
                    f"{im['start']}..{im['end']} · {im['cost_basis']}"), im["fidelity"]

    if account == "__backtest":
        if not LAST_BACKTEST.get("trades"):
            raise SystemExit("No backtest has been run yet.")
        bt = LAST_BACKTEST
        m = bt["meta"]
        return bt["trades"], (f"{len(bt['trades'])} trades from a backtest: "
                              f"{m['label']} · {m['contract']} · {m['timeframe']} · "
                              f"{m['start']}..{m['end']}"), None

    if account.startswith("__strategy:"):
        # One NinjaScript strategy's own trades, isolated from everything else on
        # the account. NinjaTrader records the attribution; nothing is inferred.
        sid = account.split(":", 1)[1]
        tl = nttrades.read_trades(strategy=sid)
        if not tl:
            raise SystemExit("that strategy has no recorded trades any more")
        where = "Playback" if tl[0].is_replay else "live/sim"
        return tl, (f"{len(tl)} trades from strategy {tl[0].strategy or sid} "
                    f"({where}, account {tl[0].account})"), None

    # the user's own trades, straight out of NinjaTrader -- no export step
    tl = nttrades.read_trades(account=account)
    if not tl:
        return None, None, None
    return tl, f"{len(tl)} real trades from account {account}", None


def run_stream(q, write):
    """Execute one simulation, streaming progress then the final payload."""
    firm = q.get("firm", ["my_funded_futures"])[0]
    variant = q.get("variant", ["rapid"])[0]
    size = int(q.get("size", ["50000"])[0])
    sims = max(100, min(int(q.get("sims", ["10000"])[0]), 100_000))
    fee = float(q.get("fee", ["0"])[0])
    profile = q.get("profile", ["modest edge 1:1"])[0]
    policy_name = q.get("policy", ["fixed-3"])[0]

    ev_rules = pr.select(firm, variant, "evaluation", size)
    try:
        fu_rules = pr.select(firm, variant, "funded_sim", size)
    except KeyError:
        # LucidDirect has no evaluation, LucidMaxx no funded-sim: the phase
        # chain is per variant, so fall back rather than inventing a phase.
        fu_rules = ev_rules
    if fee == 0 and ev_rules.account_cost:
        fee = ev_rules.account_cost

    trades = (q.get("trades", [""])[0] or "").strip()
    account = (q.get("account", [""])[0] or "").strip()
    pool = None
    tl, src, fidelity = trade_source(account)
    if tl:
        pool = nttrades.to_pool(tl)
    elif trades:
        pool = load_trades(trades, friction=float(q.get("tfric", ["5"])[0]))
        src = f"trade list {trades}"
    prof = PROFILES[profile] if profile in PROFILES else PROFILES["modest edge 1:1"]
    if pool:
        # a real trade list replaces the Bernoulli entirely: day blocks carry
        # the within-day structure every prop-firm rule depends on
        prof = dict(prof, pool=pool)
    policy = next((p for p in POLICIES if p.__name__ == policy_name), POLICIES[1])

    # Scoring is logged so the record is complete, but it does NOT inflate the
    # multiplicity correction: re-pricing a trade list you already have tells you
    # about prop-firm rules, not about whether the edge is real.
    ledger.append("score", firm=firm, variant=variant, size=size, sims=sims,
                  source=(src or profile), policy=policy_name)

    thin = None
    if pool:
        n_days = int(pool["n_days"])
        distinct = int(pool.get("distinct_days") or n_days)
        if min(n_days, distinct) < MIN_DAYS:
            thin = dict(n_days=n_days, distinct=distinct,
                        needed=MIN_DAYS, meaningful=MEANINGFUL_DAYS)

    write("meta", dict(
        firm=firm, variant=variant, size=size, sims=sims, fee=fee,
        profile=profile, policy=policy_name, fidelity=fidelity,
        insufficient=thin,
        ledger=ledger.stats(),
        target=ev_rules.profit_target, max_dd=ev_rules.max_dd,
        start=ev_rules.start_balance, dd_lock=ev_rules.dd_floor_lock,
        hwm_basis=ev_rules.hwm_basis, breach_basis=ev_rules.breach_basis,
        warnings=ev_rules.warnings(), unmodeled=ev_rules.unmodeled_rules,
        retrieved=ev_rules.retrieved,
        pool=(None if not pool else dict(
            n_trades=pool["n_trades"], n_days=pool["n_days"],
            have_mae=pool["have_mae"], mean=round(pool["mean"], 2),
            distinct_days=pool.get("distinct_days"),
            trades_per_day=round(pool.get("trades_per_day", 0), 1),
            source=src or pool["source"])),
    ))

    def on_day(day, p_pass, p_bust, bal):
        write("progress", dict(day=day, p_pass=float(p_pass),
                               p_bust=float(p_bust),
                               median=float(np.median(bal))))

    rng = np.random.default_rng(7)
    r = sim_attempt(prof, policy, sims, rng, ev_rules, fu_rules, fee,
                    record_paths=PATHS_DRAWN, on_day=on_day)

    ev = r["ev"]
    paths = ev["paths"]
    # trim the all-NaN tail so the client does not draw 90 empty days
    last = int(np.max(np.where(~np.isnan(paths).all(axis=0))[0])) if paths.size else 0
    paths = paths[:, :last + 1]
    outcome = ev["outcome"][:paths.shape[0]].tolist()

    write("result", dict(
        p_pass=float(ev["p_pass"]), p_bust=float(ev["p_bust"]),
        p_timeout=float(ev["p_timeout"]),
        days_pass=None if np.isnan(ev["days"]) else float(ev["days"]),
        days_bust=None if np.isnan(ev["bust_days"]) else float(ev["bust_days"]),
        net_mean=r["net_mean"], net_p5=r["net_p5"], net_p95=r["net_p95"],
        p_payout=r["p_payout"], mean_payout=r["mean_payout"],
        days_to_first_payout=(None if np.isnan(r["days_to_first_payout"])
                              else r["days_to_first_payout"]),
        pnl_median=float(np.median(ev["final"])),
        pnl_p5=float(np.percentile(ev["final"], 5)),
        pnl_p95=float(np.percentile(ev["final"], 95)),
        pct_losing=float((ev["final"] < 0).mean()),
        hist=_hist(ev["final"]),
        paths=[[None if np.isnan(v) else round(float(v), 1) for v in row]
               for row in paths],
        outcome=outcome,
        n_days=paths.shape[1],
        funded=_funded_block(r["fu"], fu_rules, r["passed"] if r["chained"] else None),
    ))


def _funded_block(fu, rules, reached=None):
    """The Funded view: what the account pays out, not whether it passes.

    `reached` is the mask of paths that actually got here. With the phases
    chained, the paths that never passed spent zero days in the funded account,
    so averaging over ALL of them would report a payout rate diluted by accounts
    that never existed. Every figure here is therefore conditional on being
    funded -- which is the question this tab asks.
    """
    import numpy as _np
    fp = fu["paths"]
    if fp is None or not fp.size:
        return None
    last = int(_np.max(_np.where(~_np.isnan(fp).all(axis=0))[0]))
    fp = fp[:, :last + 1]

    m = _np.ones(len(fu["income_path"]), bool) if reached is None else _np.asarray(reached)
    if not m.any():
        return None
    inc = fu["income_path"][m]
    out = fu["outcome"][m]
    paid = inc > 0
    fpd = fu["first_payout_day"][m][paid]
    # the drawn curves belong to specific sims, so their colours must be looked
    # up by index rather than by position
    idx = fu.get("path_idx")
    colours = (fu["outcome"][_np.asarray(idx, int)] if idx is not None
               else fu["outcome"][:fp.shape[0]])
    return dict(
        p_payout=float(paid.mean()), p_bust=float((out != 0).mean()),
        # three-way partition (see sim_funded): alive / paid-then-lost / lost-with-nothing
        p_alive=float((out == 0).mean()),
        p_paid_then_lost=float((out == 1).mean()),
        p_lost_nothing=float((out == 2).mean()),
        conditional=bool(reached is not None),
        n_funded=int(m.sum()),
        mean_income=float(inc.mean()),
        median_income=float(_np.median(inc)),
        income_p5=float(_np.percentile(inc, 5)),
        income_p95=float(_np.percentile(inc, 95)),
        mean_payouts=float(fu["payouts"] * len(fu["income_path"]) / max(int(m.sum()), 1)
                           if reached is not None else fu["payouts"]),
        survive_days=float(fu["survive_days"][m].mean()),
        days_to_first=(None if not fpd.size else float(_np.nanmean(fpd))),
        buffer=rules.buffer_required, split=rules.profit_split,
        min_payout=rules.min_payout,
        hist=_hist(inc),
        paths=[[None if _np.isnan(v) else round(float(v), 1) for v in row]
               for row in fp],
        outcome=[int(v) for v in colours[:fp.shape[0]]],
        n_days=fp.shape[1],
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                   # quiet; this is a local tool

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if u.path.startswith("/assets/logos/"):
            # Served from disk, never fetched: this page makes no external
            # requests by design (see the module docstring). A missing file is
            # a 404 and the panel falls back to the firm's monogram, so an
            # absent asset degrades to something readable.
            name = Path(u.path).name
            f = HERE / "assets" / "logos" / name
            if name in ("", ".", "..") or "/" in name or not f.is_file():
                return self._send(404, b"", "text/plain")
            mime = ("image/svg+xml" if f.suffix == ".svg"
                    else "image/png" if f.suffix == ".png" else "application/octet-stream")
            return self._send(200, f.read_bytes(), mime)
        if u.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if u.path == "/api/tape":
            inv = ntdata.inventory(ntdata.resolve_root() or Path("."))
            rows = []
            for name, v in (inv.get("tick") or {}).items():
                ready = tp.cache_path(name).exists()
                lo = hi = None; days = v["days"]
                if ready:
                    try:
                        lo, hi, days = tp.available_range(name)
                    except Exception:
                        ready = False
                # THE CACHE IS BUILT ONCE AND DOES NOT NOTICE NEW FILES. Reporting
                # only its range makes a contract with 44 days on disk look like a
                # contract with 34: the backtester then refuses dates the user
                # knows they downloaded, and nothing on the page explains why.
                # Both ranges go out, and `stale` is the difference stated plainly.
                raw_first, raw_last = _fmt_day(v["first"]), _fmt_day(v["last"])
                stale = bool(ready and hi and raw_last > hi)
                rows.append(dict(contract=name, ready=ready, days=days,
                                 first=lo or raw_first,
                                 last=hi or raw_last,
                                 raw_first=raw_first, raw_last=raw_last,
                                 raw_days=v["days"], stale=stale,
                                 mb=v["mb"], est_ticks=v["est_ticks"]))
            # ALL is not a file under db/tick, so it never appears in `rows` --
            # it is reported separately, from its own sidecar (continuous.py),
            # with the coverage holes that a stitch across missing months
            # leaves. Buried holes are how someone backtests April-June and
            # gets a fraction of the trades they expect with nothing on
            # screen saying why.
            stale, why = continuous.is_stale()
            m = continuous.load_meta()
            allrow = dict(
                ready=bool(m and continuous.cache_path().exists()),
                stale=stale, why=why,
                sessions=(m or {}).get("sessions", 0),
                first=(m or {}).get("first"), last=(m or {}).get("last"),
                ticks=(m or {}).get("ticks", 0),
                rolls=(m or {}).get("rolls", []),
                holes=(m or {}).get("holes", []),
                sources=[dict(contract=c, truncated=tp.truncated_hours(c))
                         for c in continuous.contracts_on_disk()])
            return self._send(200, json.dumps(dict(contracts=rows,
                                                   all=allrow)).encode(),
                              "application/json")
        if u.path == "/api/strategies":
            lib = []
            mine = set(plugins.installed())
            for name, S in engine.LIBRARY.items():
                lib.append(dict(name=name, label=S.label, uses_ticks=S.uses_ticks,
                                mine=name in mine,
                                params=[dict(key=k, default=v.default, lo=v.lo,
                                             hi=v.hi, desc=v.desc,
                                             fixed=getattr(v, "fixed", False))
                                        for k, v in S.params.items()]))
            return self._send(200, json.dumps(
                dict(strategies=lib, plugins=PLUGIN_REPORT,
                     dir=str(plugins.USER_DIR))).encode(), "application/json")
        if u.path == "/api/plugins/reload":
            _reload_plugins()
            return self._send(200, json.dumps(
                dict(plugins=PLUGIN_REPORT, dir=str(plugins.USER_DIR),
                     installed=plugins.installed())).encode(), "application/json")
        if u.path == "/api/ntstrategies":
            try:
                rows = nttrades.detected_strategies()
            except SystemExit as exc:
                return self._send(200, json.dumps(
                    dict(strategies=[], error=str(exc))).encode(), "application/json")
            return self._send(200, json.dumps(dict(strategies=rows)).encode(),
                              "application/json")
        if u.path == "/api/imports":
            try:
                runs = ntimport.list_runs()
            except Exception as exc:
                return self._send(200, json.dumps(
                    dict(runs=[], error=f"{type(exc).__name__}: {exc}")).encode(),
                    "application/json")
            return self._send(200, json.dumps(dict(
                runs=runs, dir=str(ntimport.RUNS_DIR))).encode(), "application/json")
        if u.path == "/api/nt8gen":
            q = parse_qs(u.query)
            strategy = (q.get("strategy", [""])[0] or "").strip()
            params = {}
            for k, v in q.items():
                if k.startswith("p_") and v and v[0] != "":
                    try:
                        params[k[2:]] = float(v[0])
                    except ValueError:
                        pass
            meta = {k: (q.get(k, [""])[0] or None) for k in
                    ("contract", "start", "end", "timeframe", "verdict", "reason")}
            for k in ("trades", "days", "trials", "slippage_ticks"):
                try:
                    meta[k] = int(float(q.get(k, [""])[0]))
                except (TypeError, ValueError):
                    meta[k] = None
            for k in ("pnl", "t_daily", "noise_t"):
                try:
                    meta[k] = float(q.get(k, [""])[0])
                except (TypeError, ValueError):
                    meta[k] = None
            try:
                res = nt8gen.generate(strategy, params, meta=meta)
                path = nt8gen.write_out(res) if res["compiles"] is not False else None
                out = dict(res, path=str(path) if path else None,
                           dir=str(nt8gen.OUT_DIR))
            except nt8gen.GenError as exc:
                out = dict(error=str(exc), available=nt8gen.available())
            except Exception as exc:
                out = dict(error=f"{type(exc).__name__}: {exc}")
            return self._send(200, json.dumps(out).encode(), "application/json")
        if u.path == "/api/nt8gen/templates":
            return self._send(200, json.dumps(
                dict(available=nt8gen.available())).encode(), "application/json")
        if u.path == "/api/ai":
            # Whether the feature is usable, and never the key itself: this response
            # is the one thing on this server a screenshot is likely to include.
            try:
                body = aiauthor.status()
            except Exception as exc:
                body = dict(error=aiauthor.redact(f"{type(exc).__name__}: {exc}"))
            return self._send(200, json.dumps(body).encode(), "application/json")
        if u.path == "/api/ai/write":
            return self._sse(parse_qs(u.query), ai_write_stream)
        if u.path == "/api/ai/translate":
            return self._sse(parse_qs(u.query), ai_translate_stream)
        if u.path == "/api/slippage":
            q = parse_qs(u.query)
            c = (q.get("contract", [""])[0] or "").strip()
            try:
                r = slippage.measure(c, files=int(q.get("files", ["8"])[0]))
            except SystemExit as exc:
                r = dict(error=str(exc))
            except Exception as exc:
                r = dict(error=f"{type(exc).__name__}: {exc}")
            return self._send(200, json.dumps(r).encode(), "application/json")
        if u.path == "/api/ledger":
            body = json.dumps(dict(stats=ledger.stats(), verify=ledger.verify(),
                                   recent=ledger.recent(30))).encode()
            return self._send(200, body, "application/json")
        if u.path == "/api/tape/prepare":
            return self._sse(parse_qs(u.query), prepare_stream)
        if u.path == "/api/analyzer":
            return self._sse(parse_qs(u.query), analyzer_stream)
        if u.path == "/api/rules":
            q = parse_qs(u.query)
            firm = q.get("firm", [""])[0]
            # Firms and plans go out under the names their own sites use, from
            # `branding.py`. The slugs stay the keys; only the labels change.
            firm_list = [dict(key=k, **branding.firm(k)) for k in pr.firms()]
            if not firm:
                return self._send(200, json.dumps(dict(
                    firms=firm_list)).encode(), "application/json")
            variant = q.get("variant", [""])[0]
            phase = q.get("phase", [""])[0]
            size = q.get("size", [""])[0]
            combos = pr.variants(firm)
            if not (variant and phase and size):
                # The dropdowns cascade from what EXISTS. LucidDirect has no
                # evaluation phase and LucidMaxx has no funded-sim one, so
                # offering the cross product would let the user build a request
                # that cannot be answered.
                return self._send(200, json.dumps(dict(
                    firms=firm_list, combos=_combos(firm, combos))).encode(),
                    "application/json")
            try:
                rs = pr.select(firm, variant, phase, int(size))
            except (KeyError, ValueError) as exc:
                return self._send(200, json.dumps(dict(
                    firms=firm_list, combos=_combos(firm, combos),
                    error=str(exc))).encode(), "application/json")
            body = json.dumps(dict(
                firms=firm_list, combos=_combos(firm, combos),
                display=branding.describe(rs),
                rules=dict(
                    firm=rs.firm, variant=rs.variant, phase=rs.phase,
                    size=rs.size, start_balance=rs.start_balance,
                    max_dd=rs.max_dd, hwm_basis=rs.hwm_basis,
                    breach_basis=rs.breach_basis,
                    dd_floor_lock=rs.dd_floor_lock,
                    profit_target=rs.profit_target,
                    daily_loss_limit=rs.daily_loss_limit,
                    max_contracts=rs.max_contracts,
                    automation_allowed=rs.automation_allowed,
                    # THE LIST, not only the count. `warnings()` says "3 rule(s)
                    # not modelled by the simulator" and stopped there, which is
                    # half a sentence: three administrative formalities and three
                    # rules that drive the trailing drawdown read identically at
                    # that resolution. On this same firm, Builder's three cannot
                    # move a verdict while Rapid's include "intraday trailing on
                    # equity incl. unrealized" -- which is the whole mechanism.
                    # A product whose case is that it says what it does not model
                    # has to actually say it.
                    unmodeled=list(rs.unmodeled_rules or []),
                    warnings=rs.warnings()))).encode()
            return self._send(200, body, "application/json")
        if u.path == "/api/optimize/pause":
            q = parse_qs(u.query)
            rid = (q.get("id", [""])[0] or "")[:64]
            on = q.get("on", ["0"])[0] == "1"
            # Only a run that is actually streaming can be held. Setting the
            # flag for an unknown id would leave an entry nothing ever clears.
            known = rid in SWEEP_PAUSED
            if known:
                SWEEP_PAUSED[rid] = on
            return self._send(200, json.dumps(
                dict(id=rid, paused=on and known, known=known)).encode(),
                "application/json")
        if u.path == "/api/optimize":
            return self._sse(parse_qs(u.query), optimize_stream)
        if u.path == "/api/startup":
            cfg = ntdata.load_config()
            q = parse_qs(u.query)
            if q.get("accept"):
                cfg = ntdata.save_config(dict(cfg, disclaimer_accepted=True))
            body = json.dumps(dict(
                accepted=bool(cfg.get("disclaimer_accepted")),
                rules=pr.rules_origin(),
                update=UPDATE_RESULT,
                configured=bool(cfg.get("nt_root")),
            )).encode()
            return self._send(200, body, "application/json")
        if u.path == "/api/accounts":
            try:
                accts = nttrades.detected_accounts()
            except SystemExit as exc:
                accts = []
                return self._send(200, json.dumps(
                    dict(accounts=[], error=str(exc))).encode(), "application/json")
            return self._send(200, json.dumps(dict(accounts=accts)).encode(),
                              "application/json")
        if u.path == "/api/datasource":
            q = parse_qs(u.query)
            root = (q.get("root", [""])[0] or "").strip()
            if root:
                ntdata.save_config(dict(ntdata.load_config(), nt_root=root))
            r = ntdata.resolve_root(root or None)
            force = bool(q.get("force"))
            body = json.dumps(ntdata.inventory(r, force=force) if r else
                              dict(root=None, ok=False, tick={}, replay={},
                                   notes=["No NinjaTrader 8 folder found - "
                                          "enter the path to it."])).encode()
            return self._send(200, body, "application/json")
        if u.path == "/api/config":
            body = json.dumps(config_tree()).encode()
            return self._send(200, body, "application/json")
        if u.path == "/api/run":
            return self._sse(parse_qs(u.query))
        self._send(404, b"not found", "text/plain")

    def _sse(self, q, runner=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        lock = threading.Lock()

        def write(event, payload):
            with lock:
                msg = f"event: {event}\ndata: {json.dumps(payload)}\n\n"
                try:
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    raise ClientGone          # client navigated away mid-run
        try:
            (runner or run_stream)(q, write)
        except ClientGone:
            return
        except (SystemExit, aiauthor.AuthorError) as exc:
            # engine/tape raise SystemExit, aiauthor raises AuthorError, for conditions
            # the USER must see. Both are already prose; neither needs a class name.
            try:
                write("error", dict(message=aiauthor.redact(exc) or
                                    "the run could not start"))
            except Exception:
                pass
        except Exception as exc:
            traceback.print_exc()
            try:
                # Redacted because an SDK-level HTTP error can quote the request it
                # failed on, and this string is rendered into a browser.
                write("error", dict(
                    message=aiauthor.redact(f"{type(exc).__name__}: {exc}")))
            except Exception:
                pass


FROZEN = getattr(__import__("sys"), "frozen", False)


def _free_port(preferred):
    """First free port at or after `preferred`.

    A packaged app cannot assume 8765 is free -- on this machine something else
    already held it during development -- and a windowed build has no console to
    report the bind error on, so it would just fail silently.
    """
    import socket
    for p in range(preferred, preferred + 40):
        with socket.socket() as s:
            # Match what the real server can do. ThreadingHTTPServer sets
            # allow_reuse_address, so it binds a port whose old connections are
            # still in TIME_WAIT -- a probe without SO_REUSEADDR does not, and the
            # app then hopped to the next port on every restart that had a browser
            # attached. The URL moving under the user is worse than the collision
            # this was guarding against.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


def main():
    # MUST COME FIRST, and it is not optional in the packaged build. A sweep
    # starts worker processes with "spawn", and under PyInstaller a spawned
    # child re-executes the frozen EXE -- which without this call means each
    # worker starts a whole new PropSim, which starts its own workers. The
    # symptom is a process bomb on a user's machine, not an error message here.
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open a browser tab")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    if not PAGE.exists():
        raise SystemExit(f"missing {PAGE}")
    pr.load()                                  # fail fast if the table is absent
    # Your own strategies, and any an AI wrote for you, from
    # ~/.prop-sim/strategies/. Each is validated before it is trusted; a rejected
    # file is reported in the UI rather than silently ignored, because a strategy
    # that quietly does not exist is indistinguishable from one that finds nothing.
    global PLUGIN_REPORT
    PLUGIN_REPORT = plugins.register_all()
    ok = [r for r in PLUGIN_REPORT if r.get("ok")]
    if PLUGIN_REPORT:
        print(f"strategies loaded from {plugins.USER_DIR}: {len(ok)} ok, "
              f"{len(PLUGIN_REPORT) - len(ok)} rejected")
    # Refresh the rule file in the background. Startup must never wait on the
    # network, and an offline user must still get the bundled rules.
    def _refresh():
        global UPDATE_RESULT
        UPDATE_RESULT = pr.update_rules()
    threading.Thread(target=_refresh, daemon=True).start()
    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"PropSim on {url}   (ctrl-c to stop)")
    # A windowed build has no console, so the browser IS the UI: open it unless
    # explicitly told not to.
    if (args.open or FROZEN) and not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
