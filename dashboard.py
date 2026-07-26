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
import threading
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
import optimize
import slippage
import tape as tp
import engine

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


class ClientGone(Exception):
    """The browser closed mid-stream. Distinct from an application error.

    These used to share SystemExit, so a genuine user-facing failure -- "this
    contract has not been prepared yet" -- was swallowed as a disconnect and
    the page simply showed nothing. A silent failure is the worst kind: the
    user cannot tell a broken program from a slow one.
    """
# The most recent backtest, so the Prop Firm tab can score it without re-running.
LAST_BACKTEST: dict = {}


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
    tp.build_cache(contract, force=bool(q.get("force")), on_progress=prog)
    lo, hi, days = tp.available_range(contract)
    write("result", dict(contract=contract, first=lo, last=hi, days=days,
                         ticks=len(tp.load_cache(contract)["ts"])))


def backtest_stream(q, write):
    """Run one backtest and keep the trades for the Prop Firm tab."""
    contract = q.get("contract", [""])[0]
    strategy = q.get("strategy", ["orb"])[0]
    tf = int(q.get("tf", ["5"])[0])
    start = q.get("start", [""])[0] or None
    end = q.get("end", [""])[0] or None
    slip = float(q.get("slippage", ["2"])[0])
    comm = float(q.get("commission", ["5"])[0])
    params = {}
    S = engine.LIBRARY[strategy]
    for k in S.params:
        v = q.get("p_" + k, [None])[0]
        if v not in (None, ""):
            params[k] = float(v)

    write("progress", dict(stage="loading the tape"))
    costs = engine.Costs(commission=comm, slippage_ticks=slip)
    trades, meta = engine.backtest(contract, strategy, tf, start, end,
                                   params=params, costs=costs)
    write("progress", dict(stage="scoring"))
    summary = engine.summarise(trades, meta)

    # RECORDED BEFORE THE RESULT IS RENDERED, on purpose. A ledger written after
    # the chart appears is a ledger that loses every run the user abandons on
    # first glance -- which are exactly the runs that make the twentieth t-stat
    # meaningless.
    ledger.append("backtest", strategy=strategy, contract=contract,
                  # Same inputs, same number: an identical re-run is one look, not
                  # two. Without this, clicking Run twice on the same settings
                  # raised the user's own noise ceiling for nothing.
                  fp=ledger.fingerprint(kind="backtest", strategy=strategy,
                                        contract=contract, tf=tf,
                                        start=meta["start"], end=meta["end"],
                                        params=meta["params"],
                                        slip=slip, comm=comm),
                  timeframe=f"{tf}m", start=meta["start"], end=meta["end"],
                  params=meta["params"], trades=len(trades),
                  pnl=round(summary["pnl"], 2),
                  t_daily=(None if summary["t_daily"] != summary["t_daily"]
                           else round(summary["t_daily"], 3)),
                  slippage_ticks=slip, commission=comm)

    global LAST_BACKTEST
    LAST_BACKTEST = dict(trades=trades, meta=meta, summary=summary)

    equity, run = [], 0.0
    for t in trades:
        run += t.pnl
        equity.append(round(run, 2))
    write("result", dict(
        summary={k: (None if isinstance(v, float) and v != v else v)
                 for k, v in summary.items() if k != "params"},
        ledger=ledger.stats(),
        params=meta["params"], equity=equity,
        trades=[dict(date=t.date, dir=t.direction,
                     entry=round(t.entry_price, 2), exit=round(t.exit_price, 2),
                     pnl=round(t.pnl, 2), mae=round(t.mae, 2), reason=t.reason)
                for t in trades[:200]],
        n_shown=min(len(trades), 200)))


def optimize_stream(q, write):
    """Sweep a parameter grid and rank it against the pre-registered gate."""
    contract = q.get("contract", [""])[0]
    strategy = q.get("strategy", ["orb"])[0]
    tf = int(q.get("tf", ["5"])[0])
    start = q.get("start", [""])[0] or None
    end = q.get("end", [""])[0] or None
    costs = engine.Costs(slippage_ticks=float(q.get("slippage", ["2"])[0]),
                         commission=float(q.get("commission", ["5"])[0]))

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

    write("meta", dict(strategy=strategy, contract=contract, timeframe=tf,
                       combos=len(optimize.grid(ranges)),
                       ranges={k: list(v) for k, v in ranges.items()}))

    def prog(done, total, secs):
        write("progress", dict(done=done, total=total, secs=round(secs, 1),
                               pct=round(100 * done / max(total, 1), 1)))

    res = optimize.sweep(contract, strategy, tf, start, end, ranges, costs,
                         on_progress=prog)
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
    src = None
    fidelity = None
    if account.startswith("__import:"):
        run_id = account.split(":", 1)[1]
        path = next((p for p in ntimport.RUNS_DIR.glob("*.json")
                     if p.stem == run_id), None)
        if path is None:
            raise SystemExit(f"imported run {run_id!r} is no longer on disk")
        tl, im = ntimport.load_run(path)
        pool = nttrades.to_pool(tl)
        fidelity = im["fidelity"]
        src = (f"{im['n_trades']} trades from a NinjaTrader run: {im['strategy']} · "
               f"{im['instrument']} · {im['start']}..{im['end']} · {im['cost_basis']}")
        # A NinjaTrader run WAS a search: someone chose a strategy, a period and
        # a parameter set and kept the result. It counts as a trial the first time
        # it is scored here, and only once -- re-reading the same run is not a new
        # look at the market.
        if not any(r.get("run_id") == run_id for r in ledger.read()):
            ledger.append("import", run_id=run_id, strategy=im["strategy"],
                          contract=im["instrument"], timeframe=im["timeframe"],
                          start=im["start"], end=im["end"], params=im["params"],
                          trades=im["n_trades"], pnl=im["pnl"],
                          fidelity=fidelity["level"])
    elif account == "__backtest":
        if not LAST_BACKTEST.get("trades"):
            raise SystemExit("No backtest has been run yet — use the Backtest tab.")
        bt = LAST_BACKTEST
        pool = nttrades.to_pool(bt["trades"])
        m = bt["meta"]
        src = (f"{len(bt['trades'])} trades from a backtest: {m['label']} · "
               f"{m['contract']} · {m['timeframe']}m · {m['start']}..{m['end']}")
    elif account.startswith("__strategy:"):
        # One NinjaScript strategy's own trades, isolated from everything else on
        # the account. NinjaTrader records the attribution; nothing is inferred.
        sid = account.split(":", 1)[1]
        tl = nttrades.read_trades(strategy=sid)
        if not tl:
            raise SystemExit("that strategy has no recorded trades any more")
        pool = nttrades.to_pool(tl)
        name = tl[0].strategy or sid
        where = "Playback" if tl[0].is_replay else "live/sim"
        src = (f"{len(tl)} trades from strategy {name} ({where}, "
               f"account {tl[0].account})")
    elif account:
        # the user's own trades, straight out of NinjaTrader -- no export step
        tl = nttrades.read_trades(account=account)
        if tl:
            pool = nttrades.to_pool(tl)
            src = f"{len(tl)} real trades from account {account}"
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
                rows.append(dict(contract=name, ready=ready, days=days,
                                 first=lo or _fmt_day(v["first"]),
                                 last=hi or _fmt_day(v["last"]),
                                 mb=v["mb"], est_ticks=v["est_ticks"]))
            return self._send(200, json.dumps(dict(contracts=rows)).encode(),
                              "application/json")
        if u.path == "/api/strategies":
            lib = []
            for name, S in engine.LIBRARY.items():
                lib.append(dict(name=name, label=S.label, uses_ticks=S.uses_ticks,
                                params=[dict(key=k, default=v.default, lo=v.lo,
                                             hi=v.hi, desc=v.desc)
                                        for k, v in S.params.items()]))
            return self._send(200, json.dumps(dict(strategies=lib)).encode(),
                              "application/json")
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
        if u.path == "/api/backtest":
            return self._sse(parse_qs(u.query), backtest_stream)
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
        except SystemExit as exc:
            # engine/tape raise SystemExit for conditions the USER must see
            try:
                write("error", dict(message=str(exc) or "the run could not start"))
            except Exception:
                pass
        except Exception as exc:
            traceback.print_exc()
            try:
                write("error", dict(message=f"{type(exc).__name__}: {exc}"))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open a browser tab")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    if not PAGE.exists():
        raise SystemExit(f"missing {PAGE}")
    pr.load()                                  # fail fast if the table is absent
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
