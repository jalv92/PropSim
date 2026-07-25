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

HERE = Path(__file__).resolve().parent
PAGE = pr._res("dashboard.html")
PATHS_DRAWN = 1000          # spaghetti lines; the screenshot's reference count
UPDATE_RESULT = {}          # filled once at startup, shown in the UI


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
    if account:
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

    write("meta", dict(
        firm=firm, variant=variant, size=size, sims=sims, fee=fee,
        profile=profile, policy=policy_name,
        target=ev_rules.profit_target, max_dd=ev_rules.max_dd,
        start=ev_rules.start_balance, dd_lock=ev_rules.dd_floor_lock,
        hwm_basis=ev_rules.hwm_basis, breach_basis=ev_rules.breach_basis,
        warnings=ev_rules.warnings(), unmodeled=ev_rules.unmodeled_rules,
        retrieved=ev_rules.retrieved,
        pool=(None if not pool else dict(
            n_trades=pool["n_trades"], n_days=pool["n_days"],
            have_mae=pool["have_mae"], mean=round(pool["mean"], 2),
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
        funded=_funded_block(r["fu"], fu_rules),
    ))


def _funded_block(fu, rules):
    """The Funded view: what the account pays out, not whether it passes."""
    import numpy as _np
    fp = fu["paths"]
    if fp is None or not fp.size:
        return None
    last = int(_np.max(_np.where(~_np.isnan(fp).all(axis=0))[0]))
    fp = fp[:, :last + 1]
    inc = fu["income_path"]
    paid = inc > 0
    fpd = fu["first_payout_day"][paid]
    return dict(
        p_payout=float(fu["p_payout"]), p_bust=float(fu["p_bust"]),
        # three-way partition (see sim_funded): alive / paid-then-lost / lost-with-nothing
        p_alive=float((fu["outcome"] == 0).mean()),
        p_paid_then_lost=float((fu["outcome"] == 1).mean()),
        p_lost_nothing=float((fu["outcome"] == 2).mean()),
        mean_income=float(inc.mean()),
        median_income=float(_np.median(inc)),
        income_p5=float(_np.percentile(inc, 5)),
        income_p95=float(_np.percentile(inc, 95)),
        mean_payouts=float(fu["payouts"]),
        survive_days=float(fu["days"]),
        days_to_first=(None if not fpd.size else float(_np.nanmean(fpd))),
        buffer=rules.buffer_required, split=rules.profit_split,
        min_payout=rules.min_payout,
        hist=_hist(inc),
        paths=[[None if _np.isnan(v) else round(float(v), 1) for v in row]
               for row in fp],
        outcome=fu["outcome"][:fp.shape[0]].tolist(),
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
        if u.path == "/api/startup":
            import ntdata
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
            import ntdata
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

    def _sse(self, q):
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
                    raise SystemExit          # client navigated away mid-run
        try:
            run_stream(q, write)
        except SystemExit:
            return
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
