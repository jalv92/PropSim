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
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import engine
import ledger
import sim
import tape as tp

# The pre-registered bar: a hypothesis earns a forward test at train t >= 1.5,
# N >= 80 trades, and positive in at least 2 of 3 sub-periods.
MIN_T = 1.5
MIN_TRADES = 80
SUBPERIODS = 3
MIN_POSITIVE_SUBPERIODS = 2
# Above this a grid is worth a second thought, and the UI says so before the run
# — but it is ADVISORY, not a wall. This used to raise and refuse. The owner of
# the tool decides how wide to search; the tool's job is to price the search
# honestly, not to withhold it. The price is already on screen before you press
# Optimize (combination count, estimated seconds, and the t that pure noise
# reaches over that many trials) and in the ledger afterwards.
BIG_SWEEP = 5000

# Starting a pool costs a couple of seconds -- every worker rebuilds the tape --
# so it only pays above a few seconds of remaining work. Measured per
# combination on 37 days of NQ: orb 14 ms, ma_cross 15 ms, fvg 166 ms,
# vwap_revert 250 ms, sweep_follow 451 ms. A 200-combination orb sweep finishes
# in 4 s and would LOSE time in a pool; the same sweep on sweep_follow is 90 s
# and gains an order of magnitude. The decision is made from a measurement of
# the first combination rather than from a table like this one, which goes stale.
PARALLEL_MIN_SECONDS = 6.0


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


def grid(ranges: dict[str, list]) -> list[dict]:
    """Every combination of the ranges. No ceiling.

    A grid is a cross product, so parameters multiply: four ranges of ten values
    are ten thousand searches, not forty. That is worth knowing before pressing
    the button, which is why the count, the estimated runtime and the noise
    ceiling are all shown next to it -- but knowing it and being allowed to do
    it are different things, and only the first is this module's business.
    """
    keys = list(ranges)
    return [dict(zip(keys, vals))
            for vals in itertools.product(*(ranges[k] for k in keys))]


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
        if p.fixed:                               # a decision, not a dial
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


def account_check(trades, rules, contracts=1) -> dict:
    """Would this parameter set have survived that account? One replay, no rng.

    THE ACCOUNT IS A CONSTRAINT HERE, NEVER THE OBJECTIVE, and the distinction is
    the whole reason this is safe to run inside a search. Ranking on P(pass)
    would be optimising a bounded, highly non-linear transform of the edge:
    near zero edge it is dominated by variance, so the winner is the luckiest
    configuration rather than the best one. That failure mode is why the module
    docstring pre-registered against it.

    What a drawdown limit legitimately IS, though, is a hard feasibility bound.
    A configuration with a great t-statistic and a $3,000 intra-trade fall
    cannot be traded on a $2,000 account at any size, and no amount of edge
    fixes that. Bounds belong inside the search; objectives do not. So the rank
    stays on the daily t and this rides alongside it.

    `sim.replay` is the same walk the Backtesting tab's verdict uses -- one pass
    over the trades, no Monte Carlo -- so it costs a few milliseconds per
    combination rather than twenty thousand resampled futures.
    """
    if rules is None or not trades:
        return {}
    r = sim.replay(sorted(trades, key=lambda t: t.entry_time), rules,
                   contracts=contracts)
    return dict(
        outcome=r["outcome"], breach_reason=r["reason"],
        breach_date=r["date"], breach_day=r["day"],
        # How close it came to the floor at its worst, even when it lived. A
        # configuration that survived by $40 has not really survived; it has
        # been lucky once, and the ranking column cannot show that.
        margin=(None if r["margin"] is None else round(float(r["margin"]), 2)),
        account_pnl=round(float(r["profit"]), 2))


def result_row(p: dict, keys: list, trades, meta, rules=None, contracts=1) -> dict:
    """One row of the results table. THE only definition of one.

    Both the serial and the pooled path call this, so a sweep cannot report a
    different shape depending on how many cores the machine happened to have.
    """
    s = engine.summarise(trades, meta)
    subs = subperiod_pnl(trades)
    t_daily = s["t_daily"]
    return dict(
        params={k: p[k] for k in keys},
        trades=s["trades"], days=s["days"], pnl=round(s["pnl"], 2),
        mean=round(s.get("mean", 0.0), 2), wr=round(s["wr"], 4),
        per_day=round(s["per_day"], 2),
        t_daily=(None if t_daily != t_daily else round(float(t_daily), 3)),
        subperiods=[round(v, 2) for v in subs],
        positive_subperiods=sum(1 for v in subs if v > 0),
        **account_check(trades, rules, contracts))


# --------------------------------------------------------------------------
# worker processes
# --------------------------------------------------------------------------
# The pool is started with "spawn", not "fork", and that is deliberate. A sweep
# launched from the dashboard runs inside an HTTP handler THREAD; forking a
# multi-threaded process gives the child a single thread plus whatever locks the
# others were holding, which is a deadlock waiting for a slow afternoon. Spawn
# costs a fresh interpreter per worker and behaves identically on Windows, where
# the packaged build runs and fork does not exist at all.

_W: dict = {}           # per-worker state; only ever touched inside a worker


_ARRAYS = ("tape", "bars")      # the two dicts of arrays a prepared ctx carries


def _share(ctx: dict) -> dict:
    """Write the prepared tape ONCE, for every worker to map read-only.

    THE POOL USED TO RE-PREPARE THE TAPE PER WORKER, and that one decision cost
    both of the things a pool is supposed to give you. Memory: nineteen private
    copies of 30 sessions of full-session NQ measured 10.9 GB of PSS on an 11.8
    GB machine -- so the pool had to be sized down to one or two workers to be
    safe, which on a 20-core box turned a 57,600-combination sweep into four and
    a half hours of one core. Time: each of those copies costs a full slice and
    re-bar before the worker takes its first task.

    Mapped, both problems are the same non-problem. The arrays live in the page
    cache, which the kernel already shares between processes, so the twentieth
    worker costs what the first one did: nothing but its own interpreter. And a
    mapping is lazy, so a worker starts in milliseconds instead of seconds.

    Read-only is not a precaution here, it is the contract. Nothing in the
    backtest path writes into the tape or the bars -- strategies index and
    allocate their own results -- and a mapping opened `r` turns any future
    violation into a loud exception in the worker rather than a private copy of
    a 400 MB array that silently un-shares it.
    """
    # `sweep` deletes its own directory on every exit it can see -- finishing,
    # raising, and the Stop button. It cannot see SIGKILL or the power going out,
    # and half a gigabyte of orphan is not a nice thing to leave in someone's
    # temp folder. A day is far longer than any sweep that has run here and far
    # shorter than forever, so anything older than that belongs to a process that
    # is not coming back. Deleted on the way IN, because that is the moment we
    # are certainly running.
    old = time.time() - 86400
    for stale in Path(tempfile.gettempdir()).glob("propsim-ctx-*"):
        if stale.is_dir() and stale.stat().st_mtime < old:
            shutil.rmtree(stale, ignore_errors=True)

    d = Path(tempfile.mkdtemp(prefix="propsim-ctx-"))
    nbytes = 0
    for key in _ARRAYS:
        for name, arr in ctx[key].items():
            np.save(d / f"{key}.{name}.npy", arr)
            nbytes += arr.nbytes
    np.save(d / "dayi.npy", ctx["dayi"])
    nbytes += ctx["dayi"].nbytes
    scalars = {k: ctx[k] for k in ("contract", "tf_secs", "rth_only", "days",
                                   "start", "end", "n_holes", "hole_days")}
    return dict(dir=str(d), scalars=scalars, mb=nbytes / (1 << 20))


def _map_shared(share: dict) -> dict:
    """The other half of `_share`, inside a worker: a ctx backed by the page cache.

    `np.asarray` AROUND EVERY MAPPING, AND IT IS NOT DECORATION. It returns a
    base-class VIEW of the same buffer -- no copy, not one byte more resident,
    `shares_memory` is True -- but it drops the `np.memmap` subclass, and the
    subclass is what the fill loop pays for. `resolve` reads `ts[i]` and `px[i]`
    one scalar at a time, a few hundred thousand times per combination, and
    every one of those reads on a memmap goes through the subclass's array
    machinery instead of straight into the buffer. Measured on 300k reads:
    23 ms as an ndarray, 74 ms as a memmap, 24 ms as a view of the memmap.
    End to end that was 384 ms per combination in a worker against 164 ms for
    the same tape in the parent -- a pool where every worker silently ran at
    43% of the speed the same code had in series, which is most of why 19 of
    them only ever bought 2.8x.
    """
    d = Path(share["dir"])
    ctx = dict(share["scalars"])
    for key in _ARRAYS:
        ctx[key] = {p.name.split(".")[1]: np.asarray(np.load(p, mmap_mode="r"))
                    for p in d.glob(f"{key}.*.npy")}
    ctx["dayi"] = np.asarray(np.load(d / "dayi.npy", mmap_mode="r"))
    return ctx


def _worker_init(job: dict):
    """Map everything a worker needs, once, before it takes any task."""
    try:
        # The user's own strategies have to exist in here too -- but only as
        # CLASSES. Validating them is the parent's job, already done, and doing
        # it again per worker is what made a pool cost more than it saved.
        import plugins
        plugins.register_all(validate=False)
    except Exception:
        pass
    _W.update(job)
    _W["ctx"] = _map_shared(job["share"])


def _run_combo(p: dict) -> dict:
    """One combination, in a worker. Returns the row -- never the trades.

    Handing back the Trade objects would pickle several hundred dataclasses per
    combination for a summary that is nine numbers wide.
    """
    trades, meta = engine.backtest(_W["contract"], _W["strategy"], _W["tf_secs"],
                                   _W["start"], _W["end"], params=p,
                                   costs=_W["costs"], ctx=_W["ctx"])
    return result_row(p, _W["keys"], trades, meta, _W.get("rules"),
                      _W.get("contracts", 1))


def _worker_name(_) -> str:
    """What a pool worker calls itself. Only the selfcheck cares."""
    return mp.current_process().name


def _memory() -> tuple[float, float] | None:
    """(this process's resident size, physically available memory), in MB.

    `None` when neither platform's counters can be read, which is the only case
    left for a hardcoded guess.

    THE WINDOWS HALF USED TO BE THE GUESS, and it cost real time: with no
    `/proc` to read, `worker_count` fell back to a flat six workers on a machine
    reporting twenty. Both counters are two ctypes calls and both were run
    against the Windows Python this project actually ships to -- 24,261 MB
    total, 3,806 MB available, 57.9 MB working set with numpy imported -- so
    the reason the guess existed ("untestable from here") was never true; WSL
    can execute Windows binaries. Every failure mode still lands on the guess.

    `ullAvailPhys` is stricter than Linux's `MemAvailable`: it counts memory
    that is physically free rather than free-or-reclaimable, so this errs
    towards fewer workers on Windows, which is the correct direction to err.
    """
    try:                                   # Linux, and WSL
        with open("/proc/self/status") as f:
            rss_mb = next(int(l.split()[1]) for l in f
                          if l.startswith("VmRSS")) / 1024
        with open("/proc/meminfo") as f:
            avail_mb = next(int(l.split()[1]) for l in f
                            if l.startswith("MemAvailable")) / 1024
        return rss_mb, avail_mb
    except Exception:
        pass
    try:                                   # Windows, where the packaged build runs
        import ctypes
        from ctypes import wintypes
        MB = float(1 << 20)
        # Only two fields are read, but the structs must be declared in full or
        # the kernel writes past what was allocated: both calls are handed
        # `sizeof` and fill everything up to it.
        class _Mem(ctypes.Structure):
            _fields_ = ([("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)]
                        + [(n, ctypes.c_ulonglong) for n in
                           ("TotalPhys", "AvailPhys", "TotalPageFile", "AvailPageFile",
                            "TotalVirtual", "AvailVirtual", "AvailExtendedVirtual")])

        class _Proc(ctypes.Structure):
            _fields_ = ([("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]
                        + [(n, ctypes.c_size_t) for n in
                           ("PeakWorkingSetSize", "WorkingSetSize",
                            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                            "PagefileUsage", "PeakPagefileUsage")])

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # WITHOUT THIS THE HANDLE IS TRUNCATED TO 32 BITS and the second call
        # simply returns FALSE -- which is how it looked the first time it ran.
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                                                wintypes.DWORD]
        m = _Mem(); m.dwLength = ctypes.sizeof(m)
        p = _Proc(); p.cb = ctypes.sizeof(p)
        if not k32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                           ctypes.byref(p), p.cb):
            return None
        return p.WorkingSetSize / MB, m.AvailPhys / MB
    except Exception:
        return None


def worker_count(shared_mb: float = 0.0) -> int:
    """How many processes this machine can actually feed.

    Bounded by MEMORY, not by cores, and the bound is MEASURED rather than
    estimated. The first attempt at this priced a worker at the size of its
    numpy arrays -- 303 MB for 37 days of NQ -- and allowed 15 of them. The real
    peak is 987 MB, 3.3x that, because loading the compressed cache, slicing to
    RTH and building bars all allocate temporaries the finished arrays do not
    show. Fifteen workers asked for 14.8 GB on an 11 GB machine and took it to
    1 GB free before it was killed.

    So: ask the kernel what this process is holding, and then SUBTRACT what the
    workers will not be paying for. `shared_mb` is the tape and the bars, which
    since `_share` exist once in the page cache no matter how many workers map
    them; what is left of the parent's RSS is the interpreter, numpy, and the
    per-combination allocations -- which is precisely what each worker DOES pay
    for, privately, and is the only thing that should be multiplied by N.

    Getting that subtraction wrong is expensive in both directions, and both have
    happened here. Charging a worker for the whole tape sized a 57,600-
    combination sweep at ONE worker on a 20-core machine: four and a half hours
    in series while nineteen cores idled. Charging it for nothing at all is how
    an earlier version asked for 14.8 GB on an 11 GB machine and got killed.
    """
    n = max(1, (os.cpu_count() or 2) - 1)
    mem = _memory()
    if mem is None:
        # Neither platform's counters could be read. A bound, not a measurement,
        # and deliberately small: six is a smaller bet than the machine is
        # likely to justify, and being wrong here costs someone's RAM.
        return max(1, min(n, 6))
    rss_mb, avail_mb = mem
    # Floored: a worker is never free, and a floor of 0 would divide the whole
    # machine by rounding error.
    private_mb = max(rss_mb - shared_mb, 64.0)
    return max(1, min(n, int(avail_mb * 0.6 / private_mb)))


def evaluate(row: dict, noise_t: float) -> tuple[str, str]:
    """(verdict, reason) for one configuration, against the pre-registered bar."""
    t = row["t_daily"]
    if t is None or t != t:
        return "no signal", (
            f"fewer than {engine.MIN_T_DAYS} trading days: below that the daily "
            f"t-statistic is not bounded by anything and the noise ceiling it "
            f"would be judged against does not apply — run this on more tape")
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


def sweep(contract, strategy, tf_secs=300, start=None, end=None, ranges=None,
          costs: engine.Costs | None = None, on_progress=None,
          log_to_ledger=True, rules=None, contracts=1) -> dict:
    """Run every combination, rank by the daily t-statistic, charge the ledger."""
    # Same rule as the backtest screen: a strategy carrying its own `contracts`
    # was already simulated at that size, so the account layer must not scale it
    # a second time. See the note beside `report.build` in dashboard.py.
    if "contracts" in engine.LIBRARY[strategy].params:
        contracts = 1
    ranges = ranges or default_ranges(strategy)
    combos = grid(ranges)
    if not combos:
        # One empty value list makes the whole product empty. This used to return
        # an empty table and charge the ledger nothing, which reads exactly like
        # "your parameters found nothing" -- say which of the two it was.
        raise SystemExit("no combinations to sweep: "
                         + ", ".join(k for k, v in ranges.items() if not v)
                         + " was given an empty range")
    costs = costs or engine.Costs()
    ctx = engine.prepare(contract, tf_secs, start, end, strategy=strategy)

    t0 = time.time()
    keys = list(ranges)
    rows = []
    done = 0
    pool = None
    share = None
    try:
      # THE FIRST COMBINATION RUNS HERE, AND ITS COST DECIDES THE REST. A pool is
      # worth about an order of magnitude on a slow strategy and a net loss on a
      # fast one, and the gap between them is 30x -- too wide to guess at and too
      # dependent on the tape length to hardcode. So measure one, keep its
      # result, and let the number choose. Nothing is wasted either way.
      trades, meta = engine.backtest(contract, strategy, tf_secs, start, end,
                                     params=combos[0], costs=costs, ctx=ctx)
      rows.append(result_row(combos[0], keys, trades, meta, rules, contracts))
      done = 1
      per_combo = time.time() - t0
      rest = combos[1:]
      if on_progress:
          on_progress(done, len(combos), time.time() - t0)

      # NEVER NEST A POOL, and this is a real fork bomb rather than a tidiness
      # rule. Spawn re-executes the caller's main module in every worker; a
      # caller that forgot `if __name__ == "__main__":` therefore re-runs its own
      # sweep inside each worker, which starts its own pool, and so on. It took
      # an 11 GB machine to its knees while this was being written. PropSim's own
      # entry points are guarded, but users script against `sweep()` -- and the
      # cost of being wrong is their machine, not an exception. `parent_process()`
      # is still None this early in a spawned child; the process NAME is not.
      in_worker = mp.current_process().name != "MainProcess"
      # Written before the pool is sized, because what it takes off a worker's
      # bill is exactly what decides how many workers fit.
      if rest and not in_worker and per_combo * len(rest) > PARALLEL_MIN_SECONDS:
          share = _share(ctx)
      n = worker_count(share["mb"]) if share else 1
      if n > 1:
        pool = mp.get_context("spawn").Pool(
            n, initializer=_worker_init,
            initargs=(dict(contract=contract, strategy=strategy,
                           tf_secs=tf_secs, start=start, end=end,
                           costs=costs, keys=keys, rules=rules,
                           share=share,
                           contracts=contracts),))
        # BATCHED, AND THE BATCH BOUNDARY IS A BARRIER ON PURPOSE. Two things
        # depend on it, and both are correctness rather than speed:
        #
        #   `done` must be EXACTLY the number of combinations searched. The
        #   ledger charges it on cancellation, and a count that drifts low turns
        #   Stop into a way to search the data for free -- the thing the partial
        #   trial below exists to prevent. After a barrier the count is exact;
        #   with work in flight it is a guess, and the guess would be low.
        #
        #   Pause is `on_progress` declining to return (see dashboard.py). That
        #   holds the sweep only if nothing is dispatched behind it, which a
        #   barrier guarantees and a prefetching iterator does not.
        #
        # The cost is the spread within one batch of otherwise-identical work.
        for i in range(0, len(rest), n):
            batch = rest[i:i + n]
            rows.extend(pool.map(_run_combo, batch, chunksize=1))
            done += len(batch)
            if on_progress:
                on_progress(done, len(combos), time.time() - t0)
      else:
        for i, p in enumerate(rest):
            trades, meta = engine.backtest(contract, strategy, tf_secs, start, end,
                                           params=p, costs=costs, ctx=ctx)
            rows.append(result_row(p, keys, trades, meta, rules, contracts))
            done = i + 2
            if on_progress and (i % 5 == 0 or i == len(rest) - 1):
                on_progress(done, len(combos), time.time() - t0)
    except BaseException:
        # A CANCELLED SWEEP STILL LOOKED AT THE DATA. The user closed the page
        # or pressed Stop, so no result is returned -- but `done` combinations
        # were searched and the noise ceiling has to know, or stopping a sweep
        # early becomes a way to search for free. The partial trial carries its
        # own fingerprint so it cannot collapse into the completed sweep that a
        # re-run would write.
        if log_to_ledger and done:
            ledger.append("sweep", strategy=strategy, contract=contract,
                          fp=ledger.fingerprint(kind="sweep_partial",
                                                strategy=strategy, contract=contract,
                                                tf=tf_secs, start=ctx["start"],
                                                end=ctx["end"], ranges=ranges,
                                                done=done,
                                                slip=costs.slippage_ticks,
                                                comm=costs.commission),
                          timeframe=tp.tf_label(tf_secs), start=ctx["start"],
                          end=ctx["end"], days=ctx["days"],
                          n_trials=done, combos=len(combos), cancelled=True,
                          ranges={k: [float(v) for v in vs] for k, vs in ranges.items()})
        raise
    finally:
        # Stop, a closed browser and a clean finish all land here. Without it a
        # cancelled sweep leaves its workers alive, each holding 300 MB of tape,
        # and the next sweep starts on a machine that is already out of memory.
        if pool is not None:
            pool.terminate()
            pool.join()
        # The mappings die with the workers; the files behind them do not.
        if share is not None:
            shutil.rmtree(share["dir"], ignore_errors=True)

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
                                            contract=contract, tf=tf_secs,
                                            start=ctx["start"], end=ctx["end"],
                                            ranges=ranges,
                                            slip=costs.slippage_ticks,
                                            comm=costs.commission),
                      timeframe=tp.tf_label(tf_secs),
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
                contract=contract, timeframe=tp.tf_label(tf_secs), tf_secs=tf_secs,
                account_label=(None if rules is None else
                               f"{rules.firm}/{rules.variant}/{rules.phase}"
                               f"/${rules.size:,}"),
                account_contracts=contracts,
                start=ctx["start"], end=ctx["end"], days=ctx["days"],
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
        f"{res['timeframe']} bars",
        f"// {row['trades']} trades over {row['days']} sessions, "
        f"net {row['pnl']:,.0f}, daily t = "
        + ("n/a" if row["t_daily"] is None else f"{row['t_daily']:.2f}"),
        f"// Best of {res['combos']} combinations. Ledger trial "
        f"{res['trials_after']}; the best of that many null trials reaches "
        f"t = {res['noise_t']:.2f} by chance.",
        f"// VERDICT: {row['verdict'].upper()} — {row['reason']}.",
    ]
    # The account result travels with the numbers too. A parameter set that
    # broke the account it was searched against is the one thing a reader must
    # not have to go back to the table for -- and "survived by $15" is a fact
    # about luck, not about the configuration, so the margin comes with it.
    if row.get("outcome"):
        said = {"busted": "BROKE THE ACCOUNT", "passed": "PASSED THE ACCOUNT",
                "open": "survived the account"}.get(row["outcome"], row["outcome"])
        lines.append(f"// {said}: {res.get('account_label', 'the chosen account')}"
                     + (f" — {row['breach_reason']}" if row.get("breach_reason") else "")
                     + (f", day {row['breach_day']}" if row.get("breach_day") else ""))
        if row.get("margin") is not None:
            lines.append(f"// Tightest margin to the drawdown floor: "
                         f"{row['margin']:,.0f}. This is one sequence, not a rate.")
    lines += [
        "// NOT VALIDATED. These values were fitted on data already searched;",
        "// they have earned one forward test, not a funded account.",
        f"// Costs assumed: {res['slippage_ticks']:g} ticks/side slippage, "
        f"${res['commission']:g} per round trip.",
    ]
    lines += [f"{name(k)} = {v:g};" for k, v in p.items()]
    return "\n".join(lines)


def selfcheck():
    c = engine.tp.sample_contract()
    if c is None:
        print("no tape cached — run: python3 tape.py --build 'NQ 09-26'")
        return

    # 1. Range parsing and grid size.
    k, vals = parse_range("stop_ticks=20:60:10")
    assert k == "stop_ticks" and vals == [20, 30, 40, 50, 60], vals
    g = grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(g) == 6, g

    # 2. A prepared tape must give byte-identical results to a cold call, or the
    #    whole speedup is silently changing the answer.
    ctx = engine.prepare(c, 300)
    a, ma = engine.backtest(c, "orb", 300, ctx=ctx)
    b, mb = engine.backtest(c, "orb", 300)
    assert len(a) == len(b) and abs(sum(t.pnl for t in a) - sum(t.pnl for t in b)) < 1e-9, \
        (len(a), len(b))

    # 3. Sub-periods split by DAY and account for the whole run.
    trades, meta = engine.backtest(c, "fvg", 300, ctx=engine.prepare(c, 300))
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
    res = sweep(c, "orb", 300, ranges={"stop_ticks": [30, 40], "rr": [1.5, 2.0]},
                log_to_ledger=False)
    assert res["combos"] == 4 and len(res["rows"]) == 4, res["combos"]
    assert all("verdict" in r for r in res["rows"])
    ts = [r["t_daily"] for r in res["rows"] if r["t_daily"] is not None]
    assert ts == sorted(ts, reverse=True) or len(set(ts)) == 1, ts
    snippet = ninjascript_block(res, res["rows"][0])
    assert "NOT VALIDATED" in snippet and "StopTicks" in snippet, snippet

    # 7. THE POOLED PATH MUST RETURN EXACTLY WHAT THE SERIAL PATH RETURNS.
    #    Whether a sweep forks out to 14 cores is decided by a stopwatch and the
    #    amount of free RAM, which means it can differ between two runs on the
    #    same machine. If the answer moved with it, a result would not be
    #    reproducible -- and the ledger has already charged for it.
    rng = {"stop_ticks": [20, 30, 40, 50, 60], "rr": [1.5, 2.0]}
    serial = sweep(c, "orb", 300, ranges=rng, log_to_ledger=False)
    g = globals()
    keep = (PARALLEL_MIN_SECONDS, worker_count)
    g["PARALLEL_MIN_SECONDS"], g["worker_count"] = -1.0, lambda *_: 2
    try:
        par = sweep(c, "orb", 300, ranges=rng, log_to_ledger=False)
    finally:
        g["PARALLEL_MIN_SECONDS"], g["worker_count"] = keep
    assert par["combos"] == serial["combos"] == 10, par["combos"]
    assert par["rows"] == serial["rows"], (
        "the pooled sweep disagrees with the serial one:\n"
        + "\n".join(f"  {a}\n  {b}" for a, b in zip(par["rows"], serial["rows"]) if a != b))

    # 7b. THE POOL MUST BE SIZED FROM A MEASUREMENT ON EVERY PLATFORM THIS RUNS
    #     ON. While the Windows branch was a hardcoded six, a twenty-thread
    #     machine ran a sweep on six threads and nothing said so. `_memory`
    #     returning None is the one case where the guess is still correct, so
    #     what is asserted is that it does NOT return None here, and that both
    #     numbers are physically sane -- the same assertion the Windows build
    #     makes for itself when this runs there.
    mem = _memory()
    assert mem is not None, "no memory counters on this platform: pool sized by guess"
    _rss, _avail = mem
    assert 1.0 < _rss < 1e6 and _avail > 1.0, mem
    #     Bounded by cores at the top and by one at the bottom, and MONOTONE in
    #     what the workers do not have to pay for: the whole point of `_share`
    #     is that a bigger shared tape buys more workers, never fewer. It read
    #     the other way round once, and that is what sized a 57,600-combination
    #     sweep at one worker on a twenty-core machine.
    _cores = max(1, (os.cpu_count() or 2) - 1)
    _counts = [worker_count(s) for s in (0.0, 64.0, 512.0, 1e9)]
    assert all(1 <= x <= _cores for x in _counts), _counts
    assert _counts == sorted(_counts), f"more shared memory must not cost workers: {_counts}"

    # 7c. AND THE MAPPED TAPE MUST NOT BE AN `np.memmap`. It is a base-class
    #     view of one, which shares the buffer but not the subclass's per-scalar
    #     cost -- the fill loop reads a few hundred thousand scalars per
    #     combination, and as a memmap that ran at 43% of the parent's speed.
    #     Check 7 above proves the numbers agree; this proves they are reached
    #     at full speed, which no assertion on the results can see.
    _share_d = _share(engine.prepare(c, 300, strategy="orb"))
    try:
        _m = _map_shared(_share_d)
        assert type(_m["tape"]["px"]) is np.ndarray, type(_m["tape"]["px"])
        assert type(_m["dayi"]) is np.ndarray, type(_m["dayi"])
    finally:
        shutil.rmtree(_share_d["dir"], ignore_errors=True)

    # 8. AND A CANCELLED POOLED SWEEP CHARGES THE EXACT COUNT. This is why the
    #    batch boundary is a barrier: with work in flight the number of
    #    combinations already searched is a guess, the guess is low, and Stop
    #    becomes a discount on the noise ceiling. One combination runs to time
    #    the decision, then batches of two -- so raising at the first progress
    #    call past 2 must charge exactly 3, never "about 3".
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "trials.jsonl"
        real_append = ledger.append
        ledger.append = lambda kind, path=None, **kw: real_append(kind, led, **kw)
        g["PARALLEL_MIN_SECONDS"], g["worker_count"] = -1.0, lambda *_: 2
        try:
            boom = RuntimeError("stop pressed")
            def die(done, total, secs):
                if done >= 2: raise boom
            try:
                sweep(c, "orb", 300, ranges={"stop_ticks": [20, 30, 40, 50, 60, 70, 80, 90]},
                      on_progress=die)
            except RuntimeError as exc:
                assert exc is boom, exc
            else:
                raise AssertionError("the cancelled pooled sweep did not propagate")
        finally:
            ledger.append = real_append
            g["PARALLEL_MIN_SECONDS"], g["worker_count"] = keep
        charged = [json.loads(l) for l in led.read_text().splitlines()]
        assert len(charged) == 1, charged
        assert charged[0]["n_trials"] == 3, (
            f"pooled cancel charged {charged[0]['n_trials']} trials, 3 were searched")

    # 9. THE SIGNAL THE FORK-BOMB GUARD RESTS ON. A worker must be able to tell
    #    it is a worker, or `sweep` nests pools inside pools the moment a user
    #    calls it from a script without a `__main__` guard. The process NAME is
    #    what carries that, and it is set before the child re-executes the main
    #    module -- unlike `parent_process()`, which is still None at that point.
    #    If a future Python changes this, the guard fails open and the failure
    #    mode is somebody's machine, so it is pinned here rather than assumed.
    with mp.get_context("spawn").Pool(1) as _p:
        assert _p.map(_worker_name, [0])[0] != "MainProcess", (
            "a pool worker reports itself as MainProcess; the nested-pool guard "
            "in sweep() is now a no-op and an unguarded caller will fork-bomb")

    # A CANCELLED SWEEP MUST STILL COST ITS LOOKS. Without this the Stop button
    # is a way to search the data for free: run 200 combinations, read the
    # progress, cancel, and the noise ceiling never hears about it.
    import ledger as _ledger, tempfile, json as _json
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as td:
        led = _Path(td) / "trials.jsonl"
        real_append = _ledger.append
        _ledger.append = lambda kind, path=None, **kw: real_append(kind, led, **kw)
        try:
            boom = RuntimeError("client went away")
            def die(done, total, secs):
                if done >= 2: raise boom
            try:
                sweep(c, "orb", 300, ranges={"stop_ticks": [20, 30, 40, 50]},
                      on_progress=die)
            except RuntimeError as exc:
                assert exc is boom, exc
            else:
                raise AssertionError("the cancelled sweep did not propagate")
        finally:
            _ledger.append = real_append
        rows = [_json.loads(l) for l in led.read_text().splitlines()]
        assert len(rows) == 1, rows
        assert rows[0]["cancelled"] is True and rows[0]["n_trials"] >= 1, rows[0]
        cancelled_trials = rows[0]["n_trials"]

    print(f"selfcheck OK: a cancelled sweep still charged {cancelled_trials} "
          f"trial(s) to the ledger; prepared tape matches a cold run exactly; sub-periods sum "
          f"to the total; the noise ceiling vetoes a t=2.0 winner at 200 trials "
          f"(ceiling {ledger.expected_max_t(200):.2f}); 4-combination sweep ran in "
          f"{res['elapsed']}s and every row carries a verdict")


def main():
    mp.freeze_support()                 # see the note in dashboard.main()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract")
    ap.add_argument("--strategy")
    ap.add_argument("--tf", default="5m", help="bar size: 30s, 2m, 15m (bare "
                                               "numbers are minutes)")
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

    res = sweep(args.contract, args.strategy, tp.tf_secs(args.tf), args.start, args.end,
                ranges, costs, on_progress=prog, log_to_ledger=not args.no_ledger)
    print()
    if args.json:
        print(json.dumps(res, indent=1))
        return

    print(f"\n{res['label']} on {res['contract']}, {res['timeframe']} bars — "
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
