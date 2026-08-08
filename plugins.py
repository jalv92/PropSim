#!/usr/bin/env python3
"""Load strategies you (or an AI) wrote, from `~/.prop-sim/strategies/*.py`.

    python3 plugins.py                 # what is installed, and what is broken
    python3 plugins.py --template      # print a documented starter file
    python3 plugins.py --check <file>  # validate one file without installing it
    python3 plugins.py --selfcheck

A strategy file defines one class. It does not import anything: the engine hands
it everything it needs, which is also what makes the file safe to check
mechanically.

    class MyBreakout(Strategy):
        name, label = "my_breakout", "My breakout"
        params = {"stop_ticks": Param(40, 4, 200, "stop distance, ticks")}

        def entries(self, bars, tape, p):
            ...
            return entry_tick_indices, direction, stop_price, target_price

WHAT THE VALIDATION IS, AND WHAT IT IS NOT. Every file is parsed and checked
against an allowlist before it is executed: no imports, no file access, no
network, no `eval`/`exec`, no dunder attributes. Then it is smoke-tested on real
ticks and its output is checked against the engine's contract -- array lengths
that match, indices in range, a stop on the correct side of the entry.

That combination catches the things that actually go wrong: a model that helpfully
adds `urllib` and silently phones home, a strategy that returns four arrays of
three different lengths, a stop above the entry on a long. **It is not a sandbox
against a determined attacker.** A plugin runs with this program's privileges.
Put your own files in that folder and nobody else's.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

import engine
import tape as tp

USER_DIR = Path.home() / ".prop-sim" / "strategies"

# Validation verdicts, keyed by the plugin file's content hash. A verdict costs
# a subprocess plus a smoke backtest over a real contract -- minutes for a
# tick-level plugin -- which is the right price per plugin VERSION and an absurd
# one per app launch (the dashboard validates every file at startup). PASSes
# are cached; failures never are, so a broken file re-explains itself each run.
CHECK_CACHE = Path.home() / ".prop-sim" / "plugin_checks.json"

# Nothing needs importing -- the engine injects numpy, the tape helpers and the
# Strategy/Param base classes. `numpy` is allowed anyway: the sandbox already
# hands a plugin the real module as `np`, so an explicit `import numpy` grants
# nothing an attacker doesn't already have -- and it lets a plugin file run its
# own selfchecks standalone, outside the sandbox, without a second code path.
ALLOWED_IMPORTS = {"math", "numpy"}
BANNED_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "exit",
    "quit", "memoryview",
}
# numpy reaches the filesystem, and `np.load(..., allow_pickle=True)` is arbitrary
# code execution with a friendly name. A strategy is handed its data; it has no
# business opening anything.
NUMPY_DENY = {
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "DataSource", "ctypeslib",
    "f2py", "distutils", "testing", "vectorize", "frompyfunc",
}
SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    # __build_class__ is what a `class` statement compiles down to, so a namespace
    # without it cannot define a strategy at all. It is unreachable from the
    # source itself: the AST check rejects dunder attribute access.
    for k in ("__build_class__",
              "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
              "float", "int", "len", "list", "map", "max", "min", "print", "range",
              "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
              "zip", "isinstance", "issubclass", "type", "ValueError",
              "TypeError", "IndexError", "KeyError", "ZeroDivisionError",
              "Exception", "True", "False", "None")
    if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
}


class Rejected(Exception):
    """The file is not a valid strategy plugin. The message says why."""


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """A doorman, not an open door.

    `__import__` has to exist in the namespace: numpy's C-level reduction path
    resolves it from the CALLING frame's builtins, so `a[0:5].max()` and
    `np.isfinite(a).all()` both fail with KeyError('__import__') without it --
    which would rule out most of what a strategy does.

    So it exists and refuses everything outside the allowlist. `os`, `socket` and
    `subprocess` are unreachable through it, and the AST check already rejects the
    name `__import__` appearing in a plugin's source at all, so this only ever
    serves numpy's own internals.
    """
    root = (name or "").split(".")[0]
    if root in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"a strategy may not import {name!r}")


def ast_check(src: str, name="<plugin>") -> list[str]:
    """Structural check. Returns the list of reasons to reject, empty if clean."""
    try:
        tree = ast.parse(src, filename=name)
    except SyntaxError as exc:
        return [f"line {exc.lineno}: {exc.msg}"]

    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    bad.append(f"line {node.lineno}: import {a.name} — not allowed; "
                               f"numpy is already available as np, tape helpers as tp")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                bad.append(f"line {node.lineno}: from {node.module} import … — "
                           f"not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                bad.append(f"line {node.lineno}: attribute {node.attr} — dunder "
                           f"access is how a restricted namespace gets escaped")
            elif (isinstance(node.value, ast.Name) and node.value.id == "np"
                    and node.attr in NUMPY_DENY):
                bad.append(f"line {node.lineno}: np.{node.attr} — reaches the "
                           f"filesystem; a strategy is handed its data")
        elif isinstance(node, ast.Name) and node.id in BANNED_CALLS:
            bad.append(f"line {node.lineno}: {node.id} — not allowed in a strategy")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad.append(f"line {node.lineno}: global/nonlocal — a strategy keeps no "
                       f"state between runs")
    return bad


class _TapeHelpers:
    """The pure time helpers, and nothing else.

    Injecting the `tape` MODULE would hand a plugin `build_cache` and `load_cache`
    -- functions that read and write files. These four are arithmetic on
    timestamps, which is all a strategy legitimately needs.
    """
    TPS = tp.TPS
    day_index = staticmethod(tp.day_index)
    sec_of_day = staticmethod(tp.sec_of_day)
    date_str = staticmethod(tp.date_str)
    to_datetime = staticmethod(tp.to_datetime)


def _namespace() -> dict:
    """Everything a strategy file may use, handed to it rather than imported."""
    return {
        "__builtins__": dict(SAFE_BUILTINS, __import__=_guarded_import),
        "__name__": "propsim_plugin",   # a class statement needs a module name

        "Strategy": engine.Strategy,
        "Param": engine.Param,
        "np": np,
        "tp": _TapeHelpers,            # day_index, sec_of_day, date_str, TPS
        # THESE TWO ARE NQ's, AND THEY CANNOT BE ANYTHING ELSE HERE. A plugin is
        # loaded once, with no contract in hand, so a global cannot know which
        # tape it will be run against. `self.tick` and `self.point_value` on the
        # Strategy base ARE the running instrument's -- `engine.backtest` sets
        # them per run -- and that is what a plugin should size its stops in.
        # These stay because every plugin already written reads them, and a
        # rename that turns working files into NameErrors is not a fix.
        "TICK_SIZE": engine.TICK_SIZE,
        "POINT_VALUE": engine.POINT_VALUE,
    }


def load_source(src: str, filename="<plugin>") -> type:
    """Validate and execute one strategy source, returning its class."""
    bad = ast_check(src, filename)
    if bad:
        raise Rejected("; ".join(bad))
    ns = _namespace()
    try:
        exec(compile(src, filename, "exec"), ns)          # noqa: S102 — see module docstring
    except Exception as exc:
        raise Rejected(f"{type(exc).__name__} while loading: {exc}")

    classes = [v for k, v in ns.items()
               if isinstance(v, type) and issubclass(v, engine.Strategy)
               and v is not engine.Strategy]
    if not classes:
        raise Rejected("no class inheriting Strategy was defined")
    if len(classes) > 1:
        raise Rejected(f"{len(classes)} Strategy classes in one file — keep one per file")
    S = classes[0]
    for attr in ("name", "label"):
        if not isinstance(getattr(S, attr, None), str) or not getattr(S, attr):
            raise Rejected(f"the class needs a {attr} string")
    # `name` becomes a filename, a URL parameter and a ledger key. Constraining it
    # here means nothing downstream has to sanitise it -- and the AI author writes
    # the file at `<name>.py`, so a name of "../../x" would otherwise escape the
    # folder entirely.
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", S.name):
        raise Rejected(f"name {S.name!r} must be lowercase snake_case, up to 40 "
                       f"characters, starting with a letter — it is used as a "
                       f"filename and a ledger key")
    if not isinstance(getattr(S, "params", None), dict):
        raise Rejected("params must be a dict of name -> Param")
    for k, v in S.params.items():
        if not isinstance(v, engine.Param):
            raise Rejected(f"params[{k!r}] is not a Param(default, lo, hi, description)")
        if not (v.lo <= v.default <= v.hi):
            raise Rejected(f"params[{k!r}]: default {v.default} outside [{v.lo}, {v.hi}]")
    if "entries" not in vars(S):
        raise Rejected("the class must define entries(self, bars, tape, p)")
    # A plugin class has no file to `inspect.getsource`, so keep the text on it.
    # The NinjaScript translator has to SHOW a model the Python it is translating.
    S._source = src
    return S


def check_output(res, tape, bars) -> list[str]:
    """The engine's contract, checked on a real run.

    These are the mistakes a generated strategy actually makes, in the order they
    are made: arrays of different lengths, an index past the end of the tape, and
    a stop on the wrong side of the entry (which the engine would silently skip,
    so the strategy would appear to take no trades for no visible reason).
    """
    if not isinstance(res, tuple) or len(res) not in (4, 5, 6):
        return [f"entries() must return 4 to 6 arrays, got {type(res).__name__}"]
    # The optional fifth (limit price) and sixth (breakeven trigger) may be None,
    # which is how a strategy says "not this one" for the fifth while still
    # returning the sixth. Only the four required arrays are checked here.
    arrs = [np.asarray(a) for a in res if a is not None]
    n = len(arrs[0])
    problems = []
    if any(len(a) != n for a in arrs):
        problems.append(f"the returned arrays have different lengths: "
                        f"{[len(a) for a in arrs]}")
        return problems
    if n == 0:
        return ["no entries at all on this data — check the signal, not the engine"]
    ei, dr, st, tg = arrs[0], arrs[1], arrs[2], arrs[3]
    if not np.issubdtype(ei.dtype, np.integer):
        problems.append("the first array must be TICK INDICES (integers), not prices")
    elif ei.min() < 0 or ei.max() >= len(tape["ts"]):
        problems.append(f"entry index out of range: {ei.min()}..{ei.max()} "
                        f"for {len(tape['ts'])} ticks")
    if not set(np.unique(dr)).issubset({-1, 1}):
        problems.append(f"direction must be +1 or -1, got {sorted(set(np.unique(dr)))}")
    for nm, a in (("stop", st), ("target", tg)):
        if not np.isfinite(a).all():
            problems.append(f"{nm} contains NaN or infinity")
    if np.isfinite(st).all() and np.isfinite(tg).all() and len(ei) and not problems:
        px = tape["px"][np.clip(ei, 0, len(tape["px"]) - 1)]
        wrong_stop = np.where(dr > 0, st > px, st < px)
        wrong_targ = np.where(dr > 0, tg < px, tg > px)
        if wrong_stop.any():
            problems.append(f"{int(wrong_stop.sum())} of {n} stops are on the wrong "
                            f"side of the entry (a long's stop must be BELOW it)")
        if wrong_targ.any():
            problems.append(f"{int(wrong_targ.sum())} of {n} targets are on the wrong "
                            f"side of the entry")
    return problems


def smoke_test(S: type, contract: str | None = None, tf_secs=300) -> dict:
    """Run the strategy once on real ticks and check what it returned."""
    contract = contract or tp.sample_contract()
    if contract is None:
        return dict(ran=False, note="no tape cached — cannot smoke-test yet")
    # A full-session strategy handed an RTH-only tape finds nothing and reads
    # exactly like "no edge" (families.py's own warning, paid for once
    # already) -- every plugin checked here before this was RTH-only, so
    # nobody had hit it. `S` is the class itself, not yet in LIBRARY, so this
    # reads `full_session` directly rather than through the strategy-name
    # lookup `engine.prepare` also supports.
    ctx = engine.prepare(contract, tf_secs, rth_only=not S.full_session)
    strat = S()
    p = {k: v.default for k, v in strat.params.items()}
    res = strat.entries(ctx["bars"], ctx["tape"], p)
    problems = check_output(res, ctx["tape"], ctx["bars"])
    if problems:
        raise Rejected("; ".join(problems))
    # backtest() resolves the strategy by name out of the library, so the class
    # has to be visible there for the smoke run -- and removed again if it turns
    # out not to be installable.
    had = engine.LIBRARY.get(S.name)
    engine.LIBRARY[S.name] = S
    try:
        trades, meta = engine.backtest(contract, S.name, tf_secs, ctx=ctx)
        s = engine.summarise(trades, meta)
    finally:
        if had is None:
            engine.LIBRARY.pop(S.name, None)
        else:
            engine.LIBRARY[S.name] = had
    return dict(ran=True, contract=contract, timeframe=tp.tf_label(tf_secs),
                signals=len(np.asarray(res[0])), trades=s["trades"],
                pnl=round(s["pnl"], 2),
                t_daily=(None if s["t_daily"] != s["t_daily"] else round(s["t_daily"], 2)))


def _isolated_check(path: Path, timeout=30) -> tuple[bool, str]:
    """Validate a file in a separate process first, so a hang or a crash cannot
    take the app with it. Cheap: it happens once, at load."""
    code = textwrap.dedent(f"""
        import sys, json
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        import plugins
        try:
            S = plugins.load_source(open({str(path)!r}).read(), {path.name!r})
            print(json.dumps(dict(ok=True, name=S.name, label=S.label)))
        except Exception as exc:
            print(json.dumps(dict(ok=False, why=str(exc))))
        """)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"still running after {timeout}s at import time — an infinite loop?"
    out = (r.stdout or "").strip().splitlines()
    if not out:
        return False, f"produced no output; stderr: {(r.stderr or '').strip()[:300]}"
    try:
        d = json.loads(out[-1])
    except ValueError:
        return False, f"unexpected output: {out[-1][:200]}"
    return bool(d.get("ok")), d.get("why", "")


def scan(directory: Path | None = None, validate: bool = True) -> list[dict]:
    """Every file in the folder, with its verdict. Never raises.

    `validate=False` loads the classes and NOTHING else. A verdict costs a
    subprocess per file plus a smoke test, and a smoke test loads the whole
    contract tape and runs a backtest on it -- which is the right price to pay
    once, in the process that is about to show a user their strategy folder, and
    an absurd one to pay in each of nineteen sweep workers that were handed the
    strategy to run and a tape to run it on. Measured: 600 MB and several seconds
    per worker, on files the parent had already validated.
    """
    d = Path(directory or USER_DIR)
    if not d.is_dir():
        return []
    cache: dict = {}
    if validate:
        try:
            cache = json.loads(CHECK_CACHE.read_text())
        except (OSError, ValueError):
            cache = {}
    dirty = False
    rows = []
    for f in sorted(d.glob("*.py")):
        if f.name.startswith("_"):
            continue
        row = dict(file=f.name, path=str(f))
        src = f.read_text()
        sha = hashlib.sha256(src.encode()).hexdigest()
        hit = cache.get(sha) if validate else None
        ok, why = (True, "") if (hit or not validate) else _isolated_check(f)
        if not ok:
            rows.append(dict(row, ok=False, error=why))
            continue
        try:
            S = load_source(src, f.name)
            if S.name in engine.LIBRARY and S.name not in _INSTALLED:
                raise Rejected(
                    f"name {S.name!r} is already taken by a built-in strategy. "
                    f"Rename it: sharing a name would make the trial ledger's "
                    f"history ambiguous.")
            row.update(ok=True, name=S.name, label=S.label,
                       params=list(S.params), cls=S)
            if validate:
                if hit:
                    row["smoke"] = hit.get("smoke", {})
                else:
                    row["smoke"] = smoke_test(S)
                    cache[sha] = dict(file=f.name, smoke=row["smoke"])
                    dirty = True
            rows.append(row)
        except Rejected as exc:
            rows.append(dict(row, ok=False, error=str(exc)))
        except Exception as exc:
            rows.append(dict(row, ok=False, error=f"{type(exc).__name__}: {exc}"))
    if dirty:
        try:
            CHECK_CACHE.parent.mkdir(parents=True, exist_ok=True)
            CHECK_CACHE.write_text(json.dumps(cache, indent=1))
        except OSError:
            pass                      # a cache that cannot be written is just slow
    return rows


_INSTALLED: dict[str, type] = {}


def register_all(directory: Path | None = None, validate: bool = True) -> list[dict]:
    """Load every valid plugin into `engine.LIBRARY`. Returns the scan report."""
    rows = scan(directory, validate)
    for r in rows:
        if r.get("ok") and r.get("cls") is not None:
            engine.LIBRARY[r["name"]] = r["cls"]
            _INSTALLED[r["name"]] = r["cls"]
    return [{k: v for k, v in r.items() if k != "cls"} for r in rows]


def installed() -> list[str]:
    return sorted(_INSTALLED)


TEMPLATE = '''\
# A PropSim strategy. Drop this file in ~/.prop-sim/strategies/ and it appears in
# the Backtest and Optimize tabs.
#
# NOTHING IS IMPORTED. The engine hands you:
#   np          numpy
#   tp          tape helpers: tp.day_index(ts), tp.sec_of_day(ts), tp.TPS
#   TICK_SIZE   NQ's price increment, 0.25 -- see below
#   POINT_VALUE NQ's currency per point, 20.0 -- see below
#   Strategy, Param
#
# SIZE IN self.tick, NOT IN TICK_SIZE. The two globals above are NQ's and cannot
# be anything else: this file is loaded once, before any contract is chosen.
# `self.tick` and `self.point_value` are the instrument the run is ACTUALLY on,
# set on your instance before `entries` is called, and they are what the engine
# charges slippage and measures risk in. They differ: MNQ is $2 a point where NQ
# is $20, and MYM ticks at 1.00 where NQ ticks at 0.25.
#
# `bars` is a dict of arrays, one entry per bar:
#   t      timestamp of the bar's first tick   o h l c   open/high/low/close
#   v      volume                              delta     signed order flow
#   start  index of its first tick in `tape`   end       one past its last
#
# `tape` is a dict of arrays, one entry per TICK:
#   ts     timestamp        px    price
#   vol    size             side  +1 traded at the ask, -1 at the bid, 0 unknown
#
# You return parallel arrays: the TICK INDEX to enter at, the direction (+1/-1),
# the stop price and the target price. Optionally a fifth array of limit prices,
# if the entry is a resting limit rather than a market order.
#
# TWO RULES THE ENGINE CANNOT ENFORCE FOR YOU:
#
# 1. NO LOOKAHEAD. A bar's high, low and close are only known once it has closed.
#    If your signal comes from bar i, enter on bar i+1 -- or, if the entry is a
#    price level crossed mid-bar, find the actual tick that crossed it by scanning
#    tape["px"] between bars["start"][i] and bars["end"][i]. Entering at a bar's
#    first tick using that bar's own high is free money that does not exist; it is
#    the single most common way a backtest lies.
#
# 2. A LONG'S STOP GOES BELOW ITS ENTRY. The engine refuses a trade that opens
#    already past its own stop or target, so a sign error shows up as a strategy
#    that mysteriously takes no trades.


class MomentumBreak(Strategy):
    name, label = "momentum_break", "Momentum break (template)"
    uses_ticks = True

    params = {
        "lookback": Param(20, 3, 200, "bars in the range"),
        "stop_ticks": Param(40, 4, 200, "stop distance, ticks"),
        "rr": Param(2.0, 0.5, 6.0, "target as a multiple of the stop"),
        "min_delta": Param(0, -5000, 5000, "order-flow delta the bar must show"),
    }

    def entries(self, bars, tape, p):
        c, h, l = bars["c"], bars["h"], bars["l"]
        n = len(c)
        look = int(p["lookback"])
        if n < look + 2:
            return (np.array([], np.int64), np.array([], np.int8),
                    np.array([]), np.array([]))

        # Rolling extremes of the PREVIOUS `look` bars, so bar i is never part of
        # the range it is compared against.
        hi = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        for i in range(look, n):
            hi[i] = h[i - look:i].max()
            lo[i] = l[i - look:i].min()

        et, dr, st, tg = [], [], [], []
        stop = p["stop_ticks"] * TICK_SIZE
        for i in range(look, n - 1):
            if not np.isfinite(hi[i]):
                continue
            if abs(bars["delta"][i]) < p["min_delta"]:
                continue
            up = c[i] > hi[i]
            dn = c[i] < lo[i]
            if not (up or dn):
                continue
            d = 1 if up else -1
            # Enter on the NEXT bar's first tick: this bar's close is the signal.
            j = int(bars["start"][i + 1])
            fill = float(tape["px"][j])
            et.append(j)
            dr.append(d)
            st.append(fill - d * stop)
            tg.append(fill + d * stop * p["rr"])

        if not et:
            return (np.array([], np.int64), np.array([], np.int8),
                    np.array([]), np.array([]))
        return (np.array(et, np.int64), np.array(dr, np.int8),
                np.array(st), np.array(tg))
'''


def selfcheck():
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="propsim-plugins-"))

    # 1. The template must load, pass the contract check and actually trade.
    (d / "template.py").write_text(TEMPLATE)
    rows = scan(d)
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["ok"], r.get("error")
    assert r["name"] == "momentum_break", r
    if r["smoke"]["ran"]:
        assert r["smoke"]["trades"] > 0, r["smoke"]

    # 2. Things that must be refused, each for its own reason.
    cases = {
        "import os\nclass A(Strategy):\n name=label='a'\n params={}\n def entries(s,b,t,p): pass":
            "import",
        "class A(Strategy):\n name=label='a'\n params={}\n def entries(s,b,t,p):\n  return eval('1')":
            "eval",
        "class A(Strategy):\n name=label='a'\n params={}\n def entries(s,b,t,p):\n  return (1).__class__":
            "dunder",
        "x = 1": "Strategy",
        "class A(Strategy):\n name=label='a'\n params={'k': 5}\n def entries(s,b,t,p): pass":
            "Param",
        "class A(Strategy):\n name=label='a'\n params={'k': Param(9, 0, 5, 'x')}\n def entries(s,b,t,p): pass":
            "outside",
        # np.load with a pickle is arbitrary code execution wearing a friendly name
        "class A(Strategy):\n name=label='a'\n params={}\n def entries(s,b,t,p):\n  return np.load('/tmp/x.npy', allow_pickle=True)":
            "filesystem",
        # `name` becomes a filename. The AI author writes `<name>.py`, so a name that
        # walks out of the folder has to be refused at the contract, not downstream.
        "class A(Strategy):\n name='../../evil'\n label='e'\n params={}\n def entries(s,b,t,p): pass":
            "snake_case",
        "class A(Strategy):\n name='Has Spaces'\n label='e'\n params={}\n def entries(s,b,t,p): pass":
            "snake_case",
    }
    for src, expect in cases.items():
        try:
            load_source(src)
            raise AssertionError(f"should have been rejected ({expect}): {src[:40]}")
        except Rejected as exc:
            assert expect.lower() in str(exc).lower(), (expect, str(exc))

    # 2b. `import numpy` now passes: the sandbox already hands a plugin the
    # real module as `np`, so the explicit import grants nothing new.
    numpy_src = ("import numpy as np\n"
                 "class A(Strategy):\n"
                 " name=label='a'\n"
                 " params={}\n"
                 " def entries(s,b,t,p):\n"
                 "  return np.array([], np.int64)\n")
    load_source(numpy_src)   # must not raise Rejected

    # 3. Shadowing a built-in name is refused: the ledger's history would become
    #    ambiguous about which "orb" a past trial referred to.
    (d / "shadow.py").write_text(TEMPLATE.replace('"momentum_break"', '"orb"')
                                         .replace("name, label =", "name, label ="))
    bad = [x for x in scan(d) if x["file"] == "shadow.py"][0]
    assert not bad["ok"] and "already taken" in bad["error"], bad

    # 4. The output contract catches the mistakes a generated strategy makes.
    c = tp.sample_contract()
    if c:
        ctx = engine.prepare(c, 300)
        tape, bars = ctx["tape"], ctx["bars"]
        ei = np.array([10, 20], np.int64)
        px = tape["px"][ei]
        assert not check_output((ei, np.array([1, 1], np.int8), px - 1, px + 1),
                               tape, bars)
        # stop above a long's entry
        probs = check_output((ei, np.array([1, 1], np.int8), px + 1, px + 2), tape, bars)
        assert any("wrong side" in x for x in probs), probs
        # mismatched lengths
        probs = check_output((ei, np.array([1], np.int8), px, px), tape, bars)
        assert any("different lengths" in x for x in probs), probs
        # prices where indices belong
        probs = check_output((px, np.array([1, 1], np.int8), px - 1, px + 1), tape, bars)
        assert any("TICK INDICES" in x for x in probs), probs

    print(f"selfcheck OK: template loads and trades "
          f"({rows[0]['smoke'].get('trades', '—')} trades); "
          f"{len(cases)} malformed files refused with the right reason; "
          f"a built-in name cannot be shadowed; the output contract catches "
          f"wrong-side stops, length mismatches and prices-as-indices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--template", action="store_true", help="print a starter file")
    ap.add_argument("--check", metavar="FILE")
    ap.add_argument("--install-template", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.template:
        print(TEMPLATE, end="")
        return
    if args.install_template:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        p = USER_DIR / "momentum_break.py"
        if p.exists():
            raise SystemExit(f"{p} already exists — not overwriting it")
        p.write_text(TEMPLATE)
        print(f"wrote {p}")
        return
    if args.check:
        f = Path(args.check)
        ok, why = _isolated_check(f)
        if not ok:
            raise SystemExit(f"REJECTED: {why}")
        S = load_source(f.read_text(), f.name)
        print(f"OK: {S.name} — {S.label}, params: {', '.join(S.params) or 'none'}")
        try:
            print("smoke test:", smoke_test(S))
        except Rejected as exc:
            raise SystemExit(f"REJECTED by the output contract: {exc}")
        return

    d = Path(args.dir) if args.dir else USER_DIR
    rows = scan(d)
    if not rows:
        print(f"No strategy files in {d}.\n"
              f"Write one there, or start from the template:\n"
              f"  python3 plugins.py --install-template")
        return
    print(f"{'file':<28}{'name':<20}{'trades':>8}{'P&L':>11}{'t':>7}   status")
    for r in rows:
        if not r["ok"]:
            print(f"{r['file']:<28}{'—':<20}{'':>8}{'':>11}{'':>7}   "
                  f"REJECTED: {r['error'][:70]}")
            continue
        sm = r.get("smoke") or {}
        t = sm.get("t_daily")
        print(f"{r['file']:<28}{r['name']:<20}"
              f"{sm.get('trades', '—'):>8}{sm.get('pnl', 0):>11,.0f}"
              f"{(f'{t:.2f}' if t is not None else '—'):>7}   ok")


if __name__ == "__main__":
    main()
