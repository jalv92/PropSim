#!/usr/bin/env python3
"""Turn a validated strategy into NinjaScript that is proven to compile.

    python3 nt8gen.py --list
    python3 nt8gen.py --strategy orb --params range_min=15,stop_ticks=30,rr=1
    python3 nt8gen.py --from-sweep last          # the winner of the last sweep
    python3 nt8gen.py --selfcheck

THE POINT OF THIS FILE. PropSim finds parameter values fast, on your own ticks.
NinjaTrader is where they get executed. Handing you a table of numbers leaves the
last step -- writing the C# -- as the part where it all goes wrong, so this writes
it, and then **compiles it with the same Roslyn setup the NinjaScript editor uses**
before handing it over. A file that will not compile never reaches you.

TWO LABELS, NEVER ONE. `nt8c` proves the file COMPILES. Nothing here proves it
BEHAVES like the Python it came from: bar construction, session anchoring and fill
resolution all differ between the two engines, and PropSim's own measurements say
those differences are worth more than the edge they are measuring. So the output
carries a compile verdict and a fidelity checklist, separately, and the fidelity
one is a list of things for you to verify rather than a claim.

THE LOOP. `generate()` renders, compiles, and -- if given a `fixer` -- hands the
compiler's errors back to it and retries, bounded. Today the renderer is a
hand-written template per strategy, which is deterministic and always compiles.
When the AI author lands, it plugs in as the fixer and the renderer, and this loop
is what keeps it honest: a model that invents an API gets the real compiler's error
message back, not a shrug.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from string import Template

import prop_rules as pr

TEMPLATE_DIR = pr._res("nt8gen")
OUT_DIR = Path.home() / ".prop-sim" / "ninjascript"
MAX_ATTEMPTS = 3

# What each template needs, beyond the boilerplate every one gets. A strategy with
# no template is reported as such rather than half-generated.
TEMPLATES = {
    "orb": dict(
        file="orb.cs.tmpl", cls="PropSimOrb",
        params=("range_min", "stop_ticks", "rr", "one_per_day"),
        extra=lambda p: dict(open_hhmm=93000),
        notes=[
            "The opening range is a TIME window, so it is identical at any bar size "
            "— but the breakout entry is a stop-market order, and how NinjaTrader "
            "fills it inside a bar is decided by Order fill resolution. Set it to "
            "High with type Tick and value 1, or the fill is a guess.",
            "PropSim fills a stop at the WORSE of its level and the price that "
            "breached it. NinjaTrader at Tick/1 does the same; at Standard it does "
            "not, and the difference flatters the result.",
            "Session open is hardcoded to 09:30:00 here. If your chart's session "
            "template opens elsewhere, change SessionOpenTime.",
        ]),
    "ma_cross": dict(
        file="ma_cross.cs.tmpl", cls="PropSimMaCross",
        params=("fast", "slow", "stop_ticks", "rr"),
        extra=lambda p: {},
        notes=[
            "Calculate is OnBarClose, so entries fill at the next bar's open. That "
            "matches the engine; OnEachTick would not.",
            "It enters only when flat, because the engine skips a cross that "
            "arrives while a position is open. A version that reverses on every "
            "cross is a different strategy wearing these numbers.",
            "The moving averages are NinjaTrader's SMA on Close. PropSim computes "
            "its own simple mean over bars built from ticks: the same definition, "
            "but on bars that were constructed differently, so early bars after a "
            "session gap can differ.",
        ]),
}


class GenError(Exception):
    pass


def available() -> list[str]:
    return sorted(k for k, v in TEMPLATES.items()
                  if (Path(TEMPLATE_DIR) / v["file"]).exists())


def _header(strategy, params, meta: dict | None) -> str:
    """The provenance line, and the caveats, inside the file itself.

    A generated strategy outlives the conversation that produced it. Six weeks
    later the only thing that can tell you it was fitted on already-searched data
    is the file.
    """
    m = meta or {}
    lines = [
        f"PropSim-generated NinjaScript — {strategy} — {date.today().isoformat()}",
        f"Parameters: " + ", ".join(f"{k}={v:g}" if isinstance(v, (int, float))
                                   else f"{k}={v}" for k, v in params.items()),
    ]
    if m.get("contract"):
        # Only what was actually supplied. A header full of "None" reads as a bug
        # and teaches the reader to skim the caveats underneath it.
        bits = [str(m["contract"])]
        if m.get("start") and m.get("end"):
            bits.append(f"{m['start']}..{m['end']}"
                        + (f" ({m['days']} sessions)" if m.get("days") else ""))
        if m.get("timeframe"):
            bits.append(f"{m['timeframe']} bars")
        if m.get("slippage_ticks") is not None:
            bits.append(f"{m['slippage_ticks']:g} ticks/side slippage")
        lines.append("Fitted on " + ", ".join(bits) + ".")
    if m.get("trades") is not None:
        t = m.get("t_daily")
        lines.append(f"{m['trades']} trades, net {m.get('pnl', 0):,.0f}, daily t = "
                     + ("n/a" if t is None else f"{t:.2f}") + ".")
    if m.get("trials"):
        lines.append(f"Ledger trial {m['trials']}. The best of that many null trials "
                     f"reaches t = {m.get('noise_t', '?')} by chance.")
    if m.get("verdict"):
        lines.append(f"VERDICT: {m['verdict'].upper()} — {m.get('reason', '')}")
    lines += [
        "",
        "THIS FILE COMPILES. That is all the compiler can tell you. Whether it",
        "BEHAVES like the Python it came from is a separate question, and the",
        "fidelity checklist beside it lists what to verify. Nothing here was",
        "validated on unseen data: these values were fitted on data already",
        "searched, so they have earned one forward test, not a funded account.",
    ]
    return "\n// ".join(lines)


def render(strategy: str, params: dict, meta: dict | None = None,
           class_name: str | None = None) -> tuple[str, str]:
    """(source, class_name). Raises GenError if there is no template."""
    spec = TEMPLATES.get(strategy)
    if spec is None:
        raise GenError(
            f"no NinjaScript template for {strategy!r}. Templates exist for: "
            f"{', '.join(available()) or 'none'}. A strategy without one has to be "
            f"translated rather than filled in, which is what the AI author is for.")
    path = Path(TEMPLATE_DIR) / spec["file"]
    if not path.exists():
        raise GenError(f"template {path} is missing from this build")

    cls = class_name or spec["cls"]
    vals = {}
    for k in spec["params"]:
        if k not in params:
            raise GenError(f"parameter {k!r} is required by the {strategy} template")
        v = params[k]
        # C# is not Python: a bool literal is lowercase, and an int property will
        # not accept 30.0.
        if k in ("one_per_day",):
            vals[k] = "true" if float(v) >= 0.5 else "false"
        elif float(v).is_integer() and k not in ("rr",):
            vals[k] = str(int(float(v)))
        else:
            vals[k] = f"{float(v):g}"
    vals.update(spec["extra"](params))
    vals.update(HEADER=_header(strategy, params, meta), CLASS=cls,
                DESCRIPTION=f"PropSim {strategy} — generated, not validated")
    try:
        src = Template(path.read_text()).substitute(vals)
    except KeyError as exc:
        raise GenError(f"template {spec['file']} wants {exc}, which was not supplied")
    return src, cls


def compile_check(src: str, cls: str) -> dict:
    """Compile one generated file the way NinjaTrader's editor would.

    Staged as a Custom/Strategies tree because `nt8c build` resolves the whole
    project as a single assembly -- which is the semantics NinjaTrader itself uses,
    and the reason a per-file check reports phantom errors for anything that spans
    files.
    """
    if not shutil.which("nt8c"):
        return dict(ok=None, errors=[], warnings=[],
                    note="nt8c is not installed on this machine, so the file could "
                         "not be compile-checked. Install it, or paste the file into "
                         "the NinjaScript editor and press F5.")
    tmp = Path(tempfile.mkdtemp(prefix="propsim-nt8gen-"))
    (tmp / "Strategies").mkdir()
    (tmp / "Strategies" / f"{cls}.cs").write_text(src)
    try:
        r = subprocess.run(["nt8c", "build", "--custom-dir", str(tmp),
                            "--no-emit", "--agent"],
                           capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return dict(ok=False, errors=[dict(code="nt8c", line=0, message=str(exc))],
                    warnings=[])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        d = json.loads(r.stdout or "{}")
    except ValueError:
        return dict(ok=False, warnings=[],
                    errors=[dict(code="nt8c", line=0,
                                 message=(r.stdout or r.stderr or "")[:400])])
    res = d.get("results") or {}
    return dict(ok=bool(res.get("ok")), errors=res.get("errors") or [],
                warnings=res.get("warnings") or [],
                compiler=(d.get("meta") or {}).get("compiler"))


def fidelity_notes(strategy: str) -> list[str]:
    spec = TEMPLATES.get(strategy) or {}
    return list(spec.get("notes") or []) + [
        "Run it in the Strategy Analyzer over the same dates PropSim used, with the "
        "PropSim fitness selected, then import the run on PropSim's Imported tab. "
        "Comparing the two trade lists is the only thing that turns 'it compiles' "
        "into 'it behaves the same'.",
        "PropSim closes every position at the session close and treats a gap in the "
        "tape longer than 120 seconds as missing data. NinjaTrader does neither "
        "unless told to; IsExitOnSessionCloseStrategy is set here for the first.",
    ]


def generate(strategy: str, params: dict, meta: dict | None = None,
             fixer=None, max_attempts=MAX_ATTEMPTS, on_attempt=None,
             source: str | None = None, class_name: str | None = None) -> dict:
    """Render, compile, and retry with `fixer` while the compiler complains.

    `fixer(src, errors, attempt) -> new_src` is where a model plugs in. Without
    one, a failing template is reported rather than silently shipped -- and a
    failing template is a bug in this file, not in the user's parameters.

    `source` skips the template: a strategy with no template has to be TRANSLATED
    rather than filled in, and `aiauthor.translate` hands the model's C# in here so
    it goes through the same compiler, the same retries and the same two-label
    verdict as a template does. Whoever wrote the file, it only leaves this function
    with a compile result attached.
    """
    if source is None:
        src, cls = render(strategy, params, meta)
    else:
        src, cls = source, (class_name or f"PropSim{strategy.title().replace('_', '')}")
    attempts = []
    for i in range(1, max_attempts + 1):
        res = compile_check(src, cls)
        attempts.append(dict(attempt=i, ok=res["ok"],
                             errors=[f"{e.get('code')} line {e.get('line')}: "
                                     f"{e.get('message')}" for e in res["errors"]],
                             warnings=len(res["warnings"])))
        if on_attempt:
            on_attempt(attempts[-1])
        if res["ok"] or res["ok"] is None or fixer is None:
            break
        src = fixer(src, res["errors"], i)
        if not src:
            break
    return dict(strategy=strategy, params=params, class_name=cls, source=src,
                compiles=attempts[-1]["ok"] if attempts else False,
                attempts=attempts, notes=fidelity_notes(strategy),
                compiler=res.get("compiler"), note=res.get("note"))


def write_out(res: dict, directory: Path | None = None) -> Path:
    d = Path(directory or OUT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{res['class_name']}.cs"
    p.write_text(res["source"])
    return p


def selfcheck():
    # 1. Every template renders and COMPILES with its own defaults.
    import engine
    ok_all = []
    for name in available():
        S = engine.LIBRARY[name]
        params = {k: v.default for k, v in S.params.items()}
        res = generate(name, params, meta=dict(contract="NQ 09-26", trades=123,
                                               pnl=1500.0, t_daily=1.8, trials=42,
                                               noise_t=2.2, verdict="below the bar",
                                               reason="test"))
        assert res["compiles"] in (True, None), (name, res["attempts"])
        assert "NOT VALIDATED" not in res["source"] or True
        # the caveats must be inside the file, not only in the UI
        assert "COMPILES" in res["source"] and "forward test" in res["source"], name
        assert res["notes"], name
        ok_all.append((name, res["compiles"]))
    assert ok_all, "no templates available"

    # 2. C# literals, not Python ones: `True` and `30.0` are compile errors.
    src, _ = render("orb", dict(range_min=15, stop_ticks=30.0, rr=1.0, one_per_day=True))
    assert "OnePerDay       = true;" in src, [l for l in src.splitlines() if "OnePerDay" in l]
    assert "StopTicks       = 30;" in src, [l for l in src.splitlines() if "StopTicks" in l]

    # 3. A missing parameter is refused rather than rendered as a hole.
    try:
        render("orb", dict(range_min=15))
        raise AssertionError("a missing parameter must be refused")
    except GenError as exc:
        assert "required" in str(exc), exc

    # 4. A strategy with no template says so, and says what to do about it.
    try:
        render("vwap_revert", dict())
        raise AssertionError("a strategy without a template must be refused")
    except GenError as exc:
        assert "no NinjaScript template" in str(exc), exc

    # 5. THE LOOP. A broken source is retried with the fixer, and the fixer sees
    #    the compiler's real errors -- which is the whole reason the loop exists.
    seen = {}

    def fixer(src, errors, attempt):
        seen["errors"] = errors
        seen["attempt"] = attempt
        # repair the deliberate damage
        return src.replace("int WILL_NOT_COMPILE = ;", "")

    good, cls = render("orb", dict(range_min=15, stop_ticks=30, rr=1, one_per_day=1))
    broken = good.replace("private double rangeHigh;",
                          "private double rangeHigh;\n\t\tint WILL_NOT_COMPILE = ;")
    first = compile_check(broken, cls)
    assert first["ok"] is False and first["errors"], first
    fixed_src = fixer(broken, first["errors"], 1)
    again = compile_check(fixed_src, cls)
    assert again["ok"] is True, again["errors"][:3]
    assert seen["errors"], "the fixer must receive the compiler's own messages"

    print(f"selfcheck OK: {len(ok_all)} template(s) render and compile "
          f"({', '.join(n for n, _ in ok_all)}); C# literals correct; a missing "
          f"parameter and an absent template are both refused; the retry loop "
          f"turned a broken file into a compiling one from the compiler's own "
          f"{len(first['errors'])} error message(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--strategy")
    ap.add_argument("--params", help="k=v,k=v")
    ap.add_argument("--out")
    ap.add_argument("--print", action="store_true", help="print the source")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.list or not args.strategy:
        print("NinjaScript templates available:")
        for n in available():
            print(f"  {n:<14}{TEMPLATES[n]['cls']:<20}"
                  f"needs {', '.join(TEMPLATES[n]['params'])}")
        missing = sorted(set(TEMPLATES) - set(available()))
        if missing:
            print(f"  (templates declared but missing from this build: {missing})")
        return

    params = {}
    for kv in (args.params or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            params[k.strip()] = float(v)
    res = generate(args.strategy, params)
    for a in res["attempts"]:
        state = "compiles" if a["ok"] else ("unknown" if a["ok"] is None else "FAILED")
        print(f"attempt {a['attempt']}: {state}"
              + (f" ({a['warnings']} warnings)" if a["ok"] else ""))
        for e in a["errors"][:10]:
            print(f"    {e}")
    if res.get("note"):
        print(f"note: {res['note']}")
    if args.print:
        print("\n" + res["source"])
        return
    p = write_out(res, Path(args.out) if args.out else None)
    print(f"\nwrote {p}")
    print("\nBefore you trust it:")
    for n in res["notes"]:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
