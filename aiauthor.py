#!/usr/bin/env python3
"""Have Claude write a PropSim strategy, and prove the result before trusting it.

    python3 aiauthor.py --status
    python3 aiauthor.py --write "fade the first 30-minute range on NQ, stop at the
                                 opposite edge"  --name range_fade  --install
    python3 aiauthor.py --translate orb --params range_min=15,stop_ticks=30,rr=1
    python3 aiauthor.py --selfcheck          # offline: no API key needed

WHOSE KEY, AND WHERE IT LIVES. Yours. This reads it from the
ANTHROPIC_API_KEY environment variable, then `~/.prop-sim/anthropic-api-key.txt`,
then `anthropic_api_key` in `~/.prop-sim/config.json` -- all outside this repository,
because the repository is public. The key is never written into a generated file, the
trial ledger, or an API response, and everything that can reach a screen goes through
`redact()` first: an HTTP error from the SDK can quote the request it failed on.

WHY A MODEL IS ALLOWED NEAR THIS AT ALL. It is not allowed to be believed. What it
writes is parsed against the plugin allowlist, executed in the restricted namespace,
run on your real ticks, and then **probed for lookahead by re-running it with the
future changed** (`lookahead_probe`). If a trade's direction, stop or target moves when
data AFTER that trade moves, the file is rejected and the model is handed the specific
trade that changed. Three attempts, then it gives up rather than installing something
that failed.

GENERATION IS FREE, EVALUATION IS CHARGED. Writing a strategy costs nothing against
the trial ledger, and this deliberately does NOT report the smoke test's P&L or
t-statistic: reporting them would make generation a free search over your data, which
is exactly the thing the ledger exists to count. You get "it runs, it takes N trades".
The number that means something comes from the Backtest and Optimize tabs, which charge.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

import engine
import nt8gen
import ntdata
import plugins
import tape as tp

KEY_FILE = Path.home() / ".prop-sim" / "anthropic-api-key.txt"
DEFAULT_MODEL = "claude-opus-5"
MAX_ATTEMPTS = 3
MAX_TOKENS = 24000

# A fenced block, with or without a language tag. The model is asked for exactly one;
# taking the longest guards against it opening with a two-line illustration.
_FENCE = re.compile(r"```[a-zA-Z#+]*\s*\n(.*?)```", re.S)
_KEYISH = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}")


class AuthorError(Exception):
    """Something the user has to see, already redacted."""


# ---------------------------------------------------------------- credentials

def _key_from(text: str | None) -> str | None:
    """Pull the key out of whatever the user pasted.

    People paste the whole line the console gave them -- `Anthropic API Key:
    sk-ant-…`, or a quoted value, or `ANTHROPIC_API_KEY=sk-ant-…`. Reading the file
    verbatim turns that into a bare 401 with nothing to act on; this file's own key
    started life that way. So find the key inside the text, and fall back to the
    stripped text only when there is no recognisable one to find.
    """
    if not text:
        return None
    m = _KEYISH.search(text)
    if m:
        return m.group(0)
    return text.strip().strip("'\"") or None


def api_key() -> str | None:
    """Environment, then the key file, then config.json. None of them in this repo."""
    for src in (os.environ.get("ANTHROPIC_API_KEY"),
                _read_key_file(),
                ntdata.load_config().get("anthropic_api_key")):
        k = _key_from(src)
        if k:
            return k
    return None


def _read_key_file() -> str | None:
    try:
        return KEY_FILE.read_text() if KEY_FILE.exists() else None
    except OSError:
        return None


def key_source() -> str | None:
    """WHERE the key came from, for the UI. Never the key."""
    if _key_from(os.environ.get("ANTHROPIC_API_KEY")):
        return "the ANTHROPIC_API_KEY environment variable"
    if _key_from(_read_key_file()):
        return str(KEY_FILE)
    if _key_from(ntdata.load_config().get("anthropic_api_key")):
        return f"anthropic_api_key in {ntdata.CONFIG}"
    return None


def redact(text) -> str:
    """Scrub anything key-shaped, and the actual key, from text bound for a screen.

    Not paranoia: `anthropic.APIStatusError` messages and tracebacks can carry request
    context, and this program renders its errors into a browser. One leak into a
    screenshot or a pasted bug report is permanent.
    """
    out = _KEYISH.sub("sk-ant-<redacted>", str(text))
    k = api_key()
    if k and len(k) > 8:
        out = out.replace(k, "<redacted>")
    return out


def model_name() -> str:
    return (ntdata.load_config().get("anthropic_model") or "").strip() or DEFAULT_MODEL


def status() -> dict:
    """Everything the UI needs to decide whether to offer the feature."""
    try:
        import anthropic                                       # noqa: F401
        sdk = getattr(anthropic, "__version__", "installed")
    except ImportError:
        sdk = None
    return dict(
        key=bool(api_key()), key_source=key_source(), sdk=sdk, model=model_name(),
        key_file=str(KEY_FILE), strategies_dir=str(plugins.USER_DIR),
        contracts=tp.cached_contracts(),
        nt8c=bool(shutil.which("nt8c")),
    )


# ---------------------------------------------------------------- the prompts

def _contract_names() -> str:
    return ", ".join(sorted(engine.LIBRARY))


def system_python() -> str:
    """The strategy-author system prompt.

    `plugins.TEMPLATE` is pasted in whole rather than paraphrased. It is the file the
    validator enforces, it already documents every injected name and both rules the
    engine cannot check, and a paraphrase of a contract is a second contract that
    drifts.
    """
    return f"""You are a quantitative developer. You write ONE strategy file for \
PropSim, a backtester that resolves every fill on the user's own NinjaTrader 8 tick \
data.

The file you write must satisfy the contract below. It is the real template shipped \
with the program, and the validator enforces exactly what it describes.

<contract>
{plugins.TEMPLATE}
</contract>

RULES
1. Reply with ONE ```python fenced block containing the complete file, and nothing
   else outside it. No explanation, no preamble.
2. Import nothing. You are handed `np`, `tp`, `TICK_SIZE`, `POINT_VALUE`, `Strategy`
   and `Param`. `import math` is tolerated; anything else is rejected outright, as is
   any dunder attribute, `eval`, `open`, and the numpy functions that touch files.
3. NO LOOKAHEAD, and this is the one you will be caught on. At the instant you decide
   a trade, use only data that existed at that instant. A bar's high, low and close
   exist only after it closes. A mean, a VWAP or a percentile taken over the WHOLE
   array is a peek at the future for every bar before the end of it -- compute such
   things over a trailing window instead.
4. A long's stop is BELOW its entry and its target ABOVE it; short reversed. The engine
   refuses a trade that opens already past its own stop, so a sign error looks like a
   strategy that silently takes no trades.
5. Return four numpy arrays of the SAME length -- entry tick indices (int64), direction
   (+1/-1), stop price, target price -- or five, with resting limit prices last.
6. Every number a user might want to change belongs in `params` as
   `Param(default, lo, hi, "what it does")`, with the default inside [lo, hi]. Give
   ranges wide enough to sweep and narrow enough to be sane.
7. `name` is lowercase snake_case and must not be one of the built-ins: \
{_contract_names()}. `label` is a short human title.
8. A plain Python loop over bars is fine and often clearer; the tape has millions of
   ticks, so do not loop over ticks without a reason.

HOW YOU ARE JUDGED. The file is parsed against the allowlist, executed in the
restricted namespace, and run on real ticks. Then it is probed for lookahead: the tape
is re-run with prices AFTER a chosen moment shifted, and every trade decided before
that moment must come out identical. If one moves, you get told which one and why, and
you fix the cause -- not the symptom. Prefer a strategy that is obviously causal over
one that is clever."""


SYSTEM_CSHARP = """You are a NinjaScript developer. You port ONE PropSim strategy to a \
NinjaTrader 8 Strategy in C#, so that it takes the same trades in NinjaTrader that it \
took in PropSim.

RULES
1. Reply with ONE ```csharp fenced block containing the complete .cs file, nothing else.
2. Keep the first line exactly as given to you in <header>: it is the provenance and
   caveat block, and it must survive into the file.
3. `namespace NinjaTrader.NinjaScript.Strategies`, one public class deriving from
   `Strategy`, named exactly as given in <class_name>.
4. Set defaults in `State.SetDefaults`; build indicators in `State.DataLoaded`; never
   call an indicator before then. Every tunable value is a `[NinjaScriptProperty]` with
   a `[Range]` and a `[Display]`, and its default is the value given in <parameters>.
5. FIDELITY IS THE POINT, so match how PropSim entered:
   - PropSim enters at a specific tick. A signal taken from a bar's CLOSE must fill on
     the NEXT bar, so use `Calculate.OnBarClose` and a market order.
   - A signal that is a PRICE LEVEL crossed intrabar must be a resting stop-market or
     limit order at that level (`EnterLongStopMarket`, `EnterShortLimit`, ...), never a
     market order on a bar boundary. Entering at a bar's open using that bar's own high
     is the lookahead that makes a backtest lie.
   - PropSim holds ONE position at a time and skips a signal that arrives while a trade
     is open. Set `EntriesPerDirection = 1` and return early when not flat.
   - Stops and targets: `SetStopLoss`/`SetProfitTarget` with `CalculationMode.Ticks`,
     from the same tick distances PropSim used.
   - `IsExitOnSessionCloseStrategy = true` and `IncludeCommission = true`: PropSim
     flattens at the session close and charges commission.
6. Use only real NinjaTrader 8 API. If you are unsure a member exists, use one you are
   sure of. You will be handed the compiler's own errors and given another attempt, so
   invented APIs cost the user money and time.
7. No `using` you do not need, and no code outside the class.

You are given the Python source, its parameter values, and the header. Port the LOGIC,
not the numpy."""


def _user_python(description: str, name: str | None) -> str:
    """Request in XML, task last -- the variable text must not read as instructions."""
    parts = [f"<request>\n{description.strip()}\n</request>"]
    if name:
        parts.append(f"Use `{name}` as the strategy's `name`.")
    parts.append("Write the complete file now, in one ```python block.")
    return "\n\n".join(parts)


def _retry_python(problems: str, attempt: int, total: int) -> str:
    return (f"<validator_output>\n{problems}\n</validator_output>\n\n"
            f"That is the checker's own message, verbatim. Fix what caused it and reply "
            f"with the complete corrected file in one ```python block. "
            f"This is attempt {attempt + 1} of {total}; there is no partial credit.")


def _user_csharp(py_src: str, params: dict, cls: str, header: str) -> str:
    vals = "\n".join(f"{k} = {v!r}" for k, v in params.items())
    return (f"<python_source>\n{py_src}\n</python_source>\n\n"
            f"<parameters>\n{vals}\n</parameters>\n\n"
            f"<class_name>{cls}</class_name>\n\n"
            f"<header>\n// {header}\n</header>\n\n"
            f"Port it now, in one ```csharp block.")


def _retry_csharp(errors: list[dict], attempt: int, total: int) -> str:
    lines = "\n".join(f"{e.get('code')} at line {e.get('line')}: {e.get('message')}"
                      for e in errors[:25])
    return (f"<compiler_errors>\n{lines}\n</compiler_errors>\n\n"
            f"Those are Roslyn's own errors from the NinjaScript compiler, on the file "
            f"you just wrote. Fix them and reply with the complete corrected file in "
            f"one ```csharp block. Attempt {attempt + 1} of {total}.")


# ---------------------------------------------------------------- the API call

def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text or "")
    if not blocks:
        raise AuthorError(
            "the model replied without a fenced code block, so there is nothing to "
            f"validate. It said: {redact((text or '').strip()[:300])!r}")
    return max(blocks, key=len).strip() + "\n"


def _ask(system: str, messages: list, model: str | None = None,
         max_tokens=MAX_TOKENS, effort="high") -> tuple[str, list, dict]:
    """One Messages call. Returns (text, content blocks, usage).

    The content blocks go back verbatim on the next turn: thinking blocks have to be
    replayed unedited on the same model, and this loop always continues on the same one.

    Streamed because `max_tokens` is large -- a non-streaming request that size risks
    an HTTP timeout, and there is a person watching a progress bar.
    """
    try:
        import anthropic
    except ImportError:
        raise AuthorError(
            "the AI author needs the official Anthropic SDK, which is not installed. "
            "Run:  pip install anthropic   (add --user on Linux if pip refuses). "
            "Everything else in PropSim works without it.")
    key = api_key()
    if not key:
        raise AuthorError(
            f"no Anthropic API key found. Put yours in {KEY_FILE} (one line, nothing "
            f"else), or set ANTHROPIC_API_KEY. It is your key and your bill; PropSim "
            f"never sends it anywhere but api.anthropic.com.")

    client = anthropic.Anthropic(api_key=key, max_retries=3)
    try:
        with client.messages.stream(
                model=model or model_name(), max_tokens=max_tokens, system=system,
                messages=messages,
                output_config={"effort": effort}) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise AuthorError(f"the API key was rejected. Check the one in "
                          f"{key_source() or KEY_FILE}.")
    except anthropic.RateLimitError:
        raise AuthorError("rate-limited by the API. Wait a minute and try again.")
    except anthropic.NotFoundError:
        raise AuthorError(f"the model {model or model_name()!r} is not available to "
                          f"this key. Set `anthropic_model` in {ntdata.CONFIG}.")
    except anthropic.APIStatusError as exc:
        raise AuthorError(f"the API returned {exc.status_code}: "
                          f"{redact(getattr(exc, 'message', exc))}")
    except anthropic.APIConnectionError:
        raise AuthorError("could not reach api.anthropic.com. Check the connection.")

    # A refusal is a successful HTTP 200 with no usable content; reading content[0]
    # blind would raise IndexError and look like a bug in PropSim.
    if msg.stop_reason == "refusal":
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        raise AuthorError(
            "the model declined this request"
            + (f" (category: {cat})" if cat else "")
            + ". Rephrase it, or write the strategy yourself from the template.")
    text = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    usage = dict(input=u.input_tokens, output=u.output_tokens,
                 cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                 model=msg.model, stop_reason=msg.stop_reason)
    if msg.stop_reason == "max_tokens":
        raise AuthorError(
            f"the reply hit the {max_tokens}-token ceiling and is cut off. Ask for a "
            f"simpler strategy, or raise MAX_TOKENS.")
    return text, msg.content, usage


# ---------------------------------------------------------------- lookahead probe

def _perturb(tape: dict, frm: int, shift_ticks=100) -> dict:
    """A copy of the tape with everything from tick `frm` onwards moved.

    Prices jump by `shift_ticks` and aggressor side flips. Bar boundaries come from
    timestamps only, so they are untouched -- which keeps the comparison exact instead
    of approximate.
    """
    t = {k: v.copy() for k, v in tape.items()}
    t["px"][frm:] += shift_ticks * engine.TICK_SIZE
    if "side" in t:
        t["side"][frm:] = -t["side"][frm:]
    return t


def _last_sessions(tape: dict, bars: dict, sessions: int, timeframe: int):
    """The tail of the tape, so the probe can afford to re-run several times.

    Causality does not need six months of ticks to show itself, and the probe re-runs
    the strategy once per cut. On a full NQ cache that would cost more than the backtest
    it is validating.
    """
    days = tp.day_index(tape["ts"])
    uniq = np.unique(days)
    if len(uniq) <= sessions:
        return tape, bars, len(uniq)
    k = int(np.searchsorted(days, uniq[-sessions]))
    t = {key: v[k:] for key, v in tape.items()}
    return t, tp.build_bars(t, timeframe), sessions


def lookahead_probe(S: type, ctx: dict | None = None, contract=None, timeframe=5,
                    sessions=15, max_cuts=6, shift_ticks=100) -> dict:
    """Does its answer for a trade change when data AFTER that trade changes?

    THE CUTS ARE THE TRADES. Perturbing at an arbitrary point only catches a peek if a
    trade happens to sit on the boundary -- a version of this cut at fixed fractions of
    the tape cleared a strategy that reads its own entry bar's high, because no trade
    landed on the straddling bar. So the cut is placed one tick AFTER a real entry: that
    trade is then decided entirely from data the perturbation left alone, and every
    earlier trade is checked in the same run for free. A strategy that reads anything at
    or after its own entry moves, deterministically, and the diff names the trade.

    WHAT IT CANNOT SEE. Only the sampled trades are pinned exactly; a peek that affects
    solely unsampled trades can hide, and the probe says nothing about whether the
    strategy makes money. A clean probe means causal, not profitable.
    """
    contracts = tp.cached_contracts()
    contract = contract or (ctx or {}).get("contract") or (contracts[0] if contracts else None)
    if contract is None:
        return dict(ran=False, clean=None, findings=[],
                    note="no tape cached, so the lookahead probe could not run")
    ctx = ctx or engine.prepare(contract, timeframe)
    strat = S()
    p = {k: v.default for k, v in strat.params.items()}

    def run(t, b):
        r = strat.entries(b, t, dict(p))
        a = [np.asarray(x) for x in r[:4]]
        return a if len(a) == 4 and len({len(x) for x in a}) == 1 else None

    # Widen once if the tail is too quiet: a strategy that trades weekly has nothing to
    # probe in fifteen sessions, and silence is not a pass.
    for window in (sessions, 0):
        tape, bars, used = (_last_sessions(ctx["tape"], ctx["bars"], window, timeframe)
                            if window else (ctx["tape"], ctx["bars"], None))
        base = run(tape, bars)
        if base is not None and len(base[0]) >= 2:
            break
    if base is None or len(base[0]) < 2:
        return dict(ran=False, clean=None, findings=[],
                    note="fewer than two comparable signals on this data, so there was "
                         "nothing to probe")

    ei = base[0]
    picks = np.unique(ei[np.linspace(0, len(ei) - 1,
                                    min(max_cuts, len(ei))).astype(int)])
    findings, checked = [], 0
    for j in picks:
        frm = int(j) + 1                    # everything STRICTLY after this entry
        if frm >= len(tape["ts"]) - 1:
            continue
        t2 = _perturb(tape, frm, shift_ticks)
        b2 = tp.build_bars(t2, timeframe)
        if len(b2["c"]) != len(bars["c"]):
            continue                        # bar grid moved; comparison would be unsound
        alt = run(t2, b2)
        if alt is None:
            findings.append(f"with prices after tick {frm} shifted, entries() returned "
                            f"malformed output — it is reacting to the future by failing")
            continue
        keep, keep2 = base[0] < frm, alt[0] < frm
        checked += int(keep.sum())
        b_ei, a_ei = base[0][keep], alt[0][keep2]
        if len(b_ei) != len(a_ei) or not np.array_equal(b_ei, a_ei):
            findings.append(
                f"{len(b_ei)} trades were decided at or before tick {int(j)}, but "
                f"{len(a_ei)} of them survive once prices AFTER that tick are changed. "
                f"Whether a trade happens cannot depend on data that did not exist yet.")
            continue
        for lbl, i in (("direction", 1), ("stop", 2), ("target", 3)):
            bv, av = base[i][keep], alt[i][keep2]
            bad = ~np.isclose(bv.astype(float), av.astype(float),
                              rtol=0, atol=engine.TICK_SIZE / 4)
            if bad.any():
                k = int(np.argmax(bad))
                findings.append(
                    f"the {lbl} of {int(bad.sum())} of {len(bv)} trades changed when "
                    f"only prices AFTER tick {int(j)} were changed — e.g. the trade "
                    f"entering at tick {int(b_ei[k])}: {lbl} {float(bv[k]):.2f} became "
                    f"{float(av[k]):.2f}. entries() is reading data from at or after "
                    f"the entry it is deciding.")
                break
        if findings:
            break                           # one proven peek is the whole verdict
    return dict(ran=True, clean=not findings, findings=findings, cuts=len(picks),
                sessions=used, compared=checked, contract=contract)


# ---------------------------------------------------------------- validation

def validate(src: str, contract=None, timeframe=5) -> tuple[str, dict]:
    """Everything a generated strategy has to survive. ("", info) when it survives.

    Order matters: the structural check is free, the smoke test costs seconds, the
    lookahead probe costs three more runs. Each returns the message the model needs,
    which is why they are not collapsed into one "invalid" string.
    """
    info: dict = {}
    tmp = Path(tempfile.mkdtemp(prefix="propsim-ai-")) / "candidate.py"
    tmp.write_text(src)
    try:
        # A separate process first: a generated file can loop forever at import time,
        # and that must not take the dashboard with it.
        ok, why = plugins._isolated_check(tmp)
        if not ok:
            return why, info
        S = plugins.load_source(src, "candidate.py")
        if S.name in engine.LIBRARY and S.name not in plugins._INSTALLED:
            return (f"name {S.name!r} is already taken by a built-in strategy. Choose "
                    f"another: two strategies sharing a name make the trial ledger's "
                    f"history ambiguous."), info
        info.update(name=S.name, label=S.label, params=list(S.params))
        sm = plugins.smoke_test(S, contract, timeframe)
        # Deliberately NOT the P&L or the t-statistic: see the module docstring.
        info["smoke"] = dict(ran=sm.get("ran"), trades=sm.get("trades"),
                             signals=sm.get("signals"), contract=sm.get("contract"),
                             note=sm.get("note"))
        if sm.get("ran") and not sm.get("trades"):
            return ("it loads and produces signals, but the engine resolved zero trades "
                    "from them — every entry was rejected. The usual cause is a stop or "
                    "target on the wrong side of the entry, or an entry index that lands "
                    "outside a session."), info
        probe = lookahead_probe(S, contract=contract or sm.get("contract"),
                                timeframe=timeframe)
        info["lookahead"] = probe
        if probe.get("findings"):
            return "LOOKAHEAD: " + " ".join(probe["findings"]), info
        return "", info
    except plugins.Rejected as exc:
        return str(exc), info
    except Exception as exc:
        return f"{type(exc).__name__}: {redact(exc)}", info
    finally:
        try:
            tmp.unlink()
            tmp.parent.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------- write a strategy

def _provenance(description: str, model: str, info: dict) -> str:
    pr = info.get("lookahead") or {}
    sm = info.get("smoke") or {}
    lines = [
        f"PropSim strategy written by Claude ({model}) on {date.today().isoformat()}.",
        "",
        "Request:",
    ] + [f"  {ln}" for ln in description.strip().splitlines()] + [
        "",
        "Validated: parsed against the plugin allowlist, executed in the restricted",
        f"namespace, and run on {sm.get('contract') or 'cached ticks'} "
        f"({sm.get('trades', '?')} trades resolved from {sm.get('signals', '?')} signals).",
    ]
    if pr.get("ran"):
        lines.append(f"The lookahead probe re-ran it with the future perturbed at "
                     f"{pr.get('cuts')} points and found no trade whose direction, stop")
        lines.append(f"or target depended on data after itself ({pr.get('compared')} "
                     f"trades compared).")
    else:
        lines.append(f"The lookahead probe did NOT run: {pr.get('note', 'no reason given')}.")
    lines += [
        "",
        "That is a check for mechanical honesty, NOT evidence of an edge. No parameter",
        "here has been searched, and nothing has been validated on unseen data. Backtest",
        "it, then sweep it, and let the trial ledger tell you what the best result is",
        "worth. Read the code before you run it: it was written by a model.",
    ]
    return "\n".join(f"# {ln}".rstrip() for ln in lines)


def write_strategy(description: str, name: str | None = None, model: str | None = None,
                   contract=None, timeframe=5, max_attempts=MAX_ATTEMPTS,
                   on_attempt=None) -> dict:
    """Ask, validate, and hand the validator's own words back until it passes."""
    if not (description or "").strip():
        raise AuthorError("describe the strategy you want in a sentence or two.")
    model = model or model_name()
    system = system_python()
    messages = [{"role": "user", "content": _user_python(description, name)}]
    attempts, src, info, problems = [], None, {}, "not attempted"

    for i in range(1, max_attempts + 1):
        text, content, usage = _ask(system, messages, model)
        src = extract_code(text)
        problems, info = validate(src, contract, timeframe)
        attempts.append(dict(attempt=i, ok=not problems, problems=problems,
                             usage=usage, name=info.get("name")))
        if on_attempt:
            on_attempt(attempts[-1])
        if not problems:
            break
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": _retry_python(problems, i, max_attempts)}]

    ok = not problems
    return dict(ok=ok, source=src, info=info, attempts=attempts, model=model,
                description=description, problems=problems,
                header=_provenance(description, model, info) if ok else None,
                tokens=dict(
                    input=sum(a["usage"]["input"] for a in attempts),
                    output=sum(a["usage"]["output"] for a in attempts)))


def install(res: dict, directory: Path | None = None, overwrite=False) -> Path:
    """Write a passing strategy into the folder the dashboard loads from."""
    if not res.get("ok"):
        raise AuthorError("that strategy did not pass validation, so it is not "
                          "installable. The last problem was: " + str(res.get("problems")))
    d = Path(directory or plugins.USER_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{res['info']['name']}.py"
    if p.exists() and not overwrite:
        raise AuthorError(f"{p} already exists. Delete it, or rename the strategy.")
    p.write_text(res["header"] + "\n\n" + res["source"])
    return p


# ---------------------------------------------------------------- translate to C#

def strategy_source(name: str) -> str:
    """The Python a translation has to be faithful to."""
    S = engine.LIBRARY.get(name)
    if S is None:
        raise AuthorError(f"no strategy called {name!r}. Loaded: {_contract_names()}")
    src = getattr(S, "_source", None)
    if src:
        return src
    import inspect
    return inspect.getsource(S)


def translate(strategy: str, params: dict, meta: dict | None = None,
              model: str | None = None, max_attempts=MAX_ATTEMPTS,
              on_attempt=None) -> dict:
    """Port a strategy to NinjaScript, and let the real compiler have the last word.

    This exists for the strategies `nt8gen` has no template for. A template is
    deterministic and always compiles; a translation is neither, so it goes through
    `nt8gen.generate`, which compiles it with the same Roslyn setup the NinjaScript
    editor uses and hands the errors back here for another attempt.
    """
    model = model or model_name()
    py = strategy_source(strategy)
    cls = f"PropSim{strategy.title().replace('_', '')}"
    header = nt8gen._header(strategy, params, meta).lstrip("\n// ")
    messages = [{"role": "user", "content": _user_csharp(py, params, cls, header)}]
    calls: list[dict] = []

    def ask(msgs) -> str:
        text, content, usage = _ask(SYSTEM_CSHARP, msgs, model)
        calls.append(usage)
        msgs.append({"role": "assistant", "content": content})
        return extract_code(text)

    src = ask(messages)

    def fixer(bad_src, errors, attempt):
        if attempt >= max_attempts:
            return None
        messages.append({"role": "user",
                         "content": _retry_csharp(errors, attempt, max_attempts)})
        try:
            return ask(messages)
        except AuthorError:
            return None

    res = nt8gen.generate(strategy, params, meta=meta, fixer=fixer,
                          max_attempts=max_attempts, on_attempt=on_attempt,
                          source=src, class_name=cls)
    res["notes"] = [
        "This file was TRANSLATED by a model, not filled into a hand-written template. "
        "It compiles, and that is the only mechanical guarantee: read it before you run "
        "it, and compare its trade list against PropSim's on the same dates.",
    ] + res["notes"]
    res.update(model=model, translated=True,
               tokens=dict(input=sum(c["input"] for c in calls),
                           output=sum(c["output"] for c in calls)))
    return res


# ---------------------------------------------------------------- selfcheck

_HONEST = '''\
class ProbeHonest(Strategy):
    name, label = "probe_honest", "Honest (probe fixture)"
    params = {"look": Param(20, 3, 100, "bars"), "stop_ticks": Param(40, 4, 200, "ticks")}

    def entries(self, bars, tape, p):
        c, h = bars["c"], bars["h"]
        look, n = int(p["look"]), len(c)
        et, dr, st, tg = [], [], [], []
        stop = p["stop_ticks"] * TICK_SIZE
        for i in range(look, n - 1):
            if c[i] > h[i - look:i].max():
                j = int(bars["start"][i + 1])
                fill = float(tape["px"][j])
                et.append(j); dr.append(1)
                st.append(fill - stop); tg.append(fill + 2 * stop)
        if not et:
            return (np.array([], np.int64), np.array([], np.int8),
                    np.array([]), np.array([]))
        return (np.array(et, np.int64), np.array(dr, np.int8),
                np.array(st), np.array(tg))
'''

# Identical, except the target is the high of the bar it enters on -- which is not known
# when that bar opens. It is the exact mistake that scored a 79% win rate during
# development, and the reason this probe exists at all.
_PEEKER = _HONEST.replace('"probe_honest", "Honest (probe fixture)"',
                          '"probe_peeker", "Peeker (probe fixture)"') \
                 .replace("tg.append(fill + 2 * stop)", "tg.append(float(h[i + 1]) + stop)")


def selfcheck():
    # 1. Prompt assembly: the contract is pasted in whole, and the built-in names are
    #    live rather than a stale list that drifts as strategies are added.
    sp = system_python()
    assert "class MomentumBreak(Strategy)" in sp, "the template must be inside the prompt"
    assert "NO LOOKAHEAD" in sp and "TICK_SIZE" in sp
    for n in engine.LIBRARY:
        assert n in sp, f"built-in {n} missing from the name-collision rule"
    assert "<request>" in _user_python("do a thing", None)
    assert "sk-ant" not in sp

    # 2. Code extraction: the longest fence wins, and no fence is an error the user
    #    can act on rather than an IndexError.
    assert extract_code("blah\n```python\nx = 1\n```\n").strip() == "x = 1"
    assert extract_code("```\nab\n```\ntext\n```py\nlonger = 2\n```").strip() == "longer = 2"
    for bad in ("no code here", ""):
        try:
            extract_code(bad)
            raise AssertionError("a reply with no code block must be refused")
        except AuthorError as exc:
            assert "fenced code block" in str(exc)

    # 3. Key parsing: whatever the user pasted. The real key file started as a labelled
    #    line, which read verbatim produced a bare 401 with nothing to act on.
    fake = "sk-ant-api03-" + "A1b2C3d4" * 8
    for pasted in (fake, f"  {fake}\n", f"Anthropic API Key: {fake}",
                   f'ANTHROPIC_API_KEY="{fake}"', f"{fake}\n# a comment"):
        assert _key_from(pasted) == fake, pasted[:40]
    assert _key_from("") is None and _key_from(None) is None

    # 4. Redaction. The one thing in this file that must never fail.
    for text in (f"401 from api: x-api-key {fake} rejected",
                 f"Traceback ... api_key='{fake}'"):
        out = redact(text)
        assert fake not in out and "redacted" in out, out
    assert "sk-ant" not in redact(fake).replace("sk-ant-<redacted>", "")
    # and no prompt, header or status payload may carry one
    assert not _KEYISH.search(json.dumps(status()))
    assert not _KEYISH.search(sp + SYSTEM_CSHARP)

    # 4. THE LOOKAHEAD PROBE, on real ticks: it must clear the honest strategy and
    #    catch the peeker. This is the check the whole feature rests on.
    contracts = tp.cached_contracts()
    if contracts:
        ctx = engine.prepare(contracts[0], 5)
        good = plugins.load_source(_HONEST, "honest.py")
        bad = plugins.load_source(_PEEKER, "peeker.py")
        g = lookahead_probe(good, ctx=ctx, contract=contracts[0])
        b = lookahead_probe(bad, ctx=ctx, contract=contracts[0])
        assert g["ran"] and g["clean"], g["findings"][:2]
        assert b["ran"] and not b["clean"], "the probe failed to catch a known peeker"
        assert any("after the entry" in f or "changed when only LATER" in f
                   for f in b["findings"]), b["findings"]
        # 5. validate() must reject the peeker end to end, and say LOOKAHEAD.
        problems, info = validate(_PEEKER, contracts[0])
        assert problems.startswith("LOOKAHEAD:"), problems[:200]
        # and it must not leak the numbers that would make generation a free search
        problems2, info2 = validate(_HONEST, contracts[0])
        assert not problems2, problems2[:300]
        assert set(info2["smoke"]) == {"ran", "trades", "signals", "contract", "note"}, \
            info2["smoke"]
        probe_note = (f"{g['compared']} trade-comparisons over {g['cuts']} cuts on the "
                      f"honest one; the peeker died on cut 1")
    else:
        probe_note = "SKIPPED (no tape cached)"

    # 6. A translated file still gets the compile loop, the caveat header and the
    #    "translated by a model" note -- without calling the API.
    src, cls = nt8gen.render("orb", dict(range_min=15, stop_ticks=30, rr=1, one_per_day=1))
    res = nt8gen.generate("orb", dict(range_min=15, stop_ticks=30, rr=1, one_per_day=1),
                          source=src, class_name="PropSimOrbTranslated")
    assert res["class_name"] == "PropSimOrbTranslated"
    assert res["compiles"] in (True, None), res["attempts"]

    # 7. Provenance goes in the file, and says what the validation did and did not prove.
    hdr = _provenance("fade the opening range", "claude-opus-5",
                      dict(smoke=dict(contract="NQ 09-26", trades=12, signals=20),
                           lookahead=dict(ran=True, cuts=3, compared=9)))
    assert "NOT evidence of an edge" in hdr and "lookahead probe" in hdr
    assert all(ln.startswith("#") for ln in hdr.splitlines())

    st = status()
    print(f"selfcheck OK (offline, no API call): prompt carries the real contract and "
          f"all {len(engine.LIBRARY)} built-in names; code extraction and its failure "
          f"mode; keys redacted from errors, prompts and /api/ai; lookahead probe "
          f"cleared the honest fixture and caught the peeker [{probe_note}]; "
          f"validate() reports LOOKAHEAD and withholds P&L; a translated file goes "
          f"through the compiler. key={'yes' if st['key'] else 'no'}, "
          f"sdk={st['sdk'] or 'not installed'}, model={st['model']}")


def main():
    ap = argparse.ArgumentParser(description="Have Claude write a PropSim strategy.")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--write", metavar="DESCRIPTION")
    ap.add_argument("--name")
    ap.add_argument("--install", action="store_true", help="install it if it passes")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--translate", metavar="STRATEGY")
    ap.add_argument("--params", help="k=v,k=v for --translate")
    ap.add_argument("--contract")
    ap.add_argument("--timeframe", type=int, default=5)
    ap.add_argument("--model")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.status or not (args.write or args.translate):
        s = status()
        print(f"key:        {s['key_source'] or 'NOT FOUND — put it in ' + s['key_file']}")
        print(f"sdk:        {s['sdk'] or 'not installed — pip install anthropic'}")
        print(f"model:      {s['model']}")
        print(f"strategies: {s['strategies_dir']}")
        print(f"tape:       {', '.join(s['contracts']) or 'none cached'}")
        print(f"nt8c:       {'found' if s['nt8c'] else 'not installed (no compile check)'}")
        return

    def shout(a):
        state = "PASSED" if a.get("ok") else "rejected"
        print(f"attempt {a['attempt']}: {state}"
              + (f" — {a.get('name')}" if a.get("name") else ""))
        if a.get("problems"):
            print(f"    {a['problems'][:400]}")
        if a.get("errors"):
            for e in a["errors"][:6]:
                print(f"    {e}")

    if args.write:
        res = write_strategy(args.write, args.name, args.model, args.contract,
                             args.timeframe, on_attempt=shout)
        t = res["tokens"]
        print(f"\n{t['input']:,} input + {t['output']:,} output tokens on your key.")
        if not res["ok"]:
            raise SystemExit(f"gave up after {len(res['attempts'])} attempts. "
                             f"Last problem: {res['problems']}")
        if args.install:
            print(f"installed {install(res, overwrite=args.overwrite)}")
            print("It appears on the Backtest and Optimize tabs after a reload. Read it "
                  "first: nothing here says it has an edge.")
        else:
            print(res["header"] + "\n\n" + res["source"])
        return

    params = {}
    for kv in (args.params or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            params[k.strip()] = float(v)
    if not params:
        S = engine.LIBRARY.get(args.translate)
        params = {k: v.default for k, v in S.params.items()} if S else {}
    res = translate(args.translate, params, on_attempt=shout)
    t = res["tokens"]
    print(f"\n{t['input']:,} input + {t['output']:,} output tokens on your key.")
    if res["compiles"] is False:
        raise SystemExit("the translation never compiled; nothing was written.")
    print(f"wrote {nt8gen.write_out(res)}")
    for n in res["notes"]:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
