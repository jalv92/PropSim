#!/usr/bin/env python3
"""Append-only trial ledger: how many times you have looked.

    python3 ledger.py                 # the log, and what noise alone would score
    python3 ledger.py --verify
    python3 ledger.py --selfcheck

WHY THIS EXISTS, and it is the feature that separates this tool from a faster
overfitting machine. A backtest is a hypothesis test. Run twenty of them and the
best t-statistic you see is not evidence of an edge -- it is the maximum of
twenty draws, and the maximum of twenty draws from pure noise is about 1.9. A
tool that shows you t = 2.1 on your twentieth try, with no memory of the other
nineteen, is lying by omission.

So every run is recorded before its result is rendered, and the UI shows the
count next to the statistic together with the value noise alone would have
produced at that count. Nothing here blocks you. It just refuses to let the
denominator disappear.

WHAT COUNTS AS A TRIAL. Searching counts: a backtest, an imported Strategy
Analyzer run, a parameter sweep -- anything that could have come out differently
and be kept or discarded. Scoring does not: re-running the Monte Carlo on a trade
list you already have tells you about prop-firm rules, not about whether the
edge is real. Scoring runs are still logged (kind="score") so the record is
complete, but they do not inflate the multiplicity correction.

The file is append-only and hash-chained: each record carries the digest of the
previous one, so a deleted or edited entry is detectable rather than silent. It
is a lab notebook, not a database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import engine            # for MIN_T_DAYS only; engine never imports this module

# Degrees of freedom at the smallest sample `engine.summarise` will publish a
# daily t from. `threshold_t` prices its bar here by default.
_MIN_T_DF = engine.MIN_T_DAYS - 1

LEDGER = Path.home() / ".prop-sim" / "trials.jsonl"
SEARCH_KINDS = ("backtest", "import", "sweep")


def _digest(prev: str, body: dict) -> str:
    raw = (prev or "") + json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def read(path: Path | None = None) -> list[dict]:
    p = Path(path or LEDGER)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            out.append(dict(seq=None, corrupt=line[:120]))
    return out


def append(kind: str, path: Path | None = None, **fields) -> dict:
    """Record one trial. Called BEFORE the result is shown, never after.

    Ordering is the whole point: a ledger written after rendering is a ledger
    that loses exactly the runs a user aborts because they did not like the
    first glimpse of the number.
    """
    p = Path(path or LEDGER)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = read(p)
    prev = rows[-1].get("hash", "") if rows else ""
    body = dict(seq=len(rows) + 1, ts=datetime.now().isoformat(timespec="seconds"),
                kind=kind, **fields)
    rec = dict(body, prev=prev, hash=_digest(prev, body))
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def verify(path: Path | None = None) -> dict:
    """Walk the chain. Reports the first break rather than a boolean."""
    rows = read(path)
    prev = ""
    for i, r in enumerate(rows):
        if "hash" not in r:
            return dict(ok=False, rows=len(rows), broken_at=i + 1,
                        reason="unparseable line")
        body = {k: v for k, v in r.items() if k not in ("prev", "hash")}
        if r.get("prev", "") != prev or _digest(prev, body) != r["hash"]:
            return dict(ok=False, rows=len(rows), broken_at=r.get("seq", i + 1),
                        reason="record edited or an earlier one removed")
        prev = r["hash"]
    return dict(ok=True, rows=len(rows))


def expected_max_t(n: int) -> float:
    """What the best t out of `n` independent null trials looks like.

    The expected maximum of n standard normals, via Blom's order-statistic
    approximation. The obvious inverse-CDF-at-1/(n+1) version undershoots
    materially -- 1.67 against the true 1.87 at n = 20 -- and undershooting here
    flatters the user, which is the one direction this number must never err in.

    Reported next to the observed statistic because "t = 2.1" means something
    entirely different on trial 1 than on trial 40.
    """
    if n <= 1:
        return 0.0
    return NormalDist().inv_cdf((n - 0.375) / (n + 0.25))


def threshold_t(n: int, alpha: float = 0.05, df: int | None = None) -> float:
    """The |t| a two-sided 5% claim needs after n looks (Bonferroni).

    ON THE t DISTRIBUTION, NOT THE NORMAL, because the thing being compared
    against it is a t. `engine.summarise` computes it over daily totals and
    `engine.MIN_T_DAYS` floors that at 20 sessions, so it carries 19 degrees of
    freedom at the admissible minimum -- and a normal quantile understates the
    bar by 13% at n=20, 18% at n=200 and 22% at n=1000, a gap that WIDENS with
    the trial count because Bonferroni pushes further into the tail, which is
    exactly where the two distributions differ most. Understating flatters the
    user, and `expected_max_t` below says in as many words that this number must
    never err in that direction.

    `df` defaults to the floor. That is the most conservative admissible bar and
    the only honest default here: `stats` prices one number for a whole ledger
    of runs with different sample sizes, and cannot know any single one's.

    Cornish-Fisher off the normal quantile rather than an inverse incomplete
    beta: four terms, no dependency, and the residual at df >= 19 is under a
    tenth of a percent against published tables -- checked in `selfcheck`
    against t(19) and t(29) at three quantiles. scipy would be exact and is
    130 MB in a PyInstaller bundle for four lines of series.
    """
    n = max(int(n), 1)
    z = NormalDist().inv_cdf(1.0 - alpha / (2.0 * n))
    v = float(max(int(df if df is not None else _MIN_T_DF), 1))
    z2 = z * z
    return (z
            + z * (z2 + 1.0) / (4.0 * v)
            + z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / (96.0 * v * v)
            + z * (3.0 * z2 * z2 * z2 + 19.0 * z2 * z2 + 17.0 * z2 - 15.0)
            / (384.0 * v * v * v))


def fingerprint(**inputs) -> str:
    """Identity of a run's INPUTS, so an identical re-run is not a new look.

    "Any evaluation whose result influences a subsequent modelling choice" is a
    trial. Running the same strategy, on the same data, with the same parameters
    and the same costs a second time influences nothing -- it produces the
    identical number. Counting it again inflates the multiplicity debt and makes
    the noise ceiling unfairly harsh, which is just as dishonest as the
    undercounting this ledger exists to prevent.
    """
    clean = {k: v for k, v in inputs.items() if v is not None}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _t_sessions(r: dict) -> int:
    """How many sessions this record's daily t was computed over.

    THIS FILE IS APPEND-ONLY AND HASH-CHAINED, so a record written before
    `engine.MIN_T_DAYS` existed cannot be edited or removed -- and one of them
    is a three-session run that recorded t = 216.08 and became the "best result
    you have ever found" on every screen that compares against it. The record
    still carries the range it ran over, so the number can be recognised as
    unpublishable now without rewriting a single byte of history.

    Weekdays in `start..end`, which over-counts holidays. Over-counting keeps a
    borderline record IN, and the only thing this decides is whether a t is
    admissible -- so the error falls on the side of not silently discarding
    someone's result. `days` (sweeps record it) wins when present, being exact.
    """
    if isinstance(r.get("days"), int):
        return r["days"]
    a, b = r.get("start"), r.get("end")
    if not (isinstance(a, str) and isinstance(b, str)):
        return 0                      # no range recorded: cannot vouch for it
    try:
        d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    except ValueError:
        return 0
    return sum(1 for i in range((d1 - d0).days + 1)
               if (d0 + timedelta(days=i)).weekday() < 5)


def stats(path: Path | None = None) -> dict:
    rows = [r for r in read(path) if r.get("kind")]
    search = [r for r in rows if r["kind"] in SEARCH_KINDS]
    # Distinct looks: identical re-runs collapse, everything without a
    # fingerprint counts on its own.
    seen, dedup = set(), []
    for r in search:
        fp = r.get("fp")
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        dedup.append(r)
    repeats = len(search) - len(dedup)
    search = dedup
    ts = [float(r["t_daily"]) for r in search
          if isinstance(r.get("t_daily"), (int, float))
          and not math.isnan(float(r["t_daily"]))
          and _t_sessions(r) >= engine.MIN_T_DAYS]
    # A sweep is not one look. "A parameter change is a trial" -- so a record may
    # declare how many it stands for, and the multiplicity arithmetic uses that.
    # A 200-combination sweep raises the noise bar for every result that follows
    # it, permanently, and that is the honest accounting.
    n = sum(int(r.get("n_trials") or 1) for r in search)
    return dict(
        trials=n, repeats=repeats,
        scores=sum(1 for r in rows if r["kind"] not in SEARCH_KINDS),
        total=len(rows),
        best_t=max(ts, default=None), tested=len(ts),
        # Shipped with the stats so the screens can say WHY a t is missing
        # without keeping a second copy of the number. One source of truth.
        min_t_days=engine.MIN_T_DAYS,
        expected_max_t=round(expected_max_t(n), 2) if n else 0.0,
        threshold_t=round(threshold_t(n), 2) if n else 0.0,
        first=search[0]["ts"] if search else None,
        last=rows[-1]["ts"] if rows else None,
        path=str(Path(path or LEDGER)),
    )


def recent(n=25, path: Path | None = None) -> list[dict]:
    return read(path)[-n:][::-1]


def selfcheck():
    import tempfile
    p = Path(tempfile.mkdtemp(prefix="propsim-ledger-")) / "trials.jsonl"

    # `days` on every record because every real one carries it: a t that cannot
    # be tied to a sample size is not admissible (see `_t_sessions`), and a
    # fixture that omitted it was testing a record shape this never writes.
    a = append("backtest", path=p, strategy="orb", t_daily=1.1, days=30)
    b = append("backtest", path=p, strategy="orb", t_daily=2.4, days=30)
    append("score", path=p, firm="my_funded_futures")
    assert a["seq"] == 1 and b["prev"] == a["hash"], (a, b)
    assert verify(p)["ok"], verify(p)

    s = stats(p)
    # A scoring run must not inflate the multiplicity correction.
    assert s["trials"] == 2 and s["scores"] == 1, s
    assert s["best_t"] == 2.4, s

    # An identical re-run is the same look, not a new one.
    fp = fingerprint(strategy="orb", contract="NQ", params={"a": 1})
    append("backtest", path=p, fp=fp, strategy="orb", t_daily=3.0)
    append("backtest", path=p, fp=fp, strategy="orb", t_daily=3.0)
    s2 = stats(p)
    assert s2["trials"] == 3, s2          # 2 + one new fingerprint, not two
    assert s2["repeats"] == 1, s2
    # ...but a different parameter set is a different look
    append("backtest", path=p, fp=fingerprint(strategy="orb", contract="NQ",
                                              params={"a": 2}), t_daily=1.0)
    assert stats(p)["trials"] == 4, stats(p)

    # Tampering has to be detectable, or "append-only" is decoration.
    lines = p.read_text().splitlines()
    lines[0] = lines[0].replace('"t_daily": 1.1', '"t_daily": 9.9')
    p.write_text("\n".join(lines) + "\n")
    v = verify(p)
    assert not v["ok"] and v["broken_at"] == 1, v

    # Deleting a record must also be detectable.
    p.write_text("\n".join(p.read_text().splitlines()[1:]) + "\n")
    assert not verify(p)["ok"], verify(p)

    # THE 216. A three-session run recorded t = 216.08 and, because this file is
    # append-only and hash-chained, it is still sitting in a real ledger on disk
    # -- so `stats` has to recognise it from the range the record carries rather
    # than by editing it away. It still counts as a TRIAL: it was a look at the
    # data and the multiplicity debt is owed either way. It just is not a result.
    # On its own chain, so the trial counts asserted above stay exact.
    q = Path(tempfile.mkdtemp(prefix="propsim-ledger-")) / "trials.jsonl"
    append("backtest", path=q, strategy="orb", t_daily=2.4, days=30)
    append("backtest", path=q, strategy="latigo_break", t_daily=216.084,
           start="2026-07-26", end="2026-07-28")
    sq = stats(q)
    assert sq["best_t"] == 2.4, f"a 3-session t must not become the best: {sq}"
    assert sq["trials"] == 2 and sq["tested"] == 1, sq
    assert verify(q)["ok"], "recognising it must not require rewriting history"

    # The noise baseline has to rise with the count, and the 5% bar sit above it.
    assert expected_max_t(1) == 0.0
    assert expected_max_t(20) < expected_max_t(200)
    assert 1.80 < expected_max_t(20) < 1.95, expected_max_t(20)   # true value 1.87
    assert threshold_t(20) > expected_max_t(20)

    # THE BONFERRONI BAR IS A t QUANTILE, AND IT IS THE PUBLISHED ONE. Against a
    # printed table, at the two sample sizes that matter -- the MIN_T_DAYS floor
    # and a month -- across three quantiles, because a series that is right at
    # the middle and wrong in the tail is worse than useless here: Bonferroni
    # spends all its time in the tail. `alpha` is driven directly so each case
    # lands on a tabulated quantile.
    for df, q, published in ((19, 0.975, 2.093), (19, 0.995, 2.861),
                             (19, 0.9995, 3.883), (29, 0.975, 2.045),
                             (29, 0.995, 2.756), (29, 0.9995, 3.659)):
        got = threshold_t(1, alpha=2 * (1 - q), df=df)
        assert abs(got - published) / published < 0.002, (df, q, got, published)
    #     And it must sit ABOVE the normal quantile it replaced, at every trial
    #     count, by a margin that grows -- understating the bar flatters the user.
    gaps = [threshold_t(n) - NormalDist().inv_cdf(1 - 0.05 / (2 * n))
            for n in (20, 200, 1000)]
    assert all(g > 0 for g in gaps) and gaps == sorted(gaps), gaps

    print(f"selfcheck OK: chain verified then broken by an edit and by a deletion; "
          f"2 trials + 1 score counted separately; "
          f"noise-expected best t at 20 trials = {expected_max_t(20):.2f}, "
          f"5% bar = {threshold_t(20):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("-n", type=int, default=25)
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.verify:
        v = verify()
        print(f"{v['rows']} records — " + ("chain intact" if v["ok"] else
              f"BROKEN at #{v['broken_at']}: {v['reason']}"))
        raise SystemExit(0 if v["ok"] else 1)

    s = stats()
    if not s["total"]:
        print(f"No trials recorded yet ({s['path']}).")
        return
    print(f"{s['trials']} search trials + {s['scores']} scoring runs "
          f"({s['first']} .. {s['last']})")
    if s["best_t"] is not None:
        verdict = ("above" if s["best_t"] > s["expected_max_t"] else "BELOW")
        print(f"best t (daily) {s['best_t']:.2f} — noise alone would reach "
              f"{s['expected_max_t']:.2f} over {s['trials']} trials ({verdict} it); "
              f"a 5% claim needs {s['threshold_t']:.2f}")
    print(f"\n{'#':>4} {'when':<20}{'kind':<10}{'what':<28}{'t':>7}{'P&L':>12}")
    for r in recent(args.n):
        what = " ".join(str(r.get(k)) for k in ("strategy", "contract", "timeframe")
                        if r.get(k) is not None) or str(r.get("firm", ""))
        t = r.get("t_daily")
        t = f"{t:7.2f}" if isinstance(t, (int, float)) and not math.isnan(t) else f"{'—':>7}"
        pnl = r.get("pnl")
        pnl = f"{pnl:12,.0f}" if isinstance(pnl, (int, float)) else f"{'—':>12}"
        print(f"{r.get('seq', '?'):>4} {r.get('ts', '')[:19]:<20}{r.get('kind', ''):<10}"
              f"{what[:27]:<28}{t}{pnl}")


if __name__ == "__main__":
    main()
