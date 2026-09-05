#!/usr/bin/env python3
"""Synthetic tick days, bootstrapped from the real tape.

Each synthetic day is a chain of blocks (5-30 min) cut from random real days
at the same time of session, glued by price DIFFERENCES so the path is
continuous. Volumes, sides, sub-second timing and the intraday profile
(open, close, 08:30 news) are the real ones; the sequence of moves is not.
What this teaches a model: survival, sizing, position management under real
microstructure. What it cannot teach: edge -- there is no information here
that the real 60-90 days did not have.

Output uses the tape-cache schema (ts, px, vol, side), so engine.py,
footprint.py and everything else read a synthetic day like a real one.

    python3 synth.py "ES 09-26" --days 1250 --seed 1     # ~/.prop-sim/synth/ES_09-26/
    python3 synth.py "ES 09-26" --selfcheck
    import synth; t = synth.load("ES 09-26", 17)         # day #17 as a tape dict
"""
import argparse, json, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import tape

SESSION_START = 18 * 3600          # ET, previous calendar day
SLOT = 60                          # block grid, seconds
SLOTS = 23 * 3600 // SLOT          # 18:00 -> 17:00
BLOCK_MIN, BLOCK_MAX = 5, 30       # block length, slots (minutes)
MIN_TICKS = 100_000                # a real day below this is a holiday / hole: not a source
SYNTH_EPOCH = datetime(2030, 1, 1) # synthetic dates start here -- obviously not real
OUT = Path.home() / ".prop-sim" / "synth"


def _sources(t: dict) -> list[dict]:
    """Real trading days (18:00 -> 17:00) as per-day slot tables."""
    ts = t["ts"]
    sec = tape.sec_of_day(ts)
    sess = (sec - SESSION_START) % 86400                       # seconds into the session
    tday = tape.day_index(ts) - (sec < SESSION_START)          # trading day = calendar day of its close
    days = []
    for d in np.unique(tday):
        m = np.flatnonzero(tday == d)
        if len(m) < MIN_TICKS:
            continue
        slot = sess[m] // SLOT
        starts = np.searchsorted(slot, np.arange(SLOTS + 1))   # tick index where each slot begins
        days.append(dict(idx=m, slot_start=starts, sess=sess[m], date=tape.date_str(int(d))))
    return days


def generate_day(t: dict, days: list[dict], rng: np.random.Generator, tick: float) -> dict:
    px_i = np.round(t["px"] / tick).astype(np.int64)
    start = days[rng.integers(len(days))]
    price = int(px_i[start["idx"][0]])
    ts_out, px_out, vol_out, side_out = [], [], [], []
    slot = 0
    while slot < SLOTS:
        d = days[rng.integers(len(days))]
        n = int(rng.integers(BLOCK_MIN, BLOCK_MAX + 1))
        a, b = d["slot_start"][slot], d["slot_start"][min(slot + n, SLOTS)]
        if b > a:
            sel = d["idx"][a:b]
            p = px_i[sel]
            p = p - p[0] + price                                # glue by difference
            price = int(p[-1])
            ts_out.append(d["sess"][a:b] * tape.TPS + t["ts"][sel] % tape.TPS)
            px_out.append(p); vol_out.append(t["vol"][sel]); side_out.append(t["side"][sel])
        slot += n
    px = np.concatenate(px_out).astype(np.float64) * tick
    return dict(ts=np.concatenate(ts_out), px=px.astype(np.float32),
                vol=np.concatenate(vol_out), side=np.concatenate(side_out))


def day_base(n: int) -> int:
    """.NET ticks of 18:00 the evening before synthetic day #n (weekdays only)."""
    d = SYNTH_EPOCH + timedelta(days=n + n // 5 * 2)           # skip weekends
    return tape.to_net(d - timedelta(days=1)) + SESSION_START * tape.TPS


def path(contract: str, n: int) -> Path:
    return OUT / contract.replace(" ", "_") / f"{n:05d}.npz"


def load(contract: str, n: int) -> dict:
    z = np.load(path(contract, n))
    return {k: z[k] for k in ("ts", "px", "vol", "side")}


def generate(contract: str, n_days: int, seed: int, rebuild=False):
    t = tape.build_cache(contract, force=rebuild)
    _, tick = __import__("engine").instrument(contract)
    days = _sources(t)
    out = path(contract, 0).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(dict(
        contract=contract, seed=seed, n_days=n_days, tick=tick, block_minutes=[BLOCK_MIN, BLOCK_MAX],
        sources=[d["date"] for d in days], generated=datetime.now().isoformat(timespec="seconds")), indent=1))
    t0 = time.time()
    for n in range(n_days):
        p = path(contract, n)
        if p.exists():
            continue
        rng = np.random.default_rng([seed, n])                  # day n is reproducible on its own
        d = generate_day(t, days, rng, tick)
        d["ts"] = d["ts"] + day_base(n)
        np.savez_compressed(p, **d)
        if n % 50 == 0:
            print(f"{contract}: day {n}/{n_days}  {len(d['ts'])} ticks  {time.time() - t0:.0f}s", flush=True)
    print(f"{contract}: done, {n_days} days in {out}, {sum(f.stat().st_size for f in out.glob('*.npz')) >> 20} MB")


def selfcheck(contract: str):
    t = tape.build_cache(contract)
    _, tick = __import__("engine").instrument(contract)
    days = _sources(t)
    assert len(days) >= 20, f"only {len(days)} usable source days"
    a = generate_day(t, days, np.random.default_rng([7, 3]), tick)
    b = generate_day(t, days, np.random.default_rng([7, 3]), tick)
    assert np.array_equal(a["px"], b["px"]) and np.array_equal(a["ts"], b["ts"]), "same seed must give the same day"
    assert np.all(np.diff(a["ts"]) >= 0), "time must not go backwards"
    jumps = np.abs(np.diff(a["px"].astype(np.float64)))
    real_jumps = np.abs(np.diff(t["px"].astype(np.float64)))
    assert jumps.max() <= real_jumps.max() + 1e-6, "a synthetic jump larger than any real one"
    real_n = [len(d["idx"]) for d in days]
    assert 0.5 * min(real_n) <= len(a["ts"]) <= 1.5 * max(real_n), "tick count outside the real range"
    # hourly volume profile must look like the real one (the whole point of bootstrapping by time of day)
    def profile(ts, vol):
        h = ((tape.sec_of_day(ts) - SESSION_START) % 86400) // 3600
        return np.bincount(h, weights=vol, minlength=23) / vol.sum()
    a["ts"] = a["ts"] + day_base(0)
    corr = np.corrcoef(profile(a["ts"], a["vol"]), profile(t["ts"], t["vol"]))[0, 1]
    assert corr > 0.9, f"volume profile corr {corr:.2f}"
    print(f"synth selfcheck OK ({contract}): {len(days)} source days, {len(a['ts'])} ticks/day, profile corr {corr:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("contract")
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rebuild", action="store_true", help="re-parse the .ncd files first (new downloads)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    selfcheck(a.contract) if a.selfcheck else generate(a.contract, a.days, a.seed, a.rebuild)
