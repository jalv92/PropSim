#!/usr/bin/env python3
"""Volumetric bars from the tick tape: the numbers NinjaTrader's Order Flow
Volumetric bars print, rebuilt from the same .ncd ticks.

Verified 2026-09-05 against a live NT8 chart (ES 09-26, 5 Min Volumetric,
2026-08-27 09:30 bar): volume, buy/sell volume, delta, min/max delta, trade
count, POC and every bid/ask level matched exactly. So a strategy that reads
delta or imbalances can be evaluated here on any timeframe, tick-true, with
`engine.py`'s pessimistic fills -- instead of eyeballing charts in Playback.

    import tape, footprint
    t = tape.build_cache("ES 09-26")               # once per contract
    s = footprint.bar_stats(t, secs=300)           # arrays, one row per bar
    fp = footprint.levels(t, s["start"][i], s["end"][i])   # one bar's ladder

Run `python3 footprint.py --selfcheck` to re-verify against the NT8 anchor.
"""
import sys
import numpy as np
import tape

IMBALANCE = 3.0     # NT8 default ratio for diagonal bid/ask imbalances
IMB_MIN = 20        # ignore imbalances on levels with fewer contracts than this


def bar_stats(t: dict, secs: int = 300) -> dict:
    """tape.build_bars + the delta family per bar, vectorised over the tape.

    Adds buy, sell, min_delta, max_delta (running delta inside the bar, the
    'Min delta' / 'Max delta' rows of the NT8 bar statistics) and poc.
    """
    b = tape.build_bars(t, secs)
    if not len(b["t"]):
        return b
    vol, side = t["vol"].astype(np.int64), t["side"].astype(np.int64)
    starts, ends = b["start"], b["end"]
    b["buy"] = np.add.reduceat(np.where(side > 0, vol, 0), starts)
    b["sell"] = np.add.reduceat(np.where(side < 0, vol, 0), starts)
    run = np.cumsum(side * vol)
    base = np.concatenate(([0], run[ends[:-1] - 1]))   # running delta at bar open
    rel = run - np.repeat(base, ends - starts)
    b["min_delta"] = np.minimum(np.minimum.reduceat(rel, starts), 0)
    b["max_delta"] = np.maximum(np.maximum.reduceat(rel, starts), 0)
    b["poc"] = np.array([_poc(t, s, e) for s, e in zip(starts, ends)])
    return b


def _poc(t, s, e):
    lv, inv = np.unique(t["px"][s:e], return_inverse=True)
    return float(lv[np.argmax(np.bincount(inv, weights=t["vol"][s:e]))])


def levels(t: dict, s: int, e: int, tick: float | None = None) -> list[dict]:
    """The bid/ask ladder of ticks [s, e): [{p, bid, ask, buy_imb?, sell_imb?}] top-down.

    Imbalances are diagonal like NT8's: ask at P against bid one tick below,
    bid at P against ask one tick above, ratio >= IMBALANCE.
    """
    px, vol, side = t["px"][s:e], t["vol"][s:e], t["side"][s:e]
    tick = tick or float(np.min(np.diff(np.unique(px)))) if len(px) > 1 else 0.25
    lv = np.round(px / tick).astype(np.int64)
    lo, hi = lv.min(), lv.max()
    bid = np.bincount(lv - lo, weights=np.where(side < 0, vol, 0), minlength=hi - lo + 1)
    ask = np.bincount(lv - lo, weights=np.where(side > 0, vol, 0), minlength=hi - lo + 1)
    out = []
    for i in range(hi - lo, -1, -1):
        r = {"p": (lo + i) * tick, "bid": int(bid[i]), "ask": int(ask[i])}
        if i > 0 and ask[i] >= IMB_MIN and ask[i] >= IMBALANCE * max(bid[i - 1], 1):
            r["buy_imb"] = True
        if i < hi - lo and bid[i] >= IMB_MIN and bid[i] >= IMBALANCE * max(ask[i + 1], 1):
            r["sell_imb"] = True
        out.append(r)
    return out


def selfcheck():
    """Anchor: the ES 09-26 2026-08-27 09:30 5-min bar as NT8 displayed it."""
    t = tape.build_cache("ES 09-26")
    s = bar_stats(t, 300)
    when = tape.to_net(__import__("datetime").datetime(2026, 8, 27, 9, 30))
    i = int(np.searchsorted(s["t"], when))      # bar "t" is its first tick, not the slot edge
    assert i < len(s["t"]) and s["t"][i] - when < 300 * tape.TPS, \
        "09:30 bar not on the tape (download ES 09-26 ticks for 2026-08-27)"
    got = dict(v=int(s["v"][i]), delta=int(s["delta"][i]), buy=int(s["buy"][i]), sell=int(s["sell"][i]),
               mn=int(s["min_delta"][i]), mx=int(s["max_delta"][i]), n=int(s["n"][i]), poc=float(s["poc"][i]))
    want = dict(v=43586, delta=1566, buy=22576, sell=21010, mn=-82, mx=1701, n=33933, poc=7706.5)
    assert got == want, f"{got} != NT8 {want}"
    fp = {r["p"]: (r["bid"], r["ask"]) for r in levels(t, s["start"][i], s["end"][i])}
    assert fp[7720.0] == (0, 6) and fp[7719.75] == (12, 72) and fp[7715.75] == (533, 614), fp
    assert fp[7710.0] == (573, 546) and fp[7706.5] == (657, 599), fp
    assert fp[7706.25] == (562, 671) and fp[7705.75] == (598, 479) and fp[7719.5] == (57, 103), fp
    assert int(s["v"][i - 1]) == 6499 and int(s["delta"][i - 1]) == 595, "09:25 bar"
    print("footprint selfcheck OK: 2026-08-27 09:30 bar matches NT8 exactly")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        print(__doc__)
