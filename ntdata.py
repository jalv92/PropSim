#!/usr/bin/env python3
"""Discover and inventory a NinjaTrader 8 data folder.

This is the whole of the end user's configuration: point at the NinjaTrader 8
folder once, and everything downstream is derived. Nothing else is asked.

    python3 ntdata.py                    # auto-detect and inventory
    python3 ntdata.py "D:/NinjaTrader 8" # explicit root

What is readable from outside NinjaTrader, and what is not:

    db/tick/<contract>/*.Last.ncd   READABLE. Proprietary but decoded by
                                    ncd_parse.py; carries the bid/ask at each
                                    trade, so the true aggressor side is
                                    recoverable. This is the product's tape.

    db/replay/<contract>/*.nrd      HEADER ONLY. The 44x80-byte header is
                                    plain and checksummed, so days, instrument,
                                    session bounds and depth quality are all
                                    readable. The EVENT STREAM is not: it
                                    carries an out-of-spec volume opcode that
                                    silently corrupts hand-written parsers a
                                    few KB in (verified on four files,
                                    2026-07-25). Ground truth for the stream is
                                    NinjaTrader's own MarketReplay.
                                    DumpMarketDepth, which only exists inside
                                    the NinjaTrader process.

So Market Replay is inventoried and reported, never replayed. Saying that in
the UI is better than a parser that returns plausible garbage.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

NET_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)
HDR_SLOT = struct.Struct("<didddddiqqq")     # last,count,max,min,first,1.0,tick,flag,t0,t1,vol
SLOT_BYTES = 80
L2_ASK, L2_BID = 10, 11                      # header slots carrying market depth

WINDOWS_DEFAULTS = [
    Path.home() / "Documents" / "NinjaTrader 8",
    Path.home() / "OneDrive" / "Documents" / "NinjaTrader 8",
    Path("C:/Users/Public/Documents/NinjaTrader 8"),
]


def candidate_roots():
    """Plausible NinjaTrader 8 folders, including the WSL view of C:."""
    out = list(WINDOWS_DEFAULTS)
    users = Path("/mnt/c/Users")
    if users.is_dir():
        for u in users.iterdir():
            if u.name in ("Public", "Default", "All Users"):
                continue
            out += [u / "Documents" / "NinjaTrader 8",
                    u / "OneDrive" / "Documents" / "NinjaTrader 8"]
    return out


def autodetect():
    for p in candidate_roots():
        try:
            if (p / "db").is_dir():
                return p
        except OSError:
            continue
    return None


def _ncd_span(path: Path):
    """(first_tick_datetime, tick_estimate) without decoding the whole file."""
    try:
        raw = path.read_bytes()[:28]
        if len(raw) < 28:
            return None, 0
        _inc, _p0, t0 = struct.unpack_from("<ddq", raw, 4)
        return NET_EPOCH + timedelta(microseconds=t0 / 10), path.stat().st_size
    except OSError:
        return None, 0


def read_nrd_header(path: Path):
    """Header-only read of a Market Replay file: day quality without decoding.

    `count` and `volumeSum` are checksummed against the full decode, so the
    mean depth-event size falls out of slots 10+11 arithmetically -- which is
    how depth is audited without touching the unparseable stream.
    """
    try:
        with path.open("rb") as fh:
            buf = fh.read(SLOT_BYTES * 44)
    except OSError:
        return None
    if len(buf) < SLOT_BYTES * 44:
        return None
    depth_n = depth_vol = 0
    tape_vol = 0
    for slot in (L2_ASK, L2_BID, 2):
        try:
            f = HDR_SLOT.unpack_from(buf, slot * SLOT_BYTES)
        except struct.error:
            continue
        count, volsum = f[1], f[10]
        if slot == 2:
            tape_vol = volsum
        else:
            depth_n += count
            depth_vol += volsum
    return dict(depth_events=depth_n, tape_volume=tape_vol,
                mean_depth_size=round(depth_vol / depth_n, 2) if depth_n else 0.0)


_INV_CACHE: dict = {}


def tick_file_day(name: str):
    """The ET calendar day an hourly .ncd actually holds, from its name.

    NT8 names an hourly tick file after the END of the hour it covers, so
    `202608070000.Last.ncd` is 23:00-24:00 on 2026-08-06 -- verified against
    the file's own header timestamp -- and not a single tick in it belongs to
    the 7th. Reading the date off the name verbatim invents a day that no
    cache can ever cover, which is not cosmetic: every "your cache stops
    before your data does" comparison downstream is `raw_last > cache_last`,
    so for any contract whose download ends on a midnight boundary (an
    expired month, a partial download) it is permanently true. The Rebuild
    button then comes back the instant it finishes, and building ALL
    re-parses those contracts from scratch every single time.

    None for a name that is not `YYYYMMDDHH...` -- a stray file in the folder
    is not a reason to take the whole Data tab down.
    """
    try:
        return (datetime.strptime(name[:10], "%Y%m%d%H")
                - timedelta(hours=1)).strftime("%Y%m%d")
    except ValueError:
        return None


def inventory(root: Path, force: bool = False):
    """Everything the app knows about a user's NinjaTrader folder.

    Cached, because this is expensive and the answer changes rarely. A tick
    folder holds thousands of hourly files and the original implementation
    called Path.stat() on every one of them -- measured at 10.4 s on a real
    folder, every time the Data tab was opened. Two fixes: os.scandir(), whose
    DirEntry carries the size without a second syscall, and this cache. The
    Rescan button passes force=True.
    """
    root = Path(root)
    key = str(root)
    if not force and key in _INV_CACHE:
        return _INV_CACHE[key]
    db = root / "db"
    out = dict(root=str(root), ok=db.is_dir(), tick={}, replay={}, notes=[])
    if not out["ok"]:
        out["notes"].append(f"no 'db' folder under {root} — is this the "
                            f"NinjaTrader 8 folder?")
        return out

    tick = db / "tick"
    if tick.is_dir():
        for cdir in sorted(p for p in tick.iterdir() if p.is_dir()):
            # one scandir pass: name and size come from the same directory
            # entry, instead of a stat() syscall per file
            with os.scandir(cdir) as it:
                files = [(e.name, e.stat().st_size) for e in it
                         if e.is_file() and e.name.endswith(".Last.ncd")]
            days = sorted({d for d in (tick_file_day(n) for n, _ in files) if d})
            if not days:
                continue
            size = sum(sz for _, sz in files)
            out["tick"][cdir.name] = dict(
                days=len(days), first=days[0], last=days[-1],
                files=len(files), mb=round(size / 1e6, 1),
                # ~5.6 bytes/tick measured across this format
                est_ticks=int(size / 5.6), usable=True)
    else:
        out["notes"].append("no db/tick folder — no tape available")

    replay = db / "replay"
    if replay.is_dir():
        for cdir in sorted(p for p in replay.iterdir() if p.is_dir()):
            with os.scandir(cdir) as it:
                entries = [(e.name, e.path, e.stat().st_size) for e in it
                           if e.is_file() and e.name.endswith(".nrd")]
            if not entries:
                continue
            files = [Path(pth) for _, pth, _ in entries]
            days = sorted({n[:8] for n, _, _ in entries})
            size = sum(sz for _, _, sz in entries)
            # one header read, on the biggest file, to gauge depth quality
            sample = read_nrd_header(Path(max(entries, key=lambda e: e[2])[1]))
            out["replay"][cdir.name] = dict(
                days=len(days), first=days[0], last=days[-1],
                gb=round(size / 1e9, 2),
                mean_depth_size=(sample or {}).get("mean_depth_size", 0.0),
                usable=False,
                note="inventory only — the event stream cannot be decoded "
                     "outside NinjaTrader")
    if out["replay"]:
        out["notes"].append(
            f"{sum(v['days'] for v in out['replay'].values())} Market Replay "
            "days found. These are catalogued but NOT replayable: the stream "
            "requires NinjaTrader's own decoder. The tape (db/tick) is used "
            "instead.")
    if not out["tick"]:
        out["notes"].append("No usable tape found. Your trades are still read "
                            "from NinjaTrader's database.")
    _INV_CACHE[key] = out
    return out


CONFIG = Path.home() / ".prop-sim" / "config.json"


def load_config():
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=1))
    return cfg


def resolve_root(explicit=None):
    """Explicit path > saved config > autodetect. Returns None if nothing."""
    if explicit:
        return Path(explicit)
    saved = load_config().get("nt_root")
    if saved and (Path(saved) / "db").is_dir():
        return Path(saved)
    return autodetect()


def selfcheck():
    # THE POINT: the midnight file. Its name says the 7th, every tick in it is
    # the 6th, and calling it the 7th is what made "your cache is out of date"
    # permanently true for a contract whose data stops at that boundary.
    assert tick_file_day("202608070000.Last.ncd") == "20260806"
    assert tick_file_day("202608062300.Last.ncd") == "20260806"
    assert tick_file_day("202608060100.Last.ncd") == "20260806"
    assert tick_file_day("202601010000.Last.ncd") == "20251231"   # and years
    assert tick_file_day("notatimestamp.Last.ncd") is None
    print("selfcheck OK: hourly .ncd names map to the ET day they hold")


def main():
    if "--selfcheck" in sys.argv[1:]:
        return selfcheck()
    root = resolve_root(sys.argv[1] if len(sys.argv) > 1 else None)
    if root is None:
        print("No NinjaTrader 8 folder found. Pass one:")
        print('  python3 ntdata.py "C:/Users/you/Documents/NinjaTrader 8"')
        for c in candidate_roots()[:4]:
            print(f"    tried {c}")
        raise SystemExit(1)
    inv = inventory(root)
    print(f"NinjaTrader 8 folder: {inv['root']}   ok={inv['ok']}\n")
    if inv["tick"]:
        print("TAPE (db/tick) — usable")
        print(f"  {'contract':<14}{'days':>5}{'files':>7}{'MB':>8}"
              f"{'est ticks':>12}   range")
        for k, v in inv["tick"].items():
            print(f"  {k:<14}{v['days']:>5}{v['files']:>7}{v['mb']:>8.1f}"
                  f"{v['est_ticks']:>12,}   {v['first']}..{v['last']}")
    if inv["replay"]:
        print("\nMARKET REPLAY (db/replay) — inventory only, not replayable")
        print(f"  {'contract':<14}{'days':>5}{'GB':>8}{'mean depth':>12}   range")
        for k, v in inv["replay"].items():
            print(f"  {k:<14}{v['days']:>5}{v['gb']:>8.2f}"
                  f"{v['mean_depth_size']:>12.2f}   {v['first']}..{v['last']}")
    for n in inv["notes"]:
        print(f"\n  note: {n}")


if __name__ == "__main__":
    main()
