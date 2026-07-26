#!/usr/bin/env python3
"""Package the NinjaScript add-on as an importable NinjaTrader archive.

    python3 nt8/build_zip.py                 # -> nt8/dist/PropSimExport.zip
    python3 nt8/build_zip.py --version 8.1.1.0

NinjaTrader validates the archive, so the structure is not negotiable: an
`Info.xml` manifest at the ROOT, and sources under their NinjaScript TYPE folder
at paths relative to `bin/Custom`. A plain zip of the .cs files is rejected with
"not a NinjaScript archive".

The declared version must be <= the importing user's NinjaTrader build, or it
reads as "made from a newer version" and is refused. The default is therefore
deliberately old: it imports on any 8.1.x. Source is shipped rather than a DLL so
the user's own NinjaTrader compiles it, which is what survives build changes.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [
    "AddOns/PropSimExportWriter.cs",
    "OptimizationFitnesses/PropSimFitness.cs",
    "PerformanceMetrics/PropSimTrades.cs",
]
INFO = """<?xml version="1.0" encoding="utf-8"?>
<NinjaTrader>
\t<Export>
\t\t<Version>{version}</Version>
\t</Export>
</NinjaTrader>
"""


def build(version="8.1.1.0", out: Path | None = None) -> Path:
    out = out or (HERE / "dist" / "PropSimExport.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (HERE / f).exists()]
    if missing:
        raise SystemExit(f"missing sources: {missing}")
    # Deterministic: a fixed timestamp keeps the committed artifact from
    # changing on every rebuild, so a diff means the code changed.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        def add(name, data):
            zi = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
        add("Info.xml", INFO.format(version=version))
        for f in FILES:
            add(f, (HERE / f).read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="8.1.1.0",
                    help="declared NinjaTrader version (must be <= the user's build)")
    ap.add_argument("--out")
    args = ap.parse_args()
    p = build(args.version, Path(args.out) if args.out else None)
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
    print(f"{p}  ({p.stat().st_size:,} bytes, version {args.version})")
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    main()
