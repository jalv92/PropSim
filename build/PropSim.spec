# PyInstaller spec for PropSim. Checked in so builds are reproducible.
#
#   py -m PyInstaller --noconfirm --clean build/PropSim.spec
#
# onedir, not onefile: onefile unpacks to a temp folder on every launch, which
# is slower and is the shape antivirus heuristics complain about most. An
# installer ships a folder anyway, so onedir costs nothing and starts faster.
#
# console=False -- the user double-clicks a Start Menu entry and a browser
# opens; a stray terminal window reads as a bug in a shipped product.

import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(ROOT, "dashboard.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "prop_rules.json"), "."),
        (os.path.join(ROOT, "account_patterns.json"), "."),
        (os.path.join(ROOT, "dashboard.html"), "."),
        # NinjaScript templates: without them the export tab has nothing to render
        (os.path.join(ROOT, "nt8gen"), "nt8gen"),
    ],
    hiddenimports=["nttrades", "ntdata", "ncd_parse", "prop_rules", "sim",
                   "tape", "engine", "optimize", "plugins", "ledger", "ntimport",
                   "slippage", "nt8gen", "aiauthor",
                   # The AI author's SDK. Bundled so the tab works in the packaged
                   # build; aiauthor imports it lazily, so a build without it
                   # (PyInstaller simply omits a missing module) loses that tab and
                   # nothing else.
                   "anthropic"],
    hookspath=[],
    excludes=[
        # numpy is required; these are not, and each drags in tens of MB
        "matplotlib", "tkinter", "PIL", "pandas", "scipy", "pytest",
        "IPython", "notebook", "setuptools", "pip",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PropSim",
    debug=False,
    strip=False,
    upx=False,          # UPX compression is a strong antivirus false-positive trigger
    console=False,
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False,
    upx=False,
    name="PropSim",
)
