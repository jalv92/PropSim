#!/usr/bin/env python3
"""Prop-firm rule set: canonical data + loader.

Source of truth for every prop-firm rule the simulator applies. Built by
normalizing the five per-firm research files in `research/*.json` (which stay
as provenance) into one flat canonical table.

    python3 prop_rules.py --build      # regenerate prop_rules.json
    python3 prop_rules.py              # self-check + summary
    python3 prop_rules.py --list mff   # show one firm's variants

Consumers:
    from prop_rules import load, select
    rs = select("my_funded_futures", "rapid", "evaluation", 50_000)
    rs.max_dd, rs.hwm_basis, rs.breach_basis, ...

DESIGN NOTE — the two-axis drawdown. Every firm's drawdown needs BOTH:
  hwm_basis   what RATCHETS the floor upward: none | eod_close | intraday_equity
  breach_basis what is TESTED against the floor: realized | intraday_equity | None
Four of five firms breach on realtime equity including unrealized P&L while
ratcheting only on the end-of-day close. Collapsing these into one field
silently inflates pass rates, because a strategy that dips below the floor
intraday and recovers by the close is dead in reality and alive in the sim.
Lucid is UNVERIFIED on breach_basis and is deliberately left None -- see
docs/superpowers/specs/2026-07-25-prop-firm-rules-matrix.md.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict, fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE / "research"          # per-firm provenance, shipped with the code


def _res(name):
    """Locate a bundled data file, whether running from source or frozen.

    PyInstaller extracts --add-data payloads to sys._MEIPASS at runtime; from
    source they sit next to this module.
    """
    import sys as _sys
    base = getattr(_sys, "_MEIPASS", None)
    return (Path(base) / name) if base else (Path(__file__).resolve().parent / name)

CANON = _res("prop_rules.json")

# hwm_basis / breach_basis vocabulary
NONE, EOD, INTRA = "none", "eod_close", "intraday_equity"
REALIZED = "realized"


@dataclass
class RuleSet:
    """One (firm, variant, phase, size) account. All money in USD."""
    firm: str
    variant: str
    phase: str
    size: int
    start_balance: float
    profit_target: float | None
    profit_target_basis: str | None      # UNVERIFIED for every firm -- see matrix
    max_dd: float | None
    hwm_basis: str                       # none | eod_close | intraday_equity
    breach_basis: str | None             # realized | intraday_equity | None=UNVERIFIED
    dd_lock_offset: float | None         # floor stops at start+offset; None = never
    daily_loss_limit: float | None
    dll_soft: bool                       # soft pause (locks the day) vs hard breach
    consistency_pct: float | None
    consistency_effect: str | None       # blocks_pass | inflates_target | blocks_payout
    min_days: int | None
    qualifying_day_min_profit: float | None
    max_contracts: int | None
    micro_ratio: float | None
    profit_split: float | None
    buffer_required: float | None
    min_payout: float | None
    max_withdrawal: float | None
    automation_allowed: bool | None
    account_cost: float | None
    reset_cost: float | None
    unmodeled_rules: list
    verified: bool
    retrieved: str
    source_url: str | None
    breach_source: str | None = None   # provenance for a firm-level breach_basis

    @property
    def dd_floor_lock(self) -> float | None:
        """Absolute account value at which the drawdown floor stops trailing."""
        if self.dd_lock_offset is None:
            return None
        return self.start_balance + self.dd_lock_offset

    def warnings(self) -> list[str]:
        """Everything a caller must surface before trusting a simulation."""
        w = []
        if self.breach_basis is None:
            w.append("breach_basis UNVERIFIED - cannot model the drawdown breach "
                     "correctly; results are provisional")
        if not self.verified:
            w.append("rule set not fully verified against a primary source")
        if self.automation_allowed is False:
            w.append("THIS FIRM PROHIBITS AUTOMATED STRATEGIES - any P(pass) here "
                     "is not actionable for an automated system")
        if self.profit_target_basis is None and self.profit_target:
            w.append("profit_target_basis UNVERIFIED (closing balance vs intraday touch)")
        if self.unmodeled_rules:
            w.append(f"{len(self.unmodeled_rules)} rule(s) not modelled by the simulator")
        return w


# --------------------------------------------------------------------------
# normalization helpers
# --------------------------------------------------------------------------

def _num(v):
    """Coerce a research value to a float, or None. Never guesses."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):                      # e.g. {"amount": 150, "period": "month"}
        return _num(v.get("amount"))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("none", "n/a", "null", "unverified", "unver", ""):
            return None
        m = re.search(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
        if not m:
            return None
        try:
            return float(m.group().rstrip(",.").replace(",", ""))
        except ValueError:
            return None
    return None


def _size_to_int(key) -> int:
    return int(_num(key) * 1000)                 # "50K"/"50k" -> 50000


def _bases_from_dd_type(row) -> tuple[str, str | None]:
    """Derive (hwm_basis, breach_basis) from whatever the researcher wrote.

    Prefers explicit fields when present (Lucid, Apex supplied them); otherwise
    reads the dd_type prose. Returns breach_basis=None when the source does not
    state it -- never defaults to the majority.
    """
    hwm = row.get("hwm_basis")
    breach = row.get("breach_basis")
    if hwm:
        hwm = (INTRA if "intraday" in hwm else EOD if "eod" in hwm
               else NONE if "none" in hwm or "static" in hwm else hwm)
    else:
        t = str(row.get("dd_type", "")).lower()
        # The RATCHET is described by the leading clause; a parenthetical (or a
        # "breach ..." clause) describes the BREACH and must not be read here.
        # Topstep writes "END_OF_DAY_TRAILING (breach evaluated REALTIME ...)" --
        # matching "realtime" anywhere would misread the ratchet as intraday.
        head = re.split(r"[(;]|\bbreach\b", t)[0]
        hwm = (NONE if "static" in head else
               EOD if "eod" in head or "end_of_day" in head or "end-of-day" in head
               else INTRA if "intraday" in head or "realtime" in head else NONE)
    if breach is not None:
        breach = INTRA if "intraday" in str(breach) or "unreal" in str(breach) else breach
    else:
        t = str(row.get("dd_type", "")).lower()
        # only when the researcher explicitly said so in the dd_type prose
        breach = INTRA if ("realtime" in t and "unrealiz" in t) else None
    return hwm, breach


def _lock_offset(row, start: float, firm: str, phase: str):
    """Canonicalise dd_lock_at to an offset from the starting balance.

    Returns None when the floor never stops trailing, or when the source is
    too ambiguous to canonicalise (the caller then treats it as unverified).
    """
    raw = row.get("dd_lock_at")
    if raw is None:
        return None
    if isinstance(raw, dict):                    # Apex: vendor-dependent
        return "VENDOR_DEPENDENT"
    if isinstance(raw, (int, float)):
        return float(raw) - start if raw >= start else float(raw)
    s = str(raw).lower()
    if "never" in s or "indefinit" in s:
        return None
    m = re.search(r"start(?:ing)?[_ ]?(?:balance)?\s*\+\s*\$?([\d,]+)", s)
    if m:
        return float(m.group(1).replace(",", ""))
    if "starting_balance" in s or "starting balance" in s or s == "start":
        return 0.0
    # "locks permanently at 50100" / "lock@$53,000"
    nums = [float(x.replace(",", "")) for x in re.findall(r"[\d,]{4,}", s)]
    cands = [n for n in nums if start <= n <= start + 20_000]
    if cands:
        return min(cands) - start
    if s.strip() in ("0", "$0"):
        return 0.0 - start if start else 0.0
    return None


def _consistency_effect(row, phase: str):
    ce = row.get("consistency_effect")
    if isinstance(ce, dict):
        eff = str(ce.get("effect", "")).lower()
        if "raise" in eff or "inflat" in eff:
            return "inflates_target"
        if ce.get("blocks_payout"):
            return "blocks_payout"
        if ce.get("blocks_advancement") or ce.get("fails_account") is False:
            return "blocks_pass"
        return None
    if isinstance(ce, str):
        s = ce.lower()
        if "none" in s or "n/a" in s:
            return None
        for key, val in (("payout", "blocks_payout"), ("inflat", "inflates_target"),
                         ("raise", "inflates_target"), ("target", "inflates_target"),
                         ("pass", "blocks_pass")):
            if key in s:
                return val
    if _num(row.get("consistency_pct")) is None:
        return None
    # percentage present, mechanic unstated -> phase-based default is NOT safe
    return None


def _contracts(v):
    if isinstance(v, dict):
        return int(v["minis"]) if "minis" in v else None
    n = _num(v)
    return int(n) if n is not None else None


def _split(v):
    n = _num(v)
    if n is None:
        return None
    return n / 100.0 if n > 1 else n


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

# Per-firm facts the research verified firm-wide but that the per-row fields do
# not all carry. `breach` is applied ONLY where a row does not state its own --
# it records a sourced fact, it does not infer one. Each entry carries the
# verbatim quote it rests on so the claim stays auditable.
#
# Lucid is deliberately None: its documentation uses "balance" in every instance
# and never once says "equity" or "unrealized". Defaulting it to the majority
# would be exactly the fabrication this whole exercise exists to prevent.
FIRM_FACTS = {
    "my_funded_futures": dict(
        automation=None, breach=INTRA,
        quote="open equity losses are taken into consideration when calculating "
              "whether or not the account failed on this rule"),
    "topstep": dict(
        automation=None, breach=INTRA,
        quote="Risk limits are monitored in real-time using Net P&L - both realized "
              "and unrealized... Final balance above the limit doesn't matter. "
              "The breach happened first"),
    "take_profit_trader": dict(
        automation=None, breach=INTRA,
        quote="If your account balance drops to the Minimum Account Balance at any "
              "time - through realized or unrealized losses - the account is "
              "immediately liquidated"),
    "apex_trader_funding": dict(
        automation=False, breach=INTRA,
        quote="Even though the threshold is calculated at end-of-day, it is "
              "enforced in real time"),
    "lucid_trading": dict(
        automation=True, breach=None,
        quote="UNVERIFIED - Lucid's docs say 'balance' throughout and never "
              "'equity' or 'unrealized'; a help-centre search for 'unrealized' "
              "returns zero articles"),
}
PHASE_ALIASES = {
    "evaluation": "evaluation", "test": "evaluation",
    "sim_funded": "funded_sim", "funded_sim": "funded_sim", "pro": "funded_sim",
    "funded_simulated_pa": "funded_sim",
    "live_funded": "live", "live": "live", "pro_plus": "live",
}


def _iter_rows(firm_key, tree):
    """Yield (phase, variant, size_key, row) across the five source shapes."""
    for phase_raw, variants in tree.items():
        if phase_raw.startswith("_"):
            continue
        phase = PHASE_ALIASES.get(phase_raw)
        if phase is None or not isinstance(variants, dict):
            continue
        for variant, node in variants.items():
            if variant.startswith("_") or not isinstance(node, dict):
                continue
            if "sizes" in node:                          # Apex: _defaults + sizes
                base = node.get("_defaults", {})
                for size_key, row in node["sizes"].items():
                    merged = {**base, **row}
                    yield phase, variant, size_key, merged
            else:
                for size_key, row in node.items():
                    if size_key.startswith("_") or not isinstance(row, dict):
                        continue
                    if not any(k in row for k in ("max_dd", "dd_type", "hwm_basis")):
                        continue                          # nested legacy sub-tree
                    yield phase, variant, size_key, row


def build() -> list[dict]:
    out = []
    for path in sorted(RESEARCH.glob("*.json")):
        doc = json.loads(path.read_text())
        firm_key = next((k for k in doc if k in FIRM_FACTS), None)
        if firm_key is None:
            raise SystemExit(f"{path.name}: no known firm key in {list(doc)}")
        facts = FIRM_FACTS[firm_key]
        for phase, variant, size_key, row in _iter_rows(firm_key, doc[firm_key]):
            try:
                size = _size_to_int(size_key)
            except (TypeError, ValueError):
                continue
            # funded-sim and live accounts of several firms start at $0 balance
            start = 0.0 if phase in ("funded_sim", "live") and firm_key in (
                "topstep", "take_profit_trader", "my_funded_futures") else float(size)
            hwm, breach = _bases_from_dd_type(row)
            if breach is None:
                breach = facts["breach"]        # sourced firm-wide fact, see FIRM_FACTS
            lock = _lock_offset(row, start, firm_key, phase)
            unmodeled = row.get("phase_only_rules") or []
            if isinstance(unmodeled, str):
                unmodeled = [unmodeled]
            if lock == "VENDOR_DEPENDENT":
                unmodeled = list(unmodeled) + [
                    "dd_lock_at is vendor-dependent (differs by trading platform)"]
                lock = None
            dll_raw = row.get("daily_loss_limit")
            rs = RuleSet(
                firm=firm_key, variant=variant, phase=phase, size=size,
                start_balance=start,
                profit_target=_num(row.get("profit_target")),
                profit_target_basis=None,      # UNVERIFIED for all five firms
                max_dd=_num(row.get("max_dd")),
                hwm_basis=hwm, breach_basis=breach, dd_lock_offset=lock,
                daily_loss_limit=_num(dll_raw),
                dll_soft=bool(row.get("dll_soft_breach")
                              or "soft" in str(row.get("dll_basis", "")).lower()),
                consistency_pct=_num(row.get("consistency_pct")),
                consistency_effect=_consistency_effect(row, phase),
                min_days=(lambda n: int(n) if n is not None else None)(
                    _num(row.get("min_days"))),
                qualifying_day_min_profit=_num(row.get("qualifying_day_min_profit")),
                max_contracts=_contracts(row.get("max_contracts")),
                micro_ratio=_num(row.get("micro_ratio")),
                profit_split=_split(row.get("profit_split")),
                buffer_required=_num(row.get("buffer_required")),
                min_payout=_num(row.get("min_payout")),
                max_withdrawal=_num(row.get("max_withdrawal")),
                automation_allowed=facts["automation"],
                account_cost=_num(row.get("account_cost")),
                reset_cost=_num(row.get("reset_cost")),
                unmodeled_rules=list(unmodeled),
                verified=bool(row.get("verified", True)) and breach is not None,
                retrieved=str(row.get("retrieved", "2026-07-25")),
                source_url=(row.get("source_url") if isinstance(
                    row.get("source_url"), str) else None),
                breach_source=facts["quote"],
            )
            out.append(asdict(rs))
    out.sort(key=lambda r: (r["firm"], r["variant"], r["phase"], r["size"]))
    return out


# --------------------------------------------------------------------------
# load / select
# --------------------------------------------------------------------------

_CACHE = None

# Prop firms ship rule changes roughly quarterly. Take Profit Trader removed its
# daily loss limit firm-wide, and Topstep's marketing page still understates the
# 50K payout cap by 2.5x -- both caught during the 2026-07-25 research. A rule
# file that ages silently is a correctness bug, not a staleness annoyance, so
# the app refreshes it and shows the retrieval date next to every number.
RULES_URL = "https://raw.githubusercontent.com/jalv92/PropSim/main/prop_rules.json"
USER_DIR = Path.home() / ".prop-sim"
USER_RULES = USER_DIR / "prop_rules.json"


def _rules_path() -> Path:
    """Prefer a downloaded rule file over the bundled one, when it is newer."""
    try:
        if USER_RULES.exists() and USER_RULES.stat().st_mtime > CANON.stat().st_mtime:
            return USER_RULES
    except OSError:
        pass
    return CANON


def update_rules(url=RULES_URL, timeout=5.0) -> dict:
    """Refresh the rule file from the public repo. Never raises, never blocks.

    Startup must not depend on the network: an offline user still gets the
    bundled rules. A download that parses to fewer rows than we ship is
    rejected rather than installed -- a truncated or error-page response must
    never replace a working rule set.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode()
        rows = json.loads(raw)
        if not isinstance(rows, list) or len(rows) < 100:
            return dict(ok=False, reason=f"rejected: {len(rows) if isinstance(rows, list) else 'not a list'} rows")
        have = len(json.loads(CANON.read_text())) if CANON.exists() else 0
        if len(rows) < have * 0.8:
            return dict(ok=False, reason=f"rejected: {len(rows)} rows vs {have} bundled")
        USER_DIR.mkdir(parents=True, exist_ok=True)
        USER_RULES.write_text(raw)
        global _CACHE
        _CACHE = None
        return dict(ok=True, rows=len(rows), path=str(USER_RULES))
    except Exception as exc:                       # offline, DNS, 404, timeout...
        return dict(ok=False, reason=f"{type(exc).__name__}: {exc}")


def load() -> list[RuleSet]:
    global _CACHE
    if _CACHE is None:
        path = _rules_path()
        if not path.exists():
            raise SystemExit(f"{path} missing - run: python3 prop_rules.py --build")
        names = {f.name for f in fields(RuleSet)}
        _CACHE = [RuleSet(**{k: v for k, v in r.items() if k in names})
                  for r in json.loads(path.read_text())]
    return _CACHE


def rules_origin() -> dict:
    """Where the rules in use came from, for the UI to display."""
    p = _rules_path()
    import datetime as _dt
    return dict(path=str(p), downloaded=(p == USER_RULES),
                mtime=_dt.date.fromtimestamp(p.stat().st_mtime).isoformat()
                if p.exists() else None)


def select(firm, variant, phase, size) -> RuleSet:
    for rs in load():
        if (rs.firm == firm and rs.variant == variant
                and rs.phase == phase and rs.size == size):
            return rs
    raise KeyError(f"no rule set for {firm}/{variant}/{phase}/{size}")


def firms():
    return sorted({rs.firm for rs in load()})


def variants(firm, phase=None):
    return sorted({(rs.variant, rs.phase, rs.size) for rs in load()
                   if rs.firm == firm and (phase is None or rs.phase == phase)})


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def selfcheck():
    rows = load()
    assert len(rows) >= 140, f"expected ~157 rows, got {len(rows)}"

    # spot-check values independently verified in the rules matrix document
    mff = select("my_funded_futures", "rapid", "evaluation", 50_000)
    assert mff.profit_target == 3000 and mff.max_dd == 2000, mff
    assert mff.hwm_basis == EOD, mff.hwm_basis
    assert mff.dd_lock_offset == 100, mff.dd_lock_offset
    assert mff.min_days == 2 and mff.max_contracts == 5, mff

    ts = select("topstep", "standard_path", "evaluation", 50_000)
    assert ts.profit_target == 3000 and ts.max_dd == 2000, ts
    assert ts.hwm_basis == EOD and ts.breach_basis == INTRA, (ts.hwm_basis, ts.breach_basis)
    # locks at the STARTING balance, not start+target -- the correction the
    # Topstep researcher pushed back on
    assert ts.dd_lock_offset == 0, ts.dd_lock_offset
    assert ts.dd_floor_lock == 50_000, ts.dd_floor_lock

    tpt = select("take_profit_trader", "standard", "evaluation", 50_000)
    assert tpt.profit_target == 3000 and tpt.max_dd == 2000, tpt
    assert tpt.daily_loss_limit is None, "TPT removed the DLL firm-wide"
    assert tpt.consistency_effect == "inflates_target", tpt.consistency_effect

    # Apex bans automation -- must be surfaced as a warning, not buried
    apex = [r for r in rows if r.firm == "apex_trader_funding"][0]
    assert apex.automation_allowed is False
    assert any("PROHIBITS AUTOMATED" in w for w in apex.warnings())

    # Lucid breach_basis is UNVERIFIED and must NOT have been defaulted to the
    # majority value; it must also mark itself unverified
    luc = [r for r in rows if r.firm == "lucid_trading"]
    assert luc, "lucid rows missing"
    assert all(r.breach_basis is None for r in luc), \
        "Lucid breach_basis was defaulted -- the source never states it"
    assert all(not r.verified for r in luc)
    assert any("breach_basis UNVERIFIED" in w for w in luc[0].warnings())

    # every row that trails must say what it breaches on, or be flagged
    for r in rows:
        assert r.hwm_basis in (NONE, EOD, INTRA), (r.firm, r.hwm_basis)
        if r.breach_basis is None:
            assert not r.verified, f"{r.firm}/{r.variant}: unverified breach but verified=True"

    n_unver = sum(1 for r in rows if not r.verified)
    print(f"selfcheck OK: {len(rows)} rule sets, {len(firms())} firms, "
          f"{n_unver} flagged unverified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="regenerate prop_rules.json")
    ap.add_argument("--list", metavar="FIRM", help="list a firm's variants")
    ap.add_argument("--update", action="store_true", help="refresh rules from the repo")
    args = ap.parse_args()

    if args.update:
        r = update_rules()
        print(f"update: {'OK ' + str(r['rows']) + ' rows -> ' + r['path'] if r['ok'] else 'FAILED (' + r['reason'] + ') - keeping bundled rules'}")
        return

    if args.build:
        rows = build()
        CANON.write_text(json.dumps(rows, indent=1))
        print(f"wrote {CANON} ({len(rows)} rule sets)")
        global _CACHE
        _CACHE = None
    if args.list:
        match = [f for f in firms() if args.list.lower() in f]
        for f in match:
            print(f"\n=== {f} ===")
            for v, ph, sz in variants(f):
                r = select(f, v, ph, sz)
                print(f"  {v:<28} {ph:<12} {sz:>7,}  dd={r.max_dd} "
                      f"hwm={r.hwm_basis} breach={r.breach_basis or 'UNVERIFIED'} "
                      f"lock={r.dd_lock_offset}")
        return
    selfcheck()


if __name__ == "__main__":
    main()
