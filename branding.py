#!/usr/bin/env python3
"""How each account is NAMED on screen, kept apart from what it DOES.

`prop_rules.py` owns the rules; this owns the labels. The slugs it maps from
are stable keys the rest of the code depends on, so nothing here renames
anything -- it only decides what a human sees.

The names are the firms' own, with their own capitalisation and marks, taken
from their plans pages or help centres. A firm that writes "INTRADAY TRAIL"
gets "Intraday Trail"; one that writes "Express Funded Account™" keeps the
trademark. Where a slug is NOT a product a trader can buy -- an add-on, a
retired plan, a risk-remediation placement -- the label says so, because a
dropdown that lists it beside the real plans is telling the user something
false.

Which axis carries the marketing name differs by firm, and that is the whole
reason this file is a table rather than a rename:

    My Funded Futures   the PLAN is the product     Rapid Plan, Pro Plan
    Lucid Trading       the PLAN is the product     LucidPro, LucidDaily
    Apex                the PLAN is the product     Intraday Trail, EOD Trail
    Topstep             the PHASE is the product    Trading Combine -> XFA -> LFA
    Take Profit Trader  the PHASE is the product    Test -> PRO -> PRO+

    python3 branding.py --selfcheck
"""
from __future__ import annotations

import argparse

# ---- firms ------------------------------------------------------------------
# `logo` is a file under assets/logos/. When it is missing the UI falls back to
# the monogram, so a missing asset degrades to something readable rather than a
# broken image.
FIRMS = {
    "my_funded_futures":   dict(display="My Funded Futures",   short="MFF",
                                colour="#2f6bff", mono="M", logo="mff.svg"),
    "apex_trader_funding": dict(display="Apex Trader Funding", short="Apex",
                                colour="#f0a020", mono="A", logo="apex.svg"),
    "lucid_trading":       dict(display="Lucid Trading",       short="Lucid",
                                colour="#22c55e", mono="L", logo="lucid.svg"),
    "topstep":             dict(display="Topstep",             short="Topstep",
                                colour="#e4483c", mono="T", logo="topstep.svg"),
    "take_profit_trader":  dict(display="Take Profit Trader",  short="TPT",
                                colour="#16a97a", mono="P", logo="tpt.svg"),
}

# ---- plans ------------------------------------------------------------------
# (firm, variant) -> (label, note). `note` is None for a plan a trader can buy
# today, and a plain-English reason otherwise -- it is rendered beside the name.
BUYABLE = None

PLANS = {
    # My Funded Futures: four plans on the checkout page. The table splits some
    # of them because the rules differ, so the extra rows say what they are.
    ("my_funded_futures", "rapid"):          ("Rapid Plan", BUYABLE),
    ("my_funded_futures", "pro"):            ("Pro Plan", BUYABLE),
    ("my_funded_futures", "builder_default"): ("Builder Plan", BUYABLE),
    ("my_funded_futures", "builder"):        ("Builder Plan", "live phase"),
    ("my_funded_futures", "builder_addon"):  ("Builder Plan", "with the add-on"),
    ("my_funded_futures", "pro_1day_addon"): ("Pro Plan", "with the 1-day add-on"),
    ("my_funded_futures", "flex_legacy"):    ("Flex Plan", "rules captured 2026-07-25, "
                                              "when the plan page returned 404"),

    # Apex writes these in caps on its own site: "INTRADAY TRAIL", "EOD TRAIL".
    ("apex_trader_funding", "intraday_trailing"): ("Intraday Trail", BUYABLE),
    ("apex_trader_funding", "eod_drawdown"):      ("EOD Trail", BUYABLE),
    ("apex_trader_funding", "legacy_full_trailing_RETIRED"):
        ("Legacy Account", "full trailing; no longer sold"),
    ("apex_trader_funding", "legacy_static_RETIRED"):
        ("Legacy Account", "static; no longer sold"),

    # Lucid sells four families. LucidDaily is four products: the buyer picks
    # the daily loss limit and the evaluation drawdown at checkout.
    ("lucid_trading", "lucidpro_eval"):    ("LucidPro", BUYABLE),
    ("lucid_trading", "lucidpro_funded"):  ("LucidPro", BUYABLE),
    ("lucid_trading", "lucidflex_eval"):   ("LucidFlex", BUYABLE),
    ("lucid_trading", "lucidflex_funded"): ("LucidFlex", BUYABLE),
    ("lucid_trading", "luciddirect_s2f"):  ("LucidDirect", BUYABLE),
    ("lucid_trading", "luciddaily_dllon_intraday"):
        ("LucidDaily", "daily loss limit ON, intraday eval drawdown"),
    ("lucid_trading", "luciddaily_dllon_eod"):
        ("LucidDaily", "daily loss limit ON, end-of-day eval drawdown"),
    ("lucid_trading", "luciddaily_dlloff_intraday"):
        ("LucidDaily", "no daily loss limit, intraday eval drawdown"),
    ("lucid_trading", "luciddaily_dlloff_eod"):
        ("LucidDaily", "no daily loss limit, end-of-day eval drawdown"),
    ("lucid_trading", "lucidlive_current"): ("Lucid Live", "reached from a funded account"),
    # Absent from the plans page because it cannot be bought, not because it is
    # gone: "Select traders may eventually earn LucidMaxx status."
    ("lucid_trading", "lucidmaxx_eval"): ("LucidMaxx", "earned status, not purchasable"),
    ("lucid_trading", "lucidmaxx_live"): ("LucidMaxx", "earned status, not purchasable"),
    ("lucid_trading", "lucidblack_eval_SUNSET"):
        ("LucidBlack", "legacy; no new sales"),
    ("lucid_trading", "lucidblack_funded_SUNSET"):
        ("LucidBlack", "legacy; no new sales"),

    # Topstep: the phase carries the product name, so the variant carries the
    # path or the programme instead.
    ("topstep", "standard_path"):          ("Standard Path", BUYABLE),
    ("topstep", "no_activation_fee_path"): ("No Activation Fee Path", BUYABLE),
    ("topstep", "xfa_standard"):           ("Standard Path", BUYABLE),
    ("topstep", "xfa_consistency"):        ("Consistency Path", BUYABLE),
    ("topstep", "lfa"):                    ("Live Funded Account®", BUYABLE),
    ("topstep", "pro_account"):
        ("Pro Account", "simulated stand-in for the LFA where live access is unavailable"),
    # UNVERIFIED NAMES: the four below are the modeller's slugs. Topstep's own
    # wording for its Labs products was not confirmed, so they are shown as
    # descriptions rather than dressed up as brand names.
    ("topstep", "labs_25k_static"):        ("Labs · 25K static", "product name unconfirmed"),
    ("topstep", "labs_25k_static_xfa"):    ("Labs · 25K static", "product name unconfirmed"),
    ("topstep", "labs_250k_freedom"):      ("Labs · 250K freedom", "product name unconfirmed"),
    ("topstep", "labs_250k_freedom_xfa"):  ("Labs · 250K freedom", "product name unconfirmed"),

    # Take Profit Trader sells ONE account, in sizes. The phases carry the
    # names, so `standard` needs no plan label of its own.
    ("take_profit_trader", "standard"): ("Account", BUYABLE),
    ("take_profit_trader", "development"):
        ("Development account", "a risk-remediation placement, not a plan you buy"),
}

# ---- phases -----------------------------------------------------------------
GENERIC_PHASES = {"evaluation": "Evaluation",
                  "funded_sim": "Funded (sim)",
                  "live": "Live"}

# Two firms name their phases, and those names are what the trader recognises.
PHASES = {
    ("topstep", "evaluation"): "Trading Combine®",
    ("topstep", "funded_sim"): "Express Funded Account™",
    ("topstep", "live"):       "Live Funded Account®",
    ("take_profit_trader", "evaluation"): "Test",
    ("take_profit_trader", "funded_sim"): "PRO",
    ("take_profit_trader", "live"):       "PRO+",
}


def firm(key: str) -> dict:
    """Everything the UI needs to draw a firm, or a readable fallback."""
    return FIRMS.get(key, dict(display=key, short=key, colour="#888",
                               mono=(key[:1].upper() or "?"), logo=None))


def plan(firm_key: str, variant: str) -> tuple[str, str | None]:
    """(label, note). Falls back to the slug rather than inventing a name."""
    return PLANS.get((firm_key, variant), (variant, "name not mapped"))


def phase(firm_key: str, phase_key: str) -> str:
    return PHASES.get((firm_key, phase_key)) or GENERIC_PHASES.get(phase_key, phase_key)


def describe(rs) -> dict:
    """Label one RuleSet for display. Takes anything with firm/variant/phase."""
    label, note = plan(rs.firm, rs.variant)
    f = firm(rs.firm)
    return dict(firm=f["display"], firm_short=f["short"], colour=f["colour"],
                mono=f["mono"], logo=f["logo"],
                plan=label, plan_note=note, phase=phase(rs.firm, rs.phase))


def selfcheck():
    import prop_rules as pr
    rows = pr.load()

    # THE ASSERT THAT EARNS THIS FILE. A variant added to the rules table
    # without a label here would silently show its slug -- `luciddaily_dllon_eod`
    # in a dropdown a trader is choosing an account from. This fails instead.
    missing = sorted({(r.firm, r.variant) for r in rows
                      if (r.firm, r.variant) not in PLANS})
    assert not missing, f"{len(missing)} variant(s) with no display name: {missing[:5]}"

    unknown_firms = sorted({r.firm for r in rows if r.firm not in FIRMS})
    assert not unknown_firms, f"firms with no branding: {unknown_firms}"
    for p in {r.phase for r in rows}:
        assert p in GENERIC_PHASES, f"phase with no generic label: {p}"

    # A label must never be emptier than the slug it replaces.
    for (fk, v), (label, _note) in PLANS.items():
        assert label and label.strip(), f"{fk}/{v}: empty label"

    # Anything a trader CANNOT buy has to say why. Silence here is the failure
    # mode: a retired plan or a punishment account listed beside the real ones,
    # with nothing to tell them apart.
    must_explain = [
        (fk, v) for (fk, v), (_l, note) in PLANS.items()
        if note is None and any(t in v for t in
                                ("RETIRED", "SUNSET", "legacy", "addon",
                                 "development", "maxx", "pro_account"))]
    assert not must_explain, f"unbuyable variants presented as plans: {must_explain}"

    # Topstep and TPT name their phases; everyone else takes the generic ones.
    assert phase("topstep", "funded_sim") == "Express Funded Account™"
    assert phase("take_profit_trader", "evaluation") == "Test"
    assert phase("lucid_trading", "evaluation") == "Evaluation"

    sample = pr.select("lucid_trading", "luciddaily_dllon_intraday", "evaluation", 50_000)
    d = describe(sample)
    assert d["firm"] == "Lucid Trading" and d["plan"] == "LucidDaily", d
    assert "intraday" in d["plan_note"], d

    noted = sum(1 for (_l, n) in PLANS.values() if n)
    print(f"selfcheck OK: {len(PLANS)} variants labelled across {len(FIRMS)} firms, "
          f"every variant in {len(rows)} rule sets has a name; {noted} carry a note "
          f"saying what they are; phases named by firm where the firm names them "
          f"(Topstep '{phase('topstep', 'funded_sim')}', TPT '{phase('take_profit_trader', 'live')}')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the whole mapping")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.list:
        for key, f in FIRMS.items():
            print(f"\n{f['display']}  ({f['short']}, {f['colour']})")
            for (fk, v), (label, note) in PLANS.items():
                if fk == key:
                    print(f"    {v:<30} {label}" + (f"  — {note}" if note else ""))


if __name__ == "__main__":
    main()
