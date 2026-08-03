# Firm logos

Drop a file here named after the firm and the panel picks it up on the next
page load. The expected names come from `branding.py`:

    mff.svg       My Funded Futures
    apex.svg      Apex Trader Funding
    lucid.svg     Lucid Trading
    topstep.svg   Topstep
    tpt.svg       Take Profit Trader

`.png` works too — change the `logo` field in `branding.py` to match.

A missing file is not an error: the panel falls back to the firm's monogram on
its brand colour, which is why this directory can ship empty. Nothing is ever
fetched from the firms' servers; this page makes no external requests at all
(see `dashboard.py`'s module docstring — it is a data-licensing constraint, not
a preference).

These are third-party trademarks. They are used here to identify the firm whose
rules are being simulated, and they are not redistributed as part of this
repository.
