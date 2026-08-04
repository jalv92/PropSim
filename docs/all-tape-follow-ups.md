# ALL tape — what was left undone

Written when `feat/all-continuous-tape` merged. Everything here was found during
that work, judged real, and deliberately not fixed in that branch — either
because it predates the feature or because fixing it is its own piece of design.
Nothing here blocks using ALL.

## Predates this feature

### ORB is not timeframe-invariant, and its own selfcheck says it must be

`engine.py --selfcheck` fails on `ORB must be timeframe-invariant`. This is not a
test artifact: the strategy returns different P&L for the same data at different
bar sizes, on two of the five cached NQ contracts.

```
NQ 03-26   30s -4700   1m -4700   5m -4100   15m -2900     FAILS
NQ 12-25   30s  -320   1m  -320   5m  -320   15m  -325     FAILS
NQ 06-26   1665 at all four                                 ok
NQ 09-25   2940 at all four                                 ok
NQ 09-26   1360 at all four                                 ok
```

ORB trades an opening range, so its entries should not depend on bar size at all.
A $1,800 spread between running the same strategy at 30 seconds and at 15 minutes
is the tool giving different answers to the same question.

Proven unrelated to the ALL work: `git diff main...feat/all-continuous-tape --
engine.py` contains no ORB lines, and `main`'s own `engine.py` reproduces the
identical numbers on the same contract. It only became visible because caching
more contracts changed which one `tape.sample_contract()` returns — it used to
land on `NQ 06-26`, which happens to be one of the three that pass.

Note for whoever picks this up: it is **not** caused by data holes. That was the
first hypothesis and it is wrong — `NQ 06-26` has two large holes (9 and 47
sessions) and passes; the two that fail have none.

### `prop_rules.py` has a `selfcheck()` but never wired `--selfcheck`

The function passes when called directly. The CLI flag was never added, so it is
absent from any sweep of the module selfchecks. Untouched by this branch.

(`nttrades.py` had the same gap and now has both, added with the live-database
snapshot fix — see `_open_readonly`.)

## Needs its own design

### The audit gate reads only `ts` and `px`

`continuous.audit()` cannot see `vol` or `side`. A corruption in either produces a
plausible, wrong backtest with the gate reporting clean — and `SweepFollow` reads
the aggressor side while `VWAPRevert` weights by volume. The gate's promise is
that a mis-stitched tape refuses to write; a third of the columns are outside its
scope.

It also accepts equal timestamps (`np.diff(ts) >= 0`), so a duplicated session is
invisible to it. The structural partition assertion in `continuous.selfcheck()`
catches that one, but only when someone runs the selfcheck.

### A thin roll overlap is a hard lockout

If a user's downloads leave fewer than `MIN_OVERLAP_SECONDS` (200) common seconds
at a roll, the spread cannot be measured, the audit refuses, and ALL cannot be
built at all. There is no override and no fallback — for example, taking the
spread from the nearest day where the overlap is sufficient, or letting the user
enter it. On this machine the thinnest real roll had 1,567 seconds, so the margin
is 7.8x, not the comfortable number the original comment claimed.

### Nothing admits or rejects a contract by size

"Update ALL" force-parses every NQ contract folder under `db/tick`. A two-day
legacy folder gets pulled into the stitch and can trip the roll gate. A minimum
session count before a contract is admitted would fix it, but picking that floor
is a judgement nobody has made yet.

### The real traded price is computed and never shown

`engine.backtest()` maps `entry_price` and `exit_price` back from the
back-adjusted scale to what actually traded, so a run on ALL can be checked
against NinjaTrader fills. Verified correct — 40 trades matched exactly across a
985.25-point adjustment. But neither the dashboard, nor `report.py`, nor any CLI
displays those two fields. The mapping is right and currently invisible, so the
design's own success criterion — "the ledger shows prices that match
NinjaTrader" — is not met in the product.

### Thresholds are absolute NQ points

`ROLL_BAND`, `SESSION_GAP_LIMIT` and the intraday step floor are all in index
points calibrated on NQ. They are safe only because `continuous.ROOT` pins the
stitch to NQ. Adding a second instrument means scaling them by that instrument's
price level, not copying the constants.

## Small and inert

Kept for whoever is next in these files; none of them changes behaviour today.

- `continuous.front_months()` has an unreachable `if not pool: continue` — `days`
  is derived from `daily`, so `pool` is never empty.
- Its tie-break on equal volume resolves by dict insertion order. Deterministic in
  practice (insertion is calendar order and the monotone rule keeps the
  incumbent), but untested and unstated.
- `second_last()`'s last-of-run mask is correct but never distinguished from a
  first-of-run mask by any fixture — every synthetic second holds one tick.
- `tape.available_range()` and `continuous.fingerprint()` do not surface
  `truncated_hours()`. The ALL panel shows it; the per-contract API stays silent.
- `continuous.selfcheck()` went from 6.4 s to 20.9 s and loads the full 108M-tick
  `ts` array plus each contract's. Dev-only, but it is the same unbounded-load
  pattern that produced the branch's one Critical finding.
