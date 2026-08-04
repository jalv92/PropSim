# How PropSim works, tab by tab

Every number on screen comes from one of five places. Nothing is fetched from a
server, and nothing is invented:

| source | what it holds | who reads it |
|---|---|---|
| `<NinjaTrader 8>/db/NinjaTrader.sqlite` | your real executions (live, Sim101, Playback) | `nttrades.py` |
| `<NinjaTrader 8>/db/tick/<contract>/*.ncd` | the tick tape, with bid/ask at every trade | `ncd_parse.py` → `tape.py` |
| `<NinjaTrader 8>/db/replay/<contract>/*.nrd` | Market Replay days — **header only** | `ntdata.py` |
| `prop_rules.json` (bundled, auto-updated) | 148 account rule sets, each with a source URL and a date | `prop_rules.py` |
| `~/.prop-sim/` | your settings, the tick cache, imported runs, the trial ledger | `ntdata.py`, `ntimport.py`, `ledger.py` |

The rule table refreshes at startup from the public repo, in the background. If
you are offline, or the download is shorter than the bundled copy, the bundled
copy is kept and the UI says which one is in use.

---

## Backtesting

**Where it comes from:** `report.py`, composing `sim.replay` (the trade-by-trade
grid) and `sim.sim_eval` (the Monte Carlo) over one run from `engine.py`.
`report.build()` does not re-derive a single drawdown rule — it arranges numbers
that already exist elsewhere, because a second definition of a rule is how this
project got its one bug from having two (`sim._day_draw`), and a third would be
the one on screen.

Picking a firm/variant/phase/size selects a `RuleSet` the same way the Prop Firm
tab does; picking a strategy, contract, date range and timeframe runs it the same
way the Backtest tab does. `build()` returns one dict — `grid`, `prop`, `noise`,
`meta`, `has_trades` — and the page reads exactly that, twice: **Raw backtest**
renders `grid` (NinjaTrader's All/Long/Short table, plus PropSim's own
intra-trade fall rows), **Prop-firm reality** renders `prop` (what this run did
against the rules picked above — verdict, day, trade, deepest fall, and the
real / fiction split of the P&L — then the same Monte Carlo projection the Prop
Firm tab shows). Switching between the two is a local re-render; it costs no
request and no trial.

The split exists because **an evaluation ends on a pass as well as on a bust**:
in both cases the trades after that point never happened, and they are labelled
accordingly rather than credited to the run. An account still open has no such
boundary and is shown without a split. The deepest intra-trade fall is taken
over the trades that happened, and it is compared to the allowance **only when
the account died on the drawdown floor** — against a static floor with a cushion
the two numbers are unrelated, and the comparison would read as a breach on a
surviving account.

**Only PropSim's own strategies run here** — the strategy picker is
`/api/strategies`, the same list the Backtest tab offers, with no import source
alongside it. That restriction is what makes the drawdown verdict exact rather
than an upper bound: these are the only runs with a tick-measured intra-trade
path (`intra_mdd` on every trade), so a real-time trailing floor can be tested
against the actual worst equity moment inside a trade, not just its close.
Runs captured from NinjaTrader (the Imported tab) carry no such path —
NinjaTrader never records one. They are still read against the intraday rules on
their own tab, but with `mfe - mae` substituted for the missing fall
(`sim.trade_path`), which is its upper bound: those verdicts are pessimistic
rather than exact, and the tab says so.

One **Run** writes one trial to the ledger, same as the Backtest tab; an
identical re-run collapses to the same fingerprint and is not charged twice.

---

## Data

**Where it comes from:** `ntdata.inventory()` walks your NinjaTrader folder.

Two lists, and the distinction is the whole point:

- **Tape (`db/tick`) — usable.** Days, file count, size and an estimated tick
  count per contract. This is what backtests and bootstraps run on.
- **Market Replay (`db/replay`) — catalogued, not replayable.** Only the 44×80-byte
  header is read, which is checksummed, so days, session bounds and mean depth
  quality are reported. The event stream is *not* decoded: it carries an
  out-of-spec volume opcode that silently corrupts any parser written outside
  NinjaTrader. Reporting what you have beats a decoder that returns plausible
  garbage.

Point the box at the folder that contains `db/` and press **Rescan**. That is the
entire configuration of the program. The inventory is cached, because a tick
folder holds thousands of files and scanning one measured 10.4 s before the cache
existed; **Rescan** forces it.

---

## Prop Firm

**Where it comes from:** the rule set you pick out of `prop_rules.json`, plus a
trade source, run through the Monte Carlo in `sim.py`.

### The pickers

`firm → variant → phase → size` selects one **rule set**: profit target, drawdown,
what ratchets the floor, what breach is tested against, the lock point, daily loss
limit, consistency rule, contract cap, profit split, buffer, minimum payout,
account cost. 148 combinations across five firms.

### The trade source

| option | data | notes |
|---|---|---|
| an account name | round trips rebuilt from `NinjaTrader.sqlite` | exact commissions, real per-trade MAE, direction from `OrderAction` |
| `backtest — …` | the last run of the Backtest tab | held in memory only |
| `NinjaTrader — …` | a Strategy Analyzer run captured by the add-on | carries a fidelity label |
| parametric profile | a synthetic win-rate / reward-risk model in `sim.py` | for when you have no trades yet |

A real trade list replaces the synthetic model entirely. It is resampled as **whole
trading days** (day-block bootstrap), never as loose trades, because every rule
that matters — daily loss limits, consistency, minimum days, end-of-day
ratcheting — is a function of within-day structure.

An account matched to a firm preselects the **firm and phase only**. Size and
variant are not derivable from an account name, so you still pick them.

### What the simulation does

10,000 paths, each one day at a time (`sim_eval`):

1. draw a real trading day (or a Bernoulli day, parametrically);
2. size each trade with the chosen policy from the live buffer;
3. after every trade, test the drawdown floor on the basis the firm states —
   **including unrealized loss** where four of the five firms say so;
4. ratchet the high-water mark on the firm's basis (end-of-day close, or intraday
   equity including unrealized gains);
5. apply the daily loss limit, as a hard breach or a pause, per firm;
6. check the target, the minimum-days rule and the consistency rule.

Then the funded phase (`sim_funded`) runs **chained** to the evaluation: each path
carries its own pass day forward, so it gets what is left of the planning horizon
rather than a fresh one, and days-to-first-payout is counted from the day the
evaluation was bought. Paths that never passed spend no days there at all.

### What you see

- **Donut** — P(pass) / P(bust) / P(timeout), and mean days to pass.
- **KPI tiles** — net EV of one attempt (fee included), payout probability, mean
  payout, days to first payout, and the 5th/95th percentiles.
- **Spaghetti** — 1,000 recorded equity paths with the target and the drawdown
  floor drawn on, coloured by outcome, scrubable day by day.
- **Histogram** — the distribution of final P&L.
- **Funded toggle** — the same account after funding: what was withdrawn, and the
  three-way split of *still alive* / *paid then lost it* / *lost it with nothing
  withdrawn*. Conditional on having passed.
- **Warnings** — thin samples, few distinct daily outcomes, unverified rules, firms
  that prohibit automation, and the fidelity label of an imported run.

---

## Verdict

**Where it comes from:** the metadata of the selected rule set. No simulation.

Rules retrieved date, whether every field is verified, the ratchet basis, the
breach basis (or `UNVERIFIED`), the lock point, the consistency mechanic, and the
list of rules the simulator **does not model** at all. This is the tab that tells
you how much to trust the other tabs.

---

## Risk & Monte Carlo

**Where it comes from:** nothing. It is documentation.

It states what the parametric model is and where it is weakest — most importantly
that 10,000 draws of the same estimated win rate are not 10,000 pieces of
evidence, and that a winner's intra-trade excursion is *modelled* (half the stop
distance) when no real trades are supplied. Pick a real trade source and that
assumption disappears.

**Regimes is disabled on purpose.** Slicing results by market regime after the
fact is p-hacking with a chart attached.

---

## Backtest

**Where it comes from:** your own ticks. `db/tick/<contract>/*.ncd` → decoded by
`ncd_parse.py` → cached columnar by `tape.py` (17 bytes/tick, ~54× faster per
pass) → bars built on demand → strategy from `engine.LIBRARY` → fills resolved
tick by tick.

Three inputs define a run, and all three change the answer:

1. **Market data** — contract and date range. `Prepare this contract` builds the
   cache once (~15 s per contract).
2. **Strategy and parameters** — `ma_cross`, `orb`, `range_break`,
   `sweep_follow`, `fvg`, `vwap_revert`, `latigo_break`. All but the first two
   read the raw tape, including the true aggressor side of every print.
   `latigo_break` (the port of the NT8 `LatigoBreakStrategy.cs`) also sets
   `full_session`: its 18:00 and 20:00 ET windows are outside RTH, so `prepare`
   drops the RTH filter when it is the strategy being run. A strategy that needs
   the overnight tape and is handed an RTH one does not fail — it finds nothing,
   which reads exactly like "no edge", which is why the strategy declares it
   rather than the caller remembering.
3. **Timeframe** — a **Type** (Minute or Second) and a **Value**, exactly as
   NinjaTrader's Data Series dialog splits them; internally both collapse to one
   number of seconds, so a 30-second bar and a 2-minute bar are the same kind of
   input. Bars are rebuilt from the same ticks, so 15s, 1m and 5m runs are measured
   against *identical* market data. In NinjaTrader the timeframe is chosen when you
   apply a strategy, not in its code: it is a property of the run. Measured, same
   strategy, same ticks: −$10,575 at 1 minute and +$300 at 5.

### The fill model, which decides the answer

- Entries pay slippage; retrace setups fill at a **resting limit** at their level.
- A **stop is a market order**: it fills at the worse of the level and the price
  that breached it. Measured on this tape the gap-through is 0–3 ticks at the
  median, 3–8 at p90, with a tail to ~60 ticks for strategies that enter in the
  fastest moments.
- When a stop and a target fall in the same unresolved window, **the stop wins**.
- **Every position is flat at the session close.** The tape is RTH-filtered, so in
  the array 16:00 sits next to 09:30 the next morning; without this, the next
  session's opening gap "breaches" today's stop.
- **A hole in the tape is not a price move.** A silence longer than 120 s inside a
  session means hourly files are missing; the position is closed at the last real
  tick and the run reports the hole rather than booking an invented fill.

`measure from my tape` runs `slippage.py`: it reads the bid and ask carried at
every print, **regular trading hours only**, and reports a *floor* (half the
median spread — no execution beats it) and a *stress* value (the p90 half-spread
or the p90 tick-gap, whichever is worse). It fills in the floor and leaves the
stress value to you. Measuring all 24 hours instead would give a median spread of
4 ticks against 3 in session, and calibrating on the overnight book buries every
real edge.

`Score this against a prop firm →` hands the resulting trades to the Prop Firm tab.

---

## AI author

**Where it comes from:** the Anthropic Messages API, called with **your** key, by
`aiauthor.py`. It is the only tab that makes a network request to anything other
than the rule-file repo, and it makes none unless you press a button.

The key is read from `ANTHROPIC_API_KEY`, then `~/.prop-sim/anthropic-api-key.txt`,
then `anthropic_api_key` in `~/.prop-sim/config.json` — all outside the repository.
`/api/ai` reports *whether* a key was found and *where from*, never its value, and
every message that can reach the screen passes through `redact()` first, because an
SDK-level HTTP error can quote the request it failed on. `pip install anthropic` is
required for this tab and nothing else; without it the tab says so and the rest of
the program is unaffected.

**Write a strategy.** The system prompt embeds `plugins.TEMPLATE` verbatim — the
same file `plugins.py --install-template` writes — rather than paraphrasing it, so
there is one contract and not two that drift. What comes back goes through the
plugin validation (allowlist → subprocess load → in-process load → smoke test →
output contract) and then the lookahead probe. Three attempts; the validator's own
message is handed back between them, unedited.

**The lookahead probe** (`aiauthor.lookahead_probe`) is the check that is specific
to generated code. It re-runs `entries()` on a tape whose prices *after* a chosen
tick are shifted 100 ticks and whose aggressor side is flipped, then requires every
trade decided at or before that tick to come out identical — same existence, same
direction, same stop, same target. Bar boundaries come from timestamps only, so the
comparison is exact rather than approximate.

The cut is placed **one tick after a real entry**, at up to six entries spread
across the trade list. That detail is the whole test: an earlier version cut at
fixed fractions of the tape and *cleared* a strategy that reads its own entry bar's
high, because no trade happened to land on the straddling bar. Cutting at the trades
themselves makes the detection deterministic for every sampled trade, and the
rejection names the trade and both values.

It probes the last 15 sessions by default (widening once if that tail is too quiet),
because causality does not need six months of ticks to show itself and the probe
re-runs the strategy once per cut. What it cannot see: a peek that affects only
unsampled trades, and anything at all about profitability.

**Port to NinjaScript.** For strategies with no hand-written template. The model is
shown the Python source (kept on the class as `_source` at load time, since a plugin
class has no file to `inspect.getsource`), the parameter values and the provenance
header, then its C# goes through `nt8gen.generate(source=…)` — the same Roslyn
compiler, the same bounded retry, the same two-label verdict. Roslyn's own errors go
back to the model verbatim.

**Generation is free; evaluation is charged.** Nothing on this tab touches the trial
ledger, and the validation run's P&L and t-statistic are deliberately withheld from
the UI and the API response. Reporting them would turn generation into an unlimited
free search over your own data, which is exactly what the ledger exists to count.
Measured on the strategies written during development: one passed validation with 264
causally-clean trades and then scored **t = −3.11** the moment it was backtested on
the Backtest tab. The validation says the code is honest. It does not say the idea is.

---

## Imported

**Where it comes from:** `~/.prop-sim/backtests/*.json`, written by the NinjaScript
add-on in `nt8/`.

NinjaTrader does not persist Strategy Analyzer results anywhere, which is why the
add-on exists. Two hooks, because neither alone is enough:

- a custom **OptimizationFitness** — receives the strategy object, so it records
  the bar type, Tick Replay setting, order-fill resolution, slippage and that
  iteration's parameter values. Fires on every iteration of an optimisation.
- a custom **PerformanceMetric** — fires on a plain backtest, but never sees the
  strategy, so the configuration is absent.

`ntimport.py` reads them and does two things it refuses to guess at:

**Fidelity.** NinjaTrader cannot run Tick Replay and High order-fill resolution at
the same time; a strategy reading `OnMarketData` with an intrabar stop needs both.
So each run is labelled *fills resolved tick by tick*, *fills guessed on the bar*,
or *configuration unknown* — and the label follows the run onto the results page.

**Accounting.** The add-on also exports NinjaTrader's own totals for the run, and
the importer reconciles the per-trade sum against them to determine whether
`ProfitCurrency` was quoted before or after commission. At $5 a round trip that
decides the answer on any high-frequency strategy, and it is not worth settling by
belief. When the totals are missing it assumes gross, subtracts the costs, and
says so.

Runs captured twice (both hooks enabled) are deduplicated, keeping the labelled
copy.

---

## Trials

**Where it comes from:** `~/.prop-sim/trials.jsonl`, append-only and hash-chained.

Every run is recorded **before** its result is drawn — a ledger written afterwards
loses exactly the runs you abandon on first glance. Each record carries the digest
of the previous one, so a deleted or edited entry is detectable; the tab shows
whether the chain is intact.

- **Search trials** count: a backtest, an imported NinjaTrader run, a sweep.
- **Scoring runs** do not: re-pricing a trade list you already have against a
  firm's rules tells you about the rules, not about whether the edge is real.

Next to your best daily t-statistic it shows what pure noise reaches at that trial
count (the expected maximum of *n* standard normals, Blom's approximation) and the
Bonferroni-corrected bar for a 5% claim. At 20 trials those are 1.87 and 3.02; at
200, 2.74 and 3.66.

**Known gap:** runs launched from the command line (`python3 engine.py --strategy …`)
do not touch the ledger. Only the UI logs.

---

## The two self-check suites

Each assertion is a regression for a bug this project shipped, not a smoke test:

```bash
python3 sim.py --selfcheck       # gambler's ruin ≈ 0.400, drawdown bases, phase chaining
python3 engine.py --selfcheck    # lookahead, timeframe, slippage monotonicity, fills
python3 tape.py --selfcheck      # tick ordering, RTH filter, bar integrity
python3 ledger.py --selfcheck    # hash chain, trial counting, noise baseline
python3 ntimport.py --selfcheck  # add-on format, fidelity, accounting, dedup
python3 slippage.py --selfcheck  # spread measurement, session filter
python3 plugins.py --selfcheck   # allowlist, 9 refusals, output contract
python3 aiauthor.py --selfcheck  # prompt, redaction, lookahead probe (no API call)
python3 nt8gen.py --selfcheck    # templates render AND compile, retry loop
python3 prop_rules.py            # 148 rule sets, 5 firms, 38 flagged unverified
```

`aiauthor.py --selfcheck` makes **no API call and needs no key**. It pins the
redaction (three key-shaped strings in three error shapes, plus `/api/ai` and both
system prompts), the key parser against every way a key gets pasted, and — the one
that matters — the lookahead probe against two fixtures on real ticks: an honest
breakout it must clear, and a peeker that reads its own entry bar's high, which it
must catch. The peeker fixture is the exact bug that once scored a 79% win rate here.

What `engine.py --selfcheck` pins, across **every** strategy in the library:

- a stop-out can never be profitable, and a target exit can never lose;
- a stop's loss equals its excursion plus exit slippage and commission;
- a stop's loss is bounded — a 40-tick stop cannot lose 400 ticks;
- no trade spans two dates;
- ORB is timeframe-invariant (its opening range is a time window) while a
  bar-counting strategy is not;
- more slippage can never improve a result;
- a date range actually restricts the data.
