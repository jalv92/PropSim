# PropSim

Answer two questions about your NinjaTrader 8 trading, using your own data and
the real published rules of five prop firms:

- **Would it pass the evaluation?**
- **Would it pay out once funded?**

Point it at your NinjaTrader folder. That is the entire configuration.

Everything runs on your machine. There is no account, no upload, and no network
call except an optional check for a newer rule file.

```
Prop Firm   Verdict   Risk & Monte Carlo   Backtest   Optimize   Imported   Trials   Data
```

---

## Install

### Requirements

- Windows (for the NinjaTrader data; the Python side also runs on Linux/WSL)
- **Python 3.9+** and **numpy** — that is the whole dependency list
- NinjaTrader 8 with tick data and/or trade history

### Run from source

```bash
git clone https://github.com/jalv92/PropSim.git
cd PropSim
pip install numpy

python3 dashboard.py --open
```

That opens `http://127.0.0.1:8765` in your browser. If PropSim cannot find your
NinjaTrader folder, open the **Data** tab and paste the path (the folder that
contains `db/`, normally `Documents\NinjaTrader 8`).

No binary release is published yet. To build the Windows installer yourself you
need [Inno Setup 6](https://jrsoftware.org/isdl.php) and PyInstaller:

```powershell
powershell -ExecutionPolicy Bypass -File build\build.ps1
```

That produces a per-user installer (~17 MB, installs ~55 MB into
`%LOCALAPPDATA%`, no admin prompt, no Python required on the target machine).

### Install the NinjaTrader add-on (optional, needed for the Imported tab)

Copy three files into `Documents\NinjaTrader 8\bin\Custom\` and press **F5** in
the NinjaScript Editor — full instructions and the one-click archive are in
**[`nt8/README.md`](nt8/README.md)**. It lets PropSim score backtests of your own
`.cs` strategies, which NinjaTrader otherwise never saves anywhere.

---

## Use

### 1. Data — point it at NinjaTrader once

The **Data** tab inventories what you have: the tick tape it can backtest on, and
your Market Replay days (catalogued, not replayable — see *Limits*). Press
**Rescan** after downloading new data.

### 2. Prop Firm — score a trade list against a firm

Pick a **firm**, **variant** and **account size**, choose where the trades come
from, and press **Run 10,000 sims**.

| trade source | what it is |
|---|---|
| your account name | your real trades, read straight from `NinjaTrader.sqlite` |
| `backtest — …` | the run you just did on the Backtest tab |
| `NinjaTrader — …` | a Strategy Analyzer run captured by the add-on |
| parametric profile | a synthetic win-rate/reward-risk model, when you have no trades yet |

You get P(pass), P(bust), days-to-pass, 1,000 simulated equity paths with the
target and drawdown floor drawn on, the distribution of outcomes, and — on the
**Funded** toggle — what the account actually pays out.

Any account matched to a firm preselects the **firm and phase only**. Size and
variant are not derivable from an account name, so you still pick those: putting
a guessed rule set behind a funding decision is the one thing never to do.

### 3. Backtest — run a strategy on your own ticks

Three choices define a run, and all three change the answer:

1. **Market data** — contract and date range. A contract needs preparing once
   (`Prepare this contract`), which parses its `.ncd` files into a columnar cache.
2. **Strategy and parameters** — five are built in: moving-average cross,
   opening-range breakout, aggressive-sweep follow, fair-value-gap retrace, VWAP
   reversion.
3. **Timeframe** — bars are built from ticks on demand, so 1m and 5m runs are
   measured against *identical* ticks. In NinjaTrader the timeframe is chosen when
   you apply a strategy, not in its code, which is why it is a property of the RUN
   here. Measured: the same MA cross on the same ticks gave −$10,575 at 1 minute
   and +$300 at 5.

Press **measure from my tape** to replace the slippage guess with the spread you
actually faced, measured from your own tick files during regular trading hours.
It fills in the *floor* (half the median spread — no execution beats it) and tells
you the *stress* value; run both.

Then **Score this against a prop firm →** hands the trades to the Prop Firm tab.

### 4. Your own strategies

Drop a Python file in `~/.prop-sim/strategies/` and it appears in the Backtest and
Optimize tabs, under **Yours**. Start from the documented template:

```bash
python3 plugins.py --install-template   # writes a starter file
python3 plugins.py                      # what is installed, and what is broken
python3 plugins.py --check my_idea.py   # validate one file without installing it
```

A strategy file **imports nothing** — the engine hands it numpy, the tape helpers
and the `Strategy`/`Param` base classes. It defines one class with an `entries()`
method that returns parallel arrays: the tick index to enter at, the direction, the
stop price and the target price.

Every file is checked before it is trusted: parsed against an allowlist (no
imports, no file access, no `eval`, no dunder attributes, no `np.load`), executed
first in a separate process so a hang cannot take the app with it, then
smoke-tested on real ticks with its output checked against the engine's contract —
matching array lengths, indices in range, and a stop on the correct side of the
entry. A rejected file is reported with the reason, never silently skipped.

**That is validation, not a sandbox.** A plugin runs with this program's
privileges. It catches the things that actually go wrong — a model that helpfully
adds `urllib`, four arrays of three different lengths, a long whose stop sits above
its entry. It will not stop someone who is trying. Put your own files there.

### 5. Optimize — sweep parameters, take the values to NinjaTrader

Give any parameter a range (`from:to:step`) and PropSim runs every combination
over your tick data — about 0.18 s each, so a 200-combination sweep takes half a
minute. Results are ranked by the cluster-robust **daily** t-statistic and each
row carries a verdict.

The winner comes with a NinjaScript block to paste into your own strategy's
`State.SetDefaults`, with the caveats inside the snippet: a parameter set copied
out of a results table arrives stripped of every qualification unless the
qualification travels with it.

**Every combination is a trial, and the ledger is told so.** The winner is judged
against the best t-statistic pure noise reaches over that many trials — the best
of 200 null searches reaches 2.74 by chance. A sweep buys speed and spends
credibility, and the tab shows the price before you press the button.

**P(pass) is never the objective.** It is a bounded, highly non-linear transform of
the edge, so an optimiser pointed at it finds lottery tickets rather than edge. The
search ranks on expectancy; the prop-firm Monte Carlo runs afterwards, on a result.

A configuration clears the bar at **t ≥ 1.5, ≥ 80 trades, and positive in at least
2 of 3 sub-periods** — and then it has earned *one forward test*, not a funded
account. Nothing on data you already own is validation.

### 6. Take it to NinjaTrader — a `.cs` proven to compile

The Optimize tab's **Generate the .cs and compile it** button writes a real
NinjaScript strategy with the winning values, and then **compiles it with the same
Roslyn setup the NinjaScript editor uses** (via [`nt8c`](https://github.com/jalv92))
before handing it over. A file that will not compile never reaches you.

```bash
python3 nt8gen.py --list
python3 nt8gen.py --strategy orb --params range_min=15,stop_ticks=30,rr=1,one_per_day=1
```

Templates exist today for `orb` and `ma_cross`. A strategy without one is refused
with that reason rather than half-generated.

**Two verdicts, never merged.** The compiler proves the file *compiles*. Nothing
proves it *behaves* like the Python it came from — bar construction, session
anchoring and fill resolution all differ between the two engines. So the output
carries a compile result and a **fidelity checklist** separately, and the checklist
is a list of things for you to verify, not a claim. The provenance and the caveats
are written **inside the `.cs` file**, because a generated strategy outlives the
session that produced it.

The last step closes the loop: run the generated strategy in the Strategy Analyzer
with the PropSim fitness, import it on the **Imported** tab, and compare the two
trade lists. That is what turns "it compiles" into "it behaves the same".

### 7. Imported — backtests of your own NinjaTrader strategies

Runs captured by the add-on, each labelled with whether its fills can be believed:

| label | meaning |
|---|---|
| **fills resolved tick by tick** | a stop and a target inside one bar were ordered correctly |
| **fills guessed on the bar** | NinjaTrader picked an order rather than measuring it — optimistic |
| **configuration unknown** | captured without the strategy's settings; not certifiable |

`score →` scores the run, carrying that label onto the results page.

### 8. Trials — how many times you have looked

Every run is recorded in an append-only, hash-chained ledger **before** its
result is drawn, and this tab shows the count next to your best t-statistic
together with the value pure noise would reach after that many trials.

A backtest is a hypothesis test. The best of forty is not the same claim as the
best of one — after 40 searches, noise alone reaches t ≈ 2.2, and a 5% claim
needs 3.2. Searching counts (backtests, imported runs, sweeps); scoring the same
trade list again does not.

### 9. Verdict — should you believe the numbers

Where each rule came from, when it was read, what is **unverified**, and which
rules the simulator does not model at all.

**A tab-by-tab explanation of what each screen does and where its data comes
from is in [`docs/how-it-works.md`](docs/how-it-works.md).**

---

## What it does that other calculators do not

**Reads your real trades.** NinjaTrader keeps executions in
`db/NinjaTrader.sqlite`; PropSim reads them directly, so you never export
anything. It reconstructs round trips with exact commissions, the correct
long/short direction, and the real per-trade maximum adverse excursion that
NinjaTrader records — which is what makes the intraday drawdown test exact rather
than approximated.

**Knows the rules.** 148 account variants across My Funded Futures, Apex,
Topstep, Take Profit Trader and Lucid Trading. Every value carries the URL it came
from and the date it was read. Rules are **data**, not code: adding a firm is a
data entry, not a patch.

**Simulates honestly.** 10,000 Monte Carlo paths bootstrapped from whole *trading
days* of your real trades — not from loose trades, because every rule that matters
(daily loss limits, consistency, minimum days, end-of-day ratcheting) depends on
within-day structure.

**Separates the two axes of drawdown.** Conflating them silently inflates every
pass rate:

| | |
|---|---|
| **what ratchets the floor** | nothing (static) / end-of-day close / intraday equity |
| **what is tested against it** | realized balance / equity including unrealized P&L |

Four of the five firms state that breach is evaluated in **real time on equity
including unrealized P&L**, while the floor only ratchets on the end-of-day close.
A simulator that checks the floor at the close lets every intraday dip that
recovers by 16:00 survive. Those accounts are dead in reality. The fifth firm
never says, so PropSim marks it `UNVERIFIED` and warns instead of guessing.

**Chains the two phases.** A path that needed 60 trading days to pass its
evaluation gets the *remaining* horizon in the funded phase, not a fresh one, and
days-to-first-payout is measured from the day the evaluation was bought.

Also surfaced rather than buried: rules the simulator does **not** model, firms
that **prohibit automated strategies** (selecting one shows a hard warning), and a
notice when your trade sample is too thin for the numbers to mean much.

---

## Limits, stated plainly

- **Market Replay is catalogued, not replayable.** The `.nrd` event stream
  carries an out-of-spec volume opcode that silently corrupts any decoder written
  outside NinjaTrader. PropSim reads only the header — which is checksummed — to
  report what you have, and uses the tick tape instead.
- **Half of an order-flow portfolio cannot be backtested in NinjaTrader at all.**
  Tick Replay and High order-fill resolution are mutually exclusive there, and a
  strategy reading `OnMarketData` with an intrabar stop needs both. Run those in
  **Market Replay / Playback** instead: those trades persist to the database, and
  PropSim reads them.
- **No queue position.** Slippage is calibrated from the quoted spread and the
  tick-gap distribution in your own data. Real queue modelling needs L2, which is
  the stream that cannot be decoded.
- **The Bernoulli profiles are a model, not evidence.** With no real trade list, a
  winner's worst excursion is assumed to be half its stop distance. Feed real
  trades and that assumption disappears.
- **The rules can go stale.** Firms ship changes roughly quarterly. Verify with
  the firm before paying for an evaluation.

---

## Development

```bash
python3 prop_rules.py             # self-check the rule table (148 sets, 5 firms)
python3 prop_rules.py --update    # refresh rules from this repo
python3 sim.py --selfcheck        # gambler's ruin, drawdown + chaining regressions
python3 engine.py --selfcheck     # lookahead, timeframe, slippage, fill invariants
python3 tape.py --selfcheck       # tick ordering, RTH filter, bar integrity
python3 ledger.py --selfcheck     # hash chain, trial counting, noise baseline
python3 ntimport.py --selfcheck   # add-on format round-trip, fidelity, dedup
python3 slippage.py --selfcheck   # spread measurement and the session filter
python3 plugins.py --selfcheck    # the allowlist, the refusals, the output contract
python3 nt8gen.py --selfcheck     # templates render, compile, and the retry loop works
python3 optimize.py --selfcheck   # grid, sub-periods, the gate, the noise ceiling
python3 optimize.py --contract "NQ 09-26" --strategy orb --range stop_ticks=20:60:10
python3 nttrades.py --list        # your accounts and their detected firm
python3 nttrades.py --strategies  # per NinjaScript strategy, from the database
python3 ledger.py                 # your trial log
```

Each self-check is a regression for a bug this project actually shipped, not a
smoke test. `engine.py --selfcheck` asserts that a stop-out can never be
profitable — it once booked +$17,635 because a strategy measured its stop from a
signal level while the engine filled at a later tick.

| file | what it owns |
|---|---|
| `prop_rules.py/.json` | the rule table, sourced and dated; `research/` holds provenance per firm |
| `sim.py` | Monte Carlo for the evaluation and funded phases |
| `nttrades.py` | reading your real trades out of `NinjaTrader.sqlite` |
| `ntdata.py` | finding and inventorying the NinjaTrader folder |
| `ncd_parse.py`, `tape.py` | decoding `.ncd` ticks and the columnar cache |
| `engine.py` | the backtest engine and strategy library |
| `slippage.py` | spread and tick-gap measurement |
| `ntimport.py` | importing runs captured by the NinjaScript add-on |
| `optimize.py` | parameter sweeps and the pre-registered gate |
| `plugins.py` | loading, validating and smoke-testing your own strategy files |
| `nt8gen.py`, `nt8gen/` | NinjaScript generation and the compile-verify loop |
| `ledger.py` | the append-only trial ledger |
| `dashboard.py/.html` | the local server and the whole UI |
| `nt8/` | the NinjaScript add-on and its importable archive |

---

## Corrections welcome

The rule file is the most valuable thing here and the most likely to drift. If a
value is wrong or out of date, open an issue with the firm's own page as the
source — every entry records where it came from, so a correction is easy to
verify. Firms contradicting themselves is normal: during research one had removed
a daily loss limit while its pricing page still displayed it, and another's
marketing page understated a payout cap by 2.5×.

## Not financial advice

A simulated pass rate is not a prediction about your account.

MIT licensed.
