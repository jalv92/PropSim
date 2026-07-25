# PropSim

Answer two questions about your NinjaTrader 8 trading, using your own trades and
the real published rules of five prop firms:

- **Would it pass the evaluation?**
- **Would it pay out once funded?**

Point it at your NinjaTrader folder. That is the entire configuration.

![tabs: Prop Firm / Verdict / Regimes / Risk & Monte Carlo / Data](docs/screenshot.png)

## Install

Download `PropSim-x.y.z-setup.exe` from Releases and run it. Per-user install,
no admin prompt, no Python required. Start Menu → PropSim opens a page in your
browser.

Everything runs on your machine. There is no account, no upload, and no network
call except an optional check for a newer rule file.

## What it does

**Reads your real trades.** NinjaTrader keeps executions in
`db/NinjaTrader.sqlite`; PropSim reads them directly, so you never export
anything. It reconstructs round trips with exact commissions, the correct
long/short direction, and the real per-trade maximum adverse excursion that
NinjaTrader records — which is what makes the intraday drawdown test exact
rather than approximated.

**Knows the rules.** 148 account variants across My Funded Futures, Apex,
Topstep, Take Profit Trader and Lucid Trading. Every value carries the URL it
came from and the date it was read. Rules are **data**, not code: adding a firm
is a data entry, not a patch.

**Simulates honestly.** 10,000 Monte Carlo paths bootstrapped from whole
*trading days* of your real trades — not from loose trades, because every rule
that matters (daily loss limits, consistency, minimum days, end-of-day
ratcheting) depends on within-day structure.

## What it tells you that other calculators do not

Drawdown has **two independent axes** and conflating them silently inflates
every pass rate:

| | |
|---|---|
| **what ratchets the floor** | nothing (static) / end-of-day close / intraday equity |
| **what is tested against it** | realized balance / equity including unrealized P&L |

Four of the five firms state that breach is evaluated in **real time on equity
including unrealized P&L**, while the floor only ratchets on the end-of-day
close. A simulator that checks the floor at the close lets every intraday dip
that recovers by 16:00 survive. Those accounts are dead in reality.

The fifth firm never says, so PropSim marks it `UNVERIFIED` and warns instead of
guessing.

Also surfaced rather than buried: rules the simulator does **not** model, firms
that **prohibit automated strategies** (selecting one shows a hard warning),
and a notice when your trade sample is too thin for the numbers to mean much.

## Limits, stated plainly

- **Market Replay is catalogued, not replayable.** The `.nrd` event stream
  carries an out-of-spec volume opcode that silently corrupts any decoder
  written outside NinjaTrader. PropSim reads only the header — which is
  checksummed — to report what you have, and uses the tick tape instead.
- **Strategy Analyzer results are not in the database.** NinjaTrader does not
  persist them. Backtests of bar-based strategies are captured by the companion
  NinjaScript add-on (in progress); order-flow strategies are evaluated by
  running them in Market Replay, where the resulting trades do persist.
- **Evaluation and funded phases are drawn independently.** A path that barely
  scraped through its evaluation starts the funded phase from the nominal
  balance. This matters most exactly at the pass boundary.
- **The rules can go stale.** Firms ship changes roughly quarterly. Verify with
  the firm before paying for an evaluation.

## Run from source

```bash
python prop_rules.py            # self-check the rule table
python prop_rules.py --update   # refresh rules from this repo
python sim.py --selfcheck       # gambler's-ruin anchor + drawdown regressions
python nttrades.py --list       # your accounts and their detected firm
python dashboard.py --open
```

Build the installer on Windows (needs Inno Setup 6):

```powershell
powershell -ExecutionPolicy Bypass -File build\build.ps1
```

## Corrections welcome

The rule file is the most valuable thing here and the most likely to drift. If a
value is wrong or out of date, open an issue with the firm's own page as the
source — every entry records where it came from so a correction is easy to
verify. Firms contradicting themselves is normal; during research one had
removed a daily loss limit while its pricing page still displayed it, and
another's marketing page understated a payout cap by 2.5×.

## Not financial advice

A simulated pass rate is not a prediction about your account.

MIT licensed.
