# PropSim add-on for NinjaTrader 8

Captures Strategy Analyzer runs of **your own `.cs` strategies** and writes them
where PropSim reads them: `%USERPROFILE%\.prop-sim\backtests\`.

NinjaTrader does not persist Strategy Analyzer results to its database, which is
why this exists. It writes files to your own user folder and makes no network
calls.

## Install — copy the files (works on every build)

1. Open the folder `Documents\NinjaTrader 8\bin\Custom\`.
2. Copy the three files there, keeping the folder names:

   | from this repo | to |
   |---|---|
   | `AddOns/PropSimExportWriter.cs` | `bin\Custom\AddOns\` |
   | `OptimizationFitnesses/PropSimFitness.cs` | `bin\Custom\OptimizationFitnesses\` |
   | `PerformanceMetrics/PropSimTrades.cs` | `bin\Custom\PerformanceMetrics\` |

3. In NinjaTrader: **New → NinjaScript Editor**, then press **F5** to compile.

## Install — the archive (one click, needs a matching build)

**Tools → Import → NinjaScript Add-On…** and pick
[`dist/PropSimExport.zip`](dist/PropSimExport.zip).

The archive declares NinjaTrader **8.1.1.0**, so it imports on any newer build.
If your NinjaTrader is *older* than that it will be refused as "made from a newer
version" — rebuild it for your own build number:

```bash
python3 nt8/build_zip.py --version 8.0.29.0
```

## Use — capture a run

**Any backtest, including a single one.** Enable the metric once in
**Tools → Options → General → Performance metrics** → tick **PropSim trade
export**. Every backtest you run from then on is captured.

**A run whose fills PropSim can certify.** In the Strategy Analyzer set
**Fitness** to **PropSim export (net profit)** and run an **Optimize** (a range
of one value per parameter is a single backtest). This path also captures every
iteration of a real sweep.

Why two: a fitness receives the strategy object, so the run's bar type, Tick
Replay setting, order-fill resolution, slippage and the iteration's parameter
values all get recorded. A performance metric never sees the strategy, so those
are absent and PropSim labels the run **configuration unknown** rather than
assuming its fills were good.

Selecting the PropSim fitness does not change your optimisation ranking: its
value is plain net profit, identical to `MaxNetProfit`.

Then open PropSim → **Imported** and score the run against a firm.

## Fidelity — the label matters more than the P&L

NinjaTrader cannot run **Tick Replay** and **High order-fill resolution** at the
same time; they are mutually exclusive. A strategy that needs both — anything
reading `OnMarketData` — cannot be backtested correctly in the Strategy Analyzer
in any configuration. PropSim reads the recorded configuration and says which
case you are in:

| label | what it means |
|---|---|
| **fills resolved tick by tick** | High fill resolution, or 1-tick bars. A stop and a target inside one bar were ordered correctly. |
| **fills guessed on the bar** | Standard fill resolution on time bars. Every intrabar stop is a guess, and the error is optimistic. |
| **configuration unknown** | Captured by the performance metric. Not certifiable. |

For order-flow strategies the correct route is not the Strategy Analyzer at all:
run them in **Market Replay / Playback**, where `OnMarketData` fires and fills
resolve tick by tick. Those trades land in `NinjaTrader.sqlite`, which PropSim
reads directly — no add-on needed.

## Accounting

Each file carries the per-trade `profit`, `commission` and `fee`, **and**
NinjaTrader's own totals for the run. PropSim reconciles the per-trade sum
against those totals to determine whether `ProfitCurrency` was quoted before or
after commission, instead of assuming. When the totals are missing it assumes
gross, subtracts the costs, and says so in the UI.

## Safety

Both hooks are wrapped so they cannot throw into the Strategy Analyzer: a failed
export must never break the run it is measuring. Neither keeps shared mutable
state, because optimisation iterations run in parallel across cores.

## Uninstall

Delete the three files and press F5 in the NinjaScript Editor. Untick the metric
in Tools → Options if you enabled it.
