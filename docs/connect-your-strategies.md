# Connecting your own NinjaScript strategies

PropSim's built-in strategies are for testing the engine. To evaluate the
strategies **you** actually run, there are three routes, and which one applies is
decided by where your strategy's entry signal comes from — not by preference.

## Pick the route in 30 seconds

| your strategy… | route |
|---|---|
| decides on **closed bars** only | **A** — Strategy Analyzer + PropSim fitness, fills certified |
| needs **tick data for its signal** (`OnMarketData`) *and* has an intrabar stop | **B** — Strategy Analyzer, fills guessed (labelled) |
| needs **market depth** (`OnMarketDepth`, Level 2) | **C** — Market Replay / Playback |
| opens positions through an **ATM template** | **C**, and see the ATM warning below |
| returns early on `State.Historical` | **C** — it is a no-op in the Strategy Analyzer by construction |

The constraint underneath all of it: **Tick Replay and High order-fill resolution
are mutually exclusive in the Strategy Analyzer**, and **market depth is never
delivered to a historical run at all**. A strategy needing tick-level signals *and*
intrabar fill ordering cannot be measured correctly there in any configuration.
That is a NinjaTrader limitation; PropSim's job is to say so rather than hide it.

---

## Route A — Strategy Analyzer, fills certified

1. Install the add-on (see [`../nt8/README.md`](../nt8/README.md)) and press **F5**
   in the NinjaScript Editor.
2. **Control Center → New → Strategy Analyzer**, pick strategy, instrument, dates.
3. **Backtest type → Optimize.** Not optional: `OnCalculatePerformanceValue` is
   only reachable on the Optimize path — a plain *Backtest* has no fitness field.
   Give every parameter a range of one value (min = max) for a single iteration.
4. **Optimize on → PropSim export (net profit)**. Its value is plain net profit,
   identical to `MaxNetProfit`, so your ranking is unchanged.
5. **Historical fill processing — all three fields:**
   - Order fill resolution = **High**
   - Order fill resolution type = **Tick**
   - Order fill resolution value = **1**

   Setting only *High* leaves the other two at their defaults (Minute / 1) and you
   get one-minute fill bars. PropSim labels that run **"fills guessed inside the
   fill bar"**, because on a liquid future a one-minute bar still straddles a
   typical stop-to-target span.
6. Tick Replay stays **off**. If your strategy needs it, you are on route B.
7. Make sure 1-tick history is downloaded for the whole window
   (**Tools → Historical Data Manager**), or the run silently degrades.
8. Run → one JSON lands in `%USERPROFILE%\.prop-sim\backtests\` → PropSim →
   **Imported**.

**You get:** the full trade list, every `[NinjaScriptProperty]` value, the bar and
fill configuration, and real per-trade MAE/MFE computed against the 1-tick fill
series.

## Route B — Strategy Analyzer, fills guessed

Identical to route A except:

- Order fill resolution stays **Standard** — you have no choice, because
- **Tick Replay must be ON**, set on the Data Series (right-click the series row).
  It is a UI setting; `Bars.IsTickReplay` is read-only and no strategy can turn it
  on for itself. Without it, an `OnMarketData`-derived value stays at zero and a
  gate built on it usually degrades to a no-op — the run then takes a *superset*
  of the intended trades while looking perfectly healthy.

PropSim labels the run **"fills guessed on the bar"** and carries that label onto
the results page. Treat its P&L as untrusted rather than pessimistic: NinjaTrader
traverses each bar Open→High→Low→Close or Open→Low→High→Close depending on where
the open sits, so the *direction* of the error changes with bar shape.

**Do not feed route-B MAE into a funding decision as if it were measured.** Four of
five firms breach on unrealised P&L, so the intraday-breach test is exactly where
a guessed excursion does the most damage.

## Route C — Market Replay / Playback → the database

No add-on involved. Your strategy runs in **real time** against replayed data, so
`OnMarketData` fires, depth is delivered, ATM works, and fills resolve in true
event order. The resulting trades persist to `NinjaTrader.sqlite`, which PropSim
reads directly.

1. **Tools → Historical Data Manager → Load → Market Replay** for the days you
   want. If your strategy needs depth, check the days actually contain it — the
   **Data** tab reports mean depth per contract from the `.nrd` header.
2. **Control Center → Connections → Playback**, pick the date, set the speed
   (1×–5× is safe; higher only guarantees order-event ordering, nothing else).
3. Open a chart with the strategy's expected primary series and enable it **from
   the chart**, not the Strategies tab.
4. Let the session run to the close. Restart the strategy between replay days if
   it keeps state that is not reset per session.
5. PropSim → the trade source picker lists it as
   `strategy <Name> — N trades, Playback, account …`.

PropSim attributes trades per strategy through NinjaTrader's own
`Strategy2Execution` → `Strategies.Classname` link, with the `IsReplay` flag — so
a strategy's trades stay separate from any discretionary trades on the same
account. Check what is already there with:

```bash
python3 nttrades.py --strategies
```

**You get:** real fills, real commissions, and **real** per-trade MAE/MFE from
NinjaTrader's own excursion tracking.

**You lose:** the parameter values (the database stores executions, not inputs —
record them yourself before the run), and any possibility of a sweep. Playback is
one manual pass at a time.

---

## The ATM trap

If your strategy opens positions through an ATM template, **all three routes
degrade at once**:

- ATM methods do not execute in historical Strategy Analyzer processing, so the
  run records nothing;
- even in real time, ATM executions are not attributed to the host strategy —
  they land in the database under a separate `AtmStrategy` row with no column
  linking back.

If your strategy has a native mode (its own stop/target instead of an ATM
template), use it for evaluation. `nttrades.py --strategies` deliberately hides
`AtmStrategy` rows, because NinjaTrader records one per template instance and they
are discretionary trades wearing a strategy's name.

## The sample-size problem

PropSim's Monte Carlo resamples **trading days**, not trades. So the unit that has
to be large is *sessions*, and an evaluation path is typically 10–30 trading days.
Bootstrapping day-blocks from fewer than ~60 distinct sessions makes the pass rate
a statement about the fortnight you happened to replay.

On route C that is the binding cost: one session is ~6.5 hours of tape, so ~60
sessions is on the order of 80 hours of wall clock at 5×, serial, with the
NinjaTrader GUI occupied. And Market Replay data has a **retention window** — days
you never downloaded are gone. Plan the corpus before you need it.

## Before you trust any of it

Run one throwaway session first and check three things:

1. **A file actually appeared** in `%USERPROFILE%\.prop-sim\backtests\` (routes
   A/B), or the strategy shows up in `nttrades.py --strategies` (route C).
2. **The net P&L matches NinjaTrader's own Trade Performance tab.** If they
   disagree, the reconstruction is wrong, not NinjaTrader.
3. **The fidelity label says what you expect.** A run you meant to be exact and
   which arrives labelled "fills guessed" means a setting did not take.
