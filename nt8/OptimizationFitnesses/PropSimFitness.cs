// PropSim — capture a Strategy Analyzer run, including every iteration of an
// optimisation, and hand it to the PropSim desktop app.
//
// Select it in the Strategy Analyzer's "Fitness" box. It is the preferred hook
// because it receives the strategy object, so it records what the run was
// CONFIGURED like -- bar type, Tick Replay, order-fill resolution, slippage,
// and this iteration's parameter values. Without that configuration PropSim
// cannot tell an exact backtest from one whose stop fills were guessed, and
// mixing those two silently is the worst thing this product could do.
//
// It does not change your optimisation ranking: Value is plain net profit, the
// same number MaxNetProfit reports.
#region Using declarations
using NinjaTrader.NinjaScript.PropSim;
#endregion

namespace NinjaTrader.NinjaScript.OptimizationFitnesses
{
	public class PropSimFitness : OptimizationFitness
	{
		protected override void OnCalculatePerformanceValue(StrategyBase strategy)
		{
			PropSimExportWriter.Write(strategy, "nt8-optimization-fitness");

			// Net profit, so selecting this fitness ranks identically to
			// MaxNetProfit. Exporting must never quietly re-rank a sweep.
			if (strategy != null && strategy.SystemPerformance != null)
				Value = strategy.SystemPerformance.AllTrades.TradesPerformance.GrossProfit
				      + strategy.SystemPerformance.AllTrades.TradesPerformance.GrossLoss;
		}

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
				Name = "PropSim export (net profit)";
		}
	}
}
