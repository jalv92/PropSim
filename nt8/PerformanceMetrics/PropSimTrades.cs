// PropSim — capture a plain Strategy Analyzer backtest.
//
// A custom fitness only runs during an OPTIMISATION. A performance metric is
// calculated for every performance report, which is what makes a single
// backtest exportable at all. Enable it once in Tools > Options > General >
// "Performance metrics", then run a backtest as usual.
//
// What it cannot know: a performance metric never sees the strategy object, so
// the run's fill configuration is absent from the file it writes. PropSim marks
// such a run "configuration unknown" and refuses to certify its fills. For a
// certified run, select the PropSim fitness and optimise (a one-value range per
// parameter is a single backtest).
//
// The reported column is cumulative net profit, so the metric is honest as a
// metric rather than a fake column existing only to trigger a side effect.
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using NinjaTrader.NinjaScript.PropSim;
#endregion

namespace NinjaTrader.NinjaScript.PerformanceMetrics
{
	public class PropSimTrades : PerformanceMetric
	{
		// Per-instance, never static: optimisation iterations run in parallel and
		// a shared list would interleave two parameter sets into one file.
		private readonly List<Cbi.Trade> captured = new List<Cbi.Trade>();

		// Declared here, not inherited -- a performance metric owns its own Values
		// array, and the Display attribute is what names the report column.
		[Display(Name = "PropSim net profit", Order = 0)]
		public double[] Values { get; private set; }

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
				Name = "PropSim trade export";
			else if (State == State.Configure)
				Values = new double[5];
			else if (State == State.Active)
			{
				Array.Clear(Values, 0, Values.Length);
				captured.Clear();               // reset before every run
			}
			else if (State == State.Terminated)
				PropSimExportWriter.Write(captured, "nt8-performance-metric", TradesPerformance);
		}

		protected override void OnAddTrade(Cbi.Trade trade)
		{
			if (trade == null || Values == null) return;
			captured.Add(trade);
			Values[(int)Cbi.PerformanceUnit.Currency] += trade.ProfitCurrency;
			Values[(int)Cbi.PerformanceUnit.Percent] += trade.ProfitPercent;
			Values[(int)Cbi.PerformanceUnit.Pips] += trade.ProfitPips;
			Values[(int)Cbi.PerformanceUnit.Points] += trade.ProfitPoints;
			Values[(int)Cbi.PerformanceUnit.Ticks] += trade.ProfitTicks;
		}

		protected override void OnCopyTo(PerformanceMetricBase target)
		{
			PropSimTrades other = target as PropSimTrades;
			if (other != null && other.Values != null && Values != null)
				Array.Copy(Values, other.Values, Values.Length);
		}

		public override string Format(object value, Cbi.PerformanceUnit unit, string propertyName)
		{
			double[] v = value as double[];
			if (v != null && v.Length == 5)
				return v[(int)unit].ToString("N2", CultureInfo.CurrentCulture);
			return value == null ? "" : value.ToString();
		}
	}
}
