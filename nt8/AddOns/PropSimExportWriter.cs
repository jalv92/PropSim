// PropSim — trade-list writer shared by the two capture hooks.
//
// Writes one JSON file per captured run to %USERPROFILE%\.prop-sim\backtests\.
// The desktop app reads that folder; nothing is sent anywhere.
//
// TWO RULES THIS FILE OBEYS, because it runs inside the user's NinjaTrader:
//
//   1. It never throws. An exception raised in OnCalculatePerformanceValue or a
//      performance metric surfaces as a failed Strategy Analyzer run, and a tool
//      that breaks the thing it is measuring is worse than no tool. Every public
//      entry point is wrapped and swallows.
//
//   2. It keeps no shared mutable state. Strategy Analyzer optimisations run
//      iterations in PARALLEL across cores, so a static buffer would interleave
//      trades from different parameter sets into one file. Each call writes its
//      own uniquely-named file and touches nothing else.
//
// JSON is hand-built: NinjaScript targets .NET Framework 4.8 without a bundled
// serialiser, and a StringBuilder is smaller than taking a dependency.
#region Using declarations
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.PropSim
{
	public static class PropSimExportWriter
	{
		// Bumped only on a breaking change. Three producers (this file, the
		// SQLite reader, the tape engine) and one consumer is exactly where a
		// format drifts silently, so the consumer refuses an unknown version.
		public const string Schema = "propsim.trades/1";

		public static string Folder
		{
			get
			{
				string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
				return Path.Combine(home, ".prop-sim", "backtests");
			}
		}

		/// <summary>Capture a finished run straight off the strategy. Full fidelity
		/// metadata is available here, which is why this is the preferred hook.</summary>
		public static string Write(StrategyBase strategy, string source)
		{
			try
			{
				if (strategy == null) return null;
				TradeCollection trades = null;
				if (strategy.SystemPerformance != null)
					trades = strategy.SystemPerformance.AllTrades;
				if (trades == null || trades.Count == 0) return null;

				StringBuilder sb = new StringBuilder(1024 + trades.Count * 160);
				sb.Append("{\n");
				Field(sb, "schema", Schema);
				Field(sb, "source", source);
				Field(sb, "written", DateTime.Now.ToString("s", CultureInfo.InvariantCulture));
				Field(sb, "strategy", Safe(() => strategy.Name));
				Field(sb, "instrument", Safe(() => strategy.Instrument.FullName));
				Num(sb, "point_value", Safe(() => strategy.Instrument.MasterInstrument.PointValue, 0.0));
				Num(sb, "tick_size", Safe(() => strategy.Instrument.MasterInstrument.TickSize, 0.0));

				// The configuration that decides whether the fills can be trusted.
				// Tick Replay and High order-fill resolution are mutually exclusive
				// in NinjaTrader, so a strategy needing both cannot be measured
				// correctly here at all -- the app says so instead of averaging the
				// two into a number that looks fine.
				Field(sb, "bar_type", Safe(() => strategy.BarsPeriod.BarsPeriodType.ToString()));
				Num(sb, "bar_value", Safe(() => (double)strategy.BarsPeriod.Value, 0.0));
				Bool(sb, "tick_replay", Safe(() => strategy.IsTickReplay, false));
				Field(sb, "fill_resolution", Safe(() => strategy.OrderFillResolution.ToString()));
				Field(sb, "fill_resolution_type", Safe(() => strategy.OrderFillResolutionType.ToString()));
				Num(sb, "fill_resolution_value", Safe(() => (double)strategy.OrderFillResolutionValue, 0.0));
				Num(sb, "slippage_ticks", Safe(() => strategy.Slippage, 0.0));
				Field(sb, "calculate", Safe(() => strategy.Calculate.ToString()));

				sb.Append("  \"params\": {");
				sb.Append(Parameters(strategy));
				sb.Append("},\n");

				Totals(sb, Safe(() => trades.TradesPerformance, null));
				Trades(sb, trades);
				sb.Append("}\n");
				return Save(sb.ToString(), Safe(() => strategy.Name) ?? "strategy");
			}
			catch (Exception) { return null; }
		}

		/// <summary>Capture from a trade list alone, for the performance-metric hook
		/// that fires on plain backtests but never sees the strategy object.</summary>
		public static string Write(IList<Trade> trades, string source, TradesPerformance perf)
		{
			try
			{
				if (trades == null || trades.Count == 0) return null;
				StringBuilder sb = new StringBuilder(512 + trades.Count * 160);
				sb.Append("{\n");
				Field(sb, "schema", Schema);
				Field(sb, "source", source);
				Field(sb, "written", DateTime.Now.ToString("s", CultureInfo.InvariantCulture));
				// NOT the strategy name. A performance metric never sees the
				// strategy object, and Execution.Name is the SIGNAL name -- "Long1",
				// "PB Long". Writing that into the strategy field labelled every
				// metric-captured run with the name of its first entry signal, which
				// reads like a strategy name and is not one. Null is honest; the
				// signal is reported separately.
				Field(sb, "strategy", null);
				Field(sb, "entry_signal", Safe(() => trades[0].Entry.Name));
				Field(sb, "instrument", Safe(() => trades[0].Entry.Instrument.FullName));
				Num(sb, "point_value", Safe(() => trades[0].Entry.Instrument.MasterInstrument.PointValue, 0.0));
				Num(sb, "tick_size", Safe(() => trades[0].Entry.Instrument.MasterInstrument.TickSize, 0.0));
				// No strategy object here, so the run's fill configuration is
				// unknown -- recorded as unknown rather than guessed, and the app
				// refuses to certify such a run's fills.
				sb.Append("  \"params\": {},\n");
				Totals(sb, perf);
				Trades(sb, trades);
				sb.Append("}\n");
				return Save(sb.ToString(), Safe(() => trades[0].Entry.Name) ?? "strategy");
			}
			catch (Exception) { return null; }
		}

		// ------------------------------------------------------------------

		/// <summary>NinjaTrader's own totals for the run.
		///
		/// These exist so the consumer never has to GUESS whether
		/// Trade.ProfitCurrency is quoted before or after commission -- it
		/// reconciles the per-trade sum against these and reports which basis it
		/// found. An accounting assumption worth $5 a round trip is not something
		/// to settle by reading a forum post.</summary>
		private static void Totals(StringBuilder sb, TradesPerformance perf)
		{
			if (perf == null)
			{
				sb.Append("  \"perf\": null,\n");
				return;
			}
			sb.Append("  \"perf\": {");
			sb.Append("\"gross_profit\":").Append(D(Safe(() => perf.GrossProfit, 0.0))).Append(",");
			sb.Append("\"gross_loss\":").Append(D(Safe(() => perf.GrossLoss, 0.0))).Append(",");
			sb.Append("\"cum_profit\":").Append(D(Safe(() => perf.Currency.CumProfit, 0.0))).Append(",");
			sb.Append("\"trades\":").Append(Safe(() => perf.TradesCount, 0));
			sb.Append("},\n");
		}

		private static void Trades(StringBuilder sb, IList<Trade> trades)
		{
			sb.Append("  \"trades\": [\n");
			for (int i = 0; i < trades.Count; i++)
			{
				Trade t = trades[i];
				if (t == null || t.Entry == null || t.Exit == null) continue;
				int dir = t.Entry.MarketPosition == MarketPosition.Long ? 1 : -1;
				sb.Append("    {");
				sb.Append("\"entry\":\"").Append(t.Entry.Time.ToString("s", CultureInfo.InvariantCulture)).Append("\",");
				sb.Append("\"exit\":\"").Append(t.Exit.Time.ToString("s", CultureInfo.InvariantCulture)).Append("\",");
				sb.Append("\"dir\":").Append(dir).Append(",");
				sb.Append("\"qty\":").Append(t.Quantity).Append(",");
				sb.Append("\"entry_price\":").Append(D(t.Entry.Price)).Append(",");
				sb.Append("\"exit_price\":").Append(D(t.Exit.Price)).Append(",");
				// ProfitCurrency is already net of nothing -- commission is
				// reported separately, exactly as NinjaTrader models it, and the
				// consumer subtracts it once.
				sb.Append("\"profit\":").Append(D(t.ProfitCurrency)).Append(",");
				sb.Append("\"commission\":").Append(D(t.Commission)).Append(",");
				sb.Append("\"fee\":").Append(D(t.Fee)).Append(",");
				sb.Append("\"mae\":").Append(D(t.MaeCurrency)).Append(",");
				sb.Append("\"mfe\":").Append(D(t.MfeCurrency)).Append(",");
				sb.Append("\"signal\":\"").Append(Esc(Safe(() => t.Entry.Name))).Append("\"}");
				sb.Append(i == trades.Count - 1 ? "\n" : ",\n");
			}
			sb.Append("  ]\n");
		}

		private static string Parameters(StrategyBase strategy)
		{
			// The parameters of THIS iteration, so a sweep's rows are
			// distinguishable. Only properties the author marked as NinjaScript
			// parameters are read; nothing else on the strategy is touched.
			List<string> parts = new List<string>();
			try
			{
				PropertyInfo[] props = strategy.GetType().GetProperties(
					BindingFlags.Public | BindingFlags.Instance);
				foreach (PropertyInfo p in props)
				{
					if (p.GetCustomAttribute<NinjaScriptPropertyAttribute>() == null) continue;
					if (!p.CanRead) continue;
					object v;
					try { v = p.GetValue(strategy, null); }
					catch (Exception) { continue; }
					if (v == null) continue;
					if (v is bool) parts.Add("\"" + Esc(p.Name) + "\":" + ((bool)v ? "true" : "false"));
					else if (v is int || v is long || v is double || v is float || v is decimal)
						parts.Add("\"" + Esc(p.Name) + "\":" + D(Convert.ToDouble(v, CultureInfo.InvariantCulture)));
					else parts.Add("\"" + Esc(p.Name) + "\":\"" + Esc(v.ToString()) + "\"");
				}
			}
			catch (Exception) { }
			return string.Join(",", parts.ToArray());
		}

		private static string Save(string json, string strategyName)
		{
			Directory.CreateDirectory(Folder);
			// Unique per call: optimisation iterations run in parallel, and two
			// of them landing on the same filename would lose a whole run.
			string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
			string name = Clean(strategyName) + "-" + stamp + "-"
			            + Guid.NewGuid().ToString("N").Substring(0, 6) + ".json";
			string path = Path.Combine(Folder, name);
			File.WriteAllText(path, json);
			return path;
		}

		private static string Clean(string s)
		{
			if (string.IsNullOrEmpty(s)) return "run";
			StringBuilder sb = new StringBuilder(s.Length);
			foreach (char c in s)
				sb.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');
			return sb.ToString();
		}

		private static void Field(StringBuilder sb, string k, string v)
		{
			sb.Append("  \"").Append(k).Append("\": ");
			sb.Append(v == null ? "null" : "\"" + Esc(v) + "\"").Append(",\n");
		}

		private static void Num(StringBuilder sb, string k, double v)
		{
			sb.Append("  \"").Append(k).Append("\": ").Append(D(v)).Append(",\n");
		}

		private static void Bool(StringBuilder sb, string k, bool v)
		{
			sb.Append("  \"").Append(k).Append("\": ").Append(v ? "true" : "false").Append(",\n");
		}

		private static string D(double v)
		{
			if (double.IsNaN(v) || double.IsInfinity(v)) return "null";
			return v.ToString("R", CultureInfo.InvariantCulture);
		}

		private static string Esc(string s)
		{
			if (string.IsNullOrEmpty(s)) return "";
			StringBuilder sb = new StringBuilder(s.Length + 8);
			foreach (char c in s)
			{
				if (c == '"' || c == '\\') sb.Append('\\').Append(c);
				else if (c == '\n') sb.Append("\\n");
				else if (c == '\r') sb.Append("\\r");
				else if (c == '\t') sb.Append("\\t");
				else if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
				else sb.Append(c);
			}
			return sb.ToString();
		}

		private static string Safe(Func<string> f)
		{
			try { return f(); } catch (Exception) { return null; }
		}

		private static T Safe<T>(Func<T> f, T fallback)
		{
			try { return f(); } catch (Exception) { return fallback; }
		}
	}
}
