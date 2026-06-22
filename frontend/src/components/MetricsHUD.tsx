import React from "react";

interface Metrics {
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  winRate: number;
  dailyPnL: number;
  totalValue: number;
}

interface Props {
  metrics: Metrics;
}

export const MetricsHUD: React.FC<Props> = ({ metrics }) => {
  const isProfit = metrics.dailyPnL >= 0;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-3">
        <div className="text-[10px] font-mono text-zinc-500 tracking-wider">NAV (USD)</div>
        <div className="text-lg font-mono font-bold text-zinc-100">
          ${metrics.totalValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </div>
      </div>

      <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-3">
        <div className="text-[10px] font-mono text-zinc-500 tracking-wider">DAILY P&L</div>
        <div className={`text-lg font-mono font-bold ${isProfit ? "text-emerald-400" : "text-rose-400"}`}>
          {isProfit ? "+" : ""}{metrics.dailyPnL.toFixed(2)}%
        </div>
      </div>

      <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-3">
        <div className="text-[10px] font-mono text-zinc-500 tracking-wider">ANN. SHARPE</div>
        <div className="text-lg font-mono font-bold text-cyan-400">
          {metrics.sharpe.toFixed(2)}
        </div>
      </div>

      <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-3">
        <div className="text-[10px] font-mono text-zinc-500 tracking-wider">SORTINO RATIO</div>
        <div className="text-lg font-mono font-bold text-indigo-400">
          {metrics.sortino.toFixed(2)}
        </div>
      </div>

      <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-3">
        <div className="text-[10px] font-mono text-zinc-500 tracking-wider">MAX DRAWDOWN</div>
        <div className="text-lg font-mono font-bold text-amber-400">
          -{(metrics.maxDrawdown * 100).toFixed(1)}%
        </div>
      </div>

      <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-3">
        <div className="text-[10px] font-mono text-zinc-500 tracking-wider">WIN RATE</div>
        <div className="text-lg font-mono font-bold text-emerald-400">
          {(metrics.winRate * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
};
