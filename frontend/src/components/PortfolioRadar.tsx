import React from "react";

interface Props {
  weights: Record<string, number>;
}

const ASSET_COLORS: Record<string, string> = {
  BTC: "#f59e0b",
  ETH: "#6366f1",
  SOL: "#10b981",
  SPY: "#38bdf8",
  CASH: "#71717a"
};

export const PortfolioRadar: React.FC<Props> = ({ weights }) => {
  const entries = Object.entries(weights);

  return (
    <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-4 shadow-xl backdrop-blur">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono font-bold text-zinc-300 tracking-wider">ACTIVE ASSET ALLOCATION</span>
        <span className="text-[10px] font-mono px-2 py-0.5 bg-emerald-950 text-emerald-400 border border-emerald-800 rounded">
          DIRICHLET REBALANCED
        </span>
      </div>

      <div className="space-y-2.5">
        {entries.map(([asset, weight]) => {
          const pct = Math.max(0, Math.min(100, weight * 100));
          const color = ASSET_COLORS[asset] || "#a1a1aa";
          return (
            <div key={asset}>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-zinc-200 font-semibold">{asset}</span>
                <span className="text-zinc-400">{pct.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${pct}%`, backgroundColor: color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
