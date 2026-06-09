import React from "react";

interface Props {
  equityHistory: number[];
  benchmarkHistory: number[];
}

export const EquityChart: React.FC<Props> = ({ equityHistory, benchmarkHistory }) => {
  if (equityHistory.length < 2) {
    return <div className="p-4 text-zinc-500 font-mono text-xs">Waiting for equity updates...</div>;
  }

  const allVals = [...equityHistory, ...benchmarkHistory];
  const minVal = Math.min(...allVals) * 0.99;
  const maxVal = Math.max(...allVals) * 1.01;
  const range = maxVal - minVal || 1.0;

  const width = 800;
  const height = 180;

  const getPoints = (arr: number[]) => {
    return arr
      .map((val, idx) => {
        const x = (idx / (arr.length - 1)) * width;
        const y = height - ((val - minVal) / range) * height;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const currAlphaVal = equityHistory[equityHistory.length - 1];
  const currBenchVal = benchmarkHistory[benchmarkHistory.length - 1];
  const alphaRet = ((currAlphaVal - equityHistory[0]) / equityHistory[0]) * 100;
  const benchRet = ((currBenchVal - benchmarkHistory[0]) / benchmarkHistory[0]) * 100;

  return (
    <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-4 shadow-xl backdrop-blur mt-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-4">
          <span className="text-xs font-mono font-bold text-zinc-300">PORTFOLIO VALUATION VS BENCHMARK</span>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            <span className="text-xs font-mono text-cyan-300">ALPHA-RL ({alphaRet >= 0 ? "+" : ""}{alphaRet.toFixed(2)}%)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-zinc-500"></span>
            <span className="text-xs font-mono text-zinc-400">B&H BENCHMARK ({benchRet >= 0 ? "+" : ""}{benchRet.toFixed(2)}%)</span>
          </div>
        </div>
        <div className="text-xs font-mono text-emerald-400 font-semibold">
          SPREAD: {(alphaRet - benchRet).toFixed(2)}%
        </div>
      </div>

      <div className="relative w-full overflow-hidden" style={{ height: `${height}px` }}>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
          {/* Baseline line at 100k */}
          {equityHistory[0] && (
            <line
              x1="0"
              y1={height - ((equityHistory[0] - minVal) / range) * height}
              x2={width}
              y2={height - ((equityHistory[0] - minVal) / range) * height}
              stroke="#3f3f46"
              strokeDasharray="2 2"
            />
          )}

          {/* Benchmark line */}
          <polyline
            fill="none"
            stroke="#71717a"
            strokeWidth="1.5"
            points={getPoints(benchmarkHistory)}
          />

          {/* Alpha-RL line */}
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth="2.5"
            points={getPoints(equityHistory)}
          />
        </svg>
      </div>
    </div>
  );
};
