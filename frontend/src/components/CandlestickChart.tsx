import React from "react";

export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  action?: string;
}

interface Props {
  candles: Candle[];
  currentPrice: number;
}

export const CandlestickChart: React.FC<Props> = ({ candles, currentPrice }) => {
  if (!candles || candles.length === 0) {
    return <div className="p-4 text-zinc-500 font-mono text-xs">Loading market feed...</div>;
  }

  const minPrice = Math.min(...candles.map(c => c.low)) * 0.998;
  const maxPrice = Math.max(...candles.map(c => c.high)) * 1.002;
  const priceRange = maxPrice - minPrice || 1.0;

  const width = 800;
  const height = 280;
  const candleWidth = Math.max(3, (width / candles.length) * 0.7);

  const getY = (price: number) => height - ((price - minPrice) / priceRange) * height;

  return (
    <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-4 shadow-xl backdrop-blur">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <span className="text-emerald-400 font-mono font-bold text-sm tracking-wider">BTC/USD // 1H</span>
          <span className="text-zinc-400 text-xs font-mono">LIVE EXECUTION FEED</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-zinc-500 font-mono">SPOT:</span>
          <span className="text-sm font-mono font-bold text-zinc-100">${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
        </div>
      </div>

      <div className="relative overflow-hidden w-full" style={{ height: `${height}px` }}>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
          {/* Grid lines */}
          {[0.2, 0.4, 0.6, 0.8].map((ratio, idx) => {
            const y = height * ratio;
            const priceLevel = maxPrice - ratio * priceRange;
            return (
              <g key={idx}>
                <line x1="0" y1={y} x2={width} y2={y} stroke="#27272a" strokeDasharray="3 3" />
                <text x={width - 60} y={y - 4} fill="#71717a" fontSize="9" fontFamily="monospace">
                  ${priceLevel.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Candles */}
          {candles.map((c, i) => {
            const x = (i / candles.length) * width + candleWidth / 2;
            const yHigh = getY(c.high);
            const yLow = getY(c.low);
            const yOpen = getY(c.open);
            const yClose = getY(c.close);
            const isGreen = c.close >= c.open;
            const candleTop = Math.min(yOpen, yClose);
            const candleHeight = Math.max(2, Math.abs(yClose - yOpen));
            const color = isGreen ? "#10b981" : "#f43f5e";

            return (
              <g key={i}>
                {/* Wick */}
                <line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={color} strokeWidth="1" />
                {/* Body */}
                <rect
                  x={x - candleWidth / 2}
                  y={candleTop}
                  width={candleWidth}
                  height={candleHeight}
                  fill={color}
                  rx="1"
                />
                {/* Action marker */}
                {c.action && c.action !== "HOLD" && (
                  <circle
                    cx={x}
                    cy={isGreen ? yHigh - 8 : yLow + 8}
                    r="3.5"
                    fill={isGreen ? "#34d399" : "#fb7185"}
                    stroke="#09090b"
                    strokeWidth="1.5"
                  />
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
