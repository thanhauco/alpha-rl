import React from "react";

interface Props {
  orderbook?: {
    bids: [number, number][];
    asks: [number, number][];
  };
}

export const OrderBookVisualizer: React.FC<Props> = ({ orderbook }) => {
  const bids = orderbook?.bids || [];
  const asks = orderbook?.asks || [];

  return (
    <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-4 shadow-xl backdrop-blur">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono font-bold text-zinc-300 tracking-wider">L2 ORDER BOOK DEPTH</span>
        <span className="text-[10px] font-mono text-zinc-500">ALMGREN-CHRISS SIM</span>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
        <div>
          <div className="text-[10px] text-zinc-500 border-b border-zinc-800 pb-1 mb-1 flex justify-between">
            <span>BID ($)</span>
            <span>QTY</span>
          </div>
          <div className="space-y-1">
            {bids.slice(0, 5).map(([price, qty], i) => (
              <div key={i} className="flex justify-between text-emerald-400">
                <span>{price.toFixed(2)}</span>
                <span className="text-zinc-400">{qty.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="text-[10px] text-zinc-500 border-b border-zinc-800 pb-1 mb-1 flex justify-between">
            <span>ASK ($)</span>
            <span>QTY</span>
          </div>
          <div className="space-y-1">
            {asks.slice(0, 5).map(([price, qty], i) => (
              <div key={i} className="flex justify-between text-rose-400">
                <span>{price.toFixed(2)}</span>
                <span className="text-zinc-400">{qty.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
