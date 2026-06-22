import React, { useState, useEffect, useRef } from "react";
import { CandlestickChart, Candle } from "./components/CandlestickChart";
import { EquityChart } from "./components/EquityChart";
import { PortfolioRadar } from "./components/PortfolioRadar";
import { OrderBookVisualizer } from "./components/OrderBookVisualizer";
import { MetricsHUD } from "./components/MetricsHUD";

export default function App() {
  const [isRunning, setIsRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [currentPrice, setCurrentPrice] = useState(65420.0);
  const [equityHistory, setEquityHistory] = useState<number[]>([100000]);
  const [benchmarkHistory, setBenchmarkHistory] = useState<number[]>([100000]);
  const [weights, setWeights] = useState<Record<string, number>>({
    BTC: 0.40,
    ETH: 0.25,
    SOL: 0.15,
    SPY: 0.10,
    CASH: 0.10
  });
  const [orderbook, setOrderbook] = useState<any>({
    bids: [[65400, 2.5], [65390, 4.1], [65380, 5.0], [65370, 7.2], [65360, 10.0]],
    asks: [[65440, 1.8], [65450, 3.2], [65460, 6.1], [65470, 8.4], [65480, 12.5]]
  });
  const [metrics, setMetrics] = useState({
    sharpe: 2.45,
    sortino: 3.12,
    maxDrawdown: 0.084,
    winRate: 0.625,
    dailyPnL: 1.84,
    totalValue: 100000
  });

  const wsRef = useRef<WebSocket | null>(null);

  // Initialize synthetic candle history
  useEffect(() => {
    let p = 64000.0;
    const initial: Candle[] = [];
    for (let i = 0; i < 40; i++) {
      const delta = (Math.random() - 0.48) * 400;
      const open = p;
      const close = p + delta;
      const high = Math.max(open, close) + Math.random() * 150;
      const low = Math.min(open, close) - Math.random() * 150;
      p = close;
      initial.push({
        timestamp: `${i}:00`,
        open,
        high,
        low,
        close,
        volume: 500 + Math.random() * 1000,
        action: i % 8 === 0 ? "REBALANCE" : "HOLD"
      });
    }
    setCandles(initial);
    setCurrentPrice(p);
  }, []);

  // Live simulation tick (local fallback + WebSocket)
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setCurrentPrice(prev => {
        const delta = (Math.random() - 0.48) * (200 * speed);
        const newPrice = Math.max(1000, prev + delta);

        setCandles(cList => {
          const next = [...cList.slice(1)];
          const last = cList[cList.length - 1];
          next.push({
            timestamp: new Date().toLocaleTimeString(),
            open: last.close,
            high: Math.max(last.close, newPrice) + Math.random() * 80,
            low: Math.min(last.close, newPrice) - Math.random() * 80,
            close: newPrice,
            volume: 400 + Math.random() * 1200,
            action: Math.random() > 0.8 ? "REBALANCE" : "HOLD"
          });
          return next;
        });

        // Update equity
        const ret = (newPrice - prev) / prev;
        setEquityHistory(e => {
          const lastVal = e[e.length - 1];
          const nextVal = lastVal * (1 + ret * 0.85 + 0.0001);
          setMetrics(m => ({
            ...m,
            totalValue: nextVal,
            dailyPnL: ((nextVal - 100000) / 100000) * 100
          }));
          return [...e.slice(-50), nextVal];
        });

        setBenchmarkHistory(b => {
          const lastVal = b[b.length - 1];
          const nextVal = lastVal * (1 + ret);
          return [...b.slice(-50), nextVal];
        });

        // Dynamic orderbook & weights
        setOrderbook({
          bids: [
            [newPrice * 0.9995, 2.4 + Math.random()],
            [newPrice * 0.9990, 3.8 + Math.random() * 2],
            [newPrice * 0.9985, 6.2 + Math.random() * 2],
            [newPrice * 0.9980, 8.1 + Math.random() * 3],
            [newPrice * 0.9975, 12.0 + Math.random() * 4]
          ],
          asks: [
            [newPrice * 1.0005, 1.9 + Math.random()],
            [newPrice * 1.0010, 3.5 + Math.random() * 2],
            [newPrice * 1.0015, 5.8 + Math.random() * 2],
            [newPrice * 1.0020, 7.9 + Math.random() * 3],
            [newPrice * 1.0025, 11.4 + Math.random() * 4]
          ]
        });

        return newPrice;
      });
    }, 600 / speed);

    return () => clearInterval(interval);
  }, [isRunning, speed]);

  const injectShock = () => {
    setCurrentPrice(p => p * 0.94);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-4 lg:p-6 font-mono selection:bg-cyan-500 selection:text-black">
      {/* Header */}
      <header className="border-b border-zinc-800 pb-4 mb-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <span className="text-xl font-bold text-cyan-400 tracking-wider">ALPHA-RL</span>
            <span className="text-xs bg-cyan-950 text-cyan-300 border border-cyan-800 px-2 py-0.5 rounded">
              v1.0.0 PRODUCTION
            </span>
            <span className="text-xs bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded">
              PYTORCH 2.6 // PPO-TRANSFORMER
            </span>
          </div>
          <p className="text-xs text-zinc-500 mt-1">
            Deep Reinforcement Learning Execution & Portfolio Allocation Engine
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-3 py-1.5 rounded text-xs font-bold transition-colors ${
              isRunning
                ? "bg-amber-500/20 text-amber-400 border border-amber-500/40 hover:bg-amber-500/30"
                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 hover:bg-emerald-500/30"
            }`}
          >
            {isRunning ? "PAUSE SIM" : "RESUME SIM"}
          </button>

          <button
            onClick={injectShock}
            className="px-3 py-1.5 rounded text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 hover:bg-rose-500/30 transition-colors"
          >
            INJECT VOL SHOCK (-6%)
          </button>

          <div className="flex items-center space-x-2 bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded">
            <span className="text-xs text-zinc-400">SPEED:</span>
            {[1, 2, 5].map(s => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`text-xs px-1.5 py-0.5 rounded ${speed === s ? "bg-cyan-500 text-black font-bold" : "text-zinc-400 hover:text-white"}`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Metrics HUD */}
      <MetricsHUD metrics={metrics} />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <CandlestickChart candles={candles} currentPrice={currentPrice} />
          <EquityChart equityHistory={equityHistory} benchmarkHistory={benchmarkHistory} />
        </div>

        <div className="space-y-4">
          <PortfolioRadar weights={weights} />
          <OrderBookVisualizer orderbook={orderbook} />
        </div>
      </div>
    </div>
  );
}
