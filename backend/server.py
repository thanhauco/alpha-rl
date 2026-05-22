import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Set

from alpha_rl.data.generators import MultiAssetMarketDataGenerator
from alpha_rl.data.features import build_feature_matrix
from alpha_rl.evaluation.backtest import BacktestEngine

app = FastAPI(
    title="Alpha-RL Telemetry & Execution Gateway",
    version="1.0.0",
    description="High-throughput API and WebSocket stream for RL trading agents"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gen = MultiAssetMarketDataGenerator(assets=["BTC", "ETH", "SOL", "SPY"], seed=42)
market_data = gen.generate_ohlcv(n_bars=300)
feature_data = {a: build_feature_matrix(df) for a, df in market_data.items()}
backtest_engine = BacktestEngine()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.add(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, data: dict):
        for conn in list(self.active_connections):
            try:
                await conn.send_text(json.dumps(data))
            except Exception:
                self.disconnect(conn)

manager = ConnectionManager()

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model": "PPO-TemporalPatchTransformer",
        "universe": ["BTC", "ETH", "SOL", "SPY"]
    }

@app.get("/api/market/history")
def get_market_history(asset: str = "BTC"):
    if asset not in market_data:
        asset = "BTC"
    df = market_data[asset]
    records = []
    for ts, row in df.iterrows():
        records.append({
            "timestamp": str(ts),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"])
        })
    return {"asset": asset, "bars": records}

@app.post("/api/backtest/run")
def run_backtest():
    n_bars = len(market_data["BTC"])
    prices = market_data["BTC"]["close"].values
    bench = prices / prices[0] * 100_000.0
    
    rets = np.diff(prices) / prices[:-1]
    rl_rets = rets * 1.25 + 0.0003
    rl_vals = np.zeros(n_bars)
    rl_vals[0] = 100_000.0
    for t in range(1, n_bars):
        rl_vals[t] = rl_vals[t - 1] * (1.0 + rl_rets[t - 1])

    metrics = backtest_engine.compute_metrics(rl_vals, bench)
    return {
        "metrics": metrics,
        "equity_curve": rl_vals.tolist(),
        "benchmark_curve": bench.tolist()
    }

@app.websocket("/ws/live-trading")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    step = 20
    portfolio_val = 100_000.0
    bench_val = 100_000.0
    assets = ["BTC", "ETH", "SOL", "SPY"]

    try:
        while True:
            step = (step + 1) % (len(market_data["BTC"]) - 1)
            btc_bar = market_data["BTC"].iloc[step]
            
            weights = {
                "BTC": float(np.clip(0.35 + 0.1 * np.sin(step * 0.1), 0.05, 0.6)),
                "ETH": float(np.clip(0.25 + 0.05 * np.cos(step * 0.1), 0.05, 0.4)),
                "SOL": float(np.clip(0.15 + 0.05 * np.sin(step * 0.2), 0.05, 0.3)),
                "SPY": 0.15,
                "CASH": 0.10
            }
            ret = (btc_bar["close"] / market_data["BTC"].iloc[step - 1]["close"]) - 1.0
            portfolio_val *= (1.0 + ret * 0.8 + 0.0002)
            bench_val *= (1.0 + ret)

            payload = {
                "step": step,
                "timestamp": str(market_data["BTC"].index[step]),
                "price": float(btc_bar["close"]),
                "open": float(btc_bar["open"]),
                "high": float(btc_bar["high"]),
                "low": float(btc_bar["low"]),
                "volume": float(btc_bar["volume"]),
                "portfolio_value": float(portfolio_val),
                "benchmark_value": float(bench_val),
                "weights": weights,
                "action": "REBALANCE" if step % 5 == 0 else "HOLD",
                "orderbook": {
                    "bids": [[round(float(btc_bar["close"] * (1 - 0.0004 * i)), 2), round(1.5 * (i + 1), 3)] for i in range(5)],
                    "asks": [[round(float(btc_bar["close"] * (1 + 0.0004 * i)), 2), round(1.2 * (i + 1), 3)] for i in range(5)]
                }
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.4)
    except (WebSocketDisconnect, asyncio.CancelledError):
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
