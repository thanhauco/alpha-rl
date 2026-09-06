# Alpha-RL Architecture Specification

Alpha-RL is an institutional-grade deep reinforcement learning platform engineered specifically for quantitative finance, multi-asset portfolio rebalancing, and high-frequency optimal order execution.

This document details the architectural blueprints, mathematical foundations, data contracts, and component interaction models of the system.

---

## 1. High-Level Macro Architecture

The system is organized into five decoupled yet harmonized layers:

```mermaid
graph TB
    subgraph DataLayer ["1. Data & Market Simulation Layer"]
        Heston["Heston Stochastic Volatility Generator<br/><code>alpha_rl.data.generators.HestonProcess</code>"]
        Merton["Merton Jump-Diffusion Process<br/><code>alpha_rl.data.generators.MertonJumpDiffusion</code>"]
        MultiAsset["Multi-Asset Correlated Generator<br/><code>alpha_rl.data.generators.MultiAssetMarketDataGenerator</code>"]
        Features["Technical & Microstructure Pipeline<br/><code>alpha_rl.data.features.build_feature_matrix</code>"]
    end

    subgraph EnvLayer ["2. Gymnasium Market Environments"]
        TradingEnv["Multi-Asset Portfolio Environment<br/><code>alpha_rl.envs.trading_env.MultiAssetTradingEnv</code>"]
        ExecEnv["Microstructure Execution Environment<br/><code>alpha_rl.envs.execution_env.OrderExecutionEnv</code>"]
        AC_Bench["Almgren-Chriss Optimal Solver<br/><code>alpha_rl.envs.execution_env.AlmgrenChrissBenchmark</code>"]
    end

    subgraph RLLayer ["3. Deep Learning & RL Engine"]
        PatchTrans["Temporal Patch Transformer<br/><code>alpha_rl.models.transformer.TemporalPatchEncoder</code>"]
        CrossAttn["Cross-Asset Attention Module<br/><code>alpha_rl.models.transformer.CrossAssetAttention</code>"]
        ActorCritic["Gaussian Actor-Critic Policy<br/><code>alpha_rl.models.policy.ActorCritic</code>"]
        PPO["PPO Agent & GAE Optimizer<br/><code>alpha_rl.agents.ppo.PPOAgent</code>"]
    end

    subgraph RiskLayer ["4. Risk Guardrails & Validation"]
        RiskManager["Institutional Risk Manager<br/><code>alpha_rl.risk.manager.RiskManager</code>"]
        PurgedKFold["Purged & Embargoed K-Fold Split<br/><code>alpha_rl.evaluation.backtest.purged_and_embargoed_kfold</code>"]
        BacktestEngine["Vectorized Tearsheet Engine<br/><code>alpha_rl.evaluation.backtest.BacktestEngine</code>"]
    end

    subgraph ServingLayer ["5. Serving Gateway & UI Terminal"]
        FastAPI["FastAPI REST & WebSocket Server<br/><code>backend.server.app</code>"]
        LiveWS["Real-Time Streaming Channel<br/><code>/ws/live-trading</code>"]
        ReactUI["React 19 Quant Terminal<br/><code>frontend/src/App.tsx</code>"]
    end

    DataLayer --> EnvLayer
    EnvLayer --> RLLayer
    RLLayer --> RiskLayer
    RiskLayer --> ServingLayer
    ServingLayer <-->|WebSocket Stream & REST| ReactUI
```

---

## 2. Neural Policy Architecture: Temporal Patch Transformer

Financial series contain localized temporal momentum and cross-asset correlation. Alpha-RL replaces slow recurrent architectures with a hybrid **Strided Patch Temporal Transformer** and **Cross-Asset Attention** mechanism.

```mermaid
graph LR
    subgraph Input ["Input Tensor"]
        Raw["Observation Tensor<br/>(B, T=20, F=6)"]
    end

    subgraph Patching ["Patch Temporal Tokenization"]
        Unfold["Strided Unfold<br/>(B, N_patches=9, F=6, P=4)"]
        LinearProj["Linear Projection<br/>Linear(P=4, d_model=64)"]
    end

    subgraph Attention ["Cross-Asset Multi-Head Attention"]
        MeanPool["Mean Across Patches<br/>(B, F=6, d_model=64)"]
        MHA["Multi-Head Self-Attention<br/>4 Heads, Pre-LN, GELU FFN"]
        Residual["Residual + LayerNorm"]
    end

    subgraph Pooling ["Trunk Pooling"]
        GlobalPool["Feature Dimension Mean<br/>(B, d_model=64)"]
        OutProj["Trunk Projection<br/>Linear(64, 128) + Tanh"]
    end

    subgraph Heads ["Dual Actor-Critic Heads"]
        subgraph Actor ["Gaussian Actor"]
            ActorMLP["Linear(128, 128) -> Tanh -> Linear(128, A)"]
            Sigmoid["Sigmoid Simplex Mean &mu;(s)"]
            LogStd["Clamped log(&sigma;) &isin; [-20, 2]"]
            NormalDist["Reparameterized Normal(&mu;, &sigma;)"]
        end

        subgraph Critic ["Value Critic"]
            CriticMLP["Linear(128, 128) -> Tanh -> Linear(128, 1)"]
            ScalarValue["Scalar Value V(s)"]
        end
    end

    Raw --> Unfold --> LinearProj --> MeanPool --> MHA --> Residual --> GlobalPool --> OutProj
    OutProj --> ActorMLP --> Sigmoid --> NormalDist
    LogStd --> NormalDist
    OutProj --> CriticMLP --> ScalarValue
```

### Mathematical Formulation

1. **Strided Patch Unfolding**:
   For lookback series $X \in \mathbb{R}^{B \times T \times F}$, we apply window slicing of length $P=4$ and stride $S=2$:
   $$X_{\text{patch}} = \text{unfold}(X, \text{size}=P, \text{step}=S) \in \mathbb{R}^{B \times N_p \times F \times P}$$
   $$Z_p = W_{\text{proj}} X_{\text{patch}} + b_{\text{proj}} \in \mathbb{R}^{B \times N_p \times F \times D}$$

2. **Cross-Asset Attention**:
   With $H = 4$ attention heads, we model correlation between asset features:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   $$\tilde{Z} = \text{LayerNorm}(Z + \text{MHA}(Z, Z, Z))$$

3. **Gaussian Policy Output**:
   The action vector $a \in [0, 1]^A$ represents target asset allocation weights:
   $$\mu(s) = \sigma(\text{Linear}(h)), \quad \log \sigma(s) = \text{clamp}(\theta_{\sigma}, -20, 2)$$
   $$a \sim \mathcal{N}(\mu(s), \sigma(s)^2), \quad a_{\text{clamped}} = \text{clip}(a, 0, 1)$$

---

## 3. Environment Step Lifecycle & Microstructure Engine

The trading environment enforces physical execution constraints, non-negative cash balances, transaction fee schedules, and market impact slippage.

```mermaid
sequenceDiagram
    autonumber
    participant Agent as PPO Policy Agent
    participant Env as MultiAssetTradingEnv
    participant Exec as Slippage & Market Impact
    participant Risk as RiskManager
    participant Reward as Reward Engine (DSR)

    Agent->>Env: step(action: target_weights)
    Env->>Risk: check_allocation(target_weights, portfolio_val, realized_vol)
    
    alt Circuit Breaker Tripped
        Risk-->>Env: override: [0.0, ..., 0.0] (100% Cash)
    else Normal Volatility
        Risk-->>Env: scaled_weights = target_weights * (target_vol / realized_vol)
    end

    Env->>Env: Enforce max allocation = 1.0 - cash_buffer_pct
    Env->>Env: Calculate delta_holdings = target_holdings - current_holdings
    Env->>Exec: Calculate turnover & Kyle's lambda slippage
    Exec-->>Env: fee = turnover * taker_fee, slippage = lambda * (trade^2) / vol
    Env->>Env: Deduct cash: cash = portfolio_val - stock_val - fee - slippage
    Env->>Env: Advance time step t -> t + 1, observe next prices
    Env->>Env: Update portfolio_value = cash + sum(holdings * next_prices)
    Env->>Reward: compute_reward(new_val, prev_val, reward_mode)
    Reward-->>Env: return differential_sharpe or log_return
    Env-->>Agent: return (observation, reward, terminated, info)
```

---

## 4. Almgren-Chriss Optimal Liquidation Microstructure

For optimal execution of large parent orders without moving the market, Alpha-RL implements the Almgren & Chriss (2000) optimal liquidation framework:

```mermaid
graph TD
    subgraph ProblemSetup ["Parent Order Liquidation"]
        Inventory["Initial Inventory X_0 = 10,000 shares"]
        Horizon["Execution Horizon T = 60 intervals"]
    end

    subgraph ImpactModels ["Price Dynamics with Market Impact"]
        Perm["Permanent Impact: g(v) = &gamma; v<br/>Causes irreversible price drift: dS = -g(v)dt + &sigma; dW"]
        Temp["Temporary Impact: h(v) = &eta; v<br/>Causes temporary slippage on executed trade: P_exec = P_mid - h(v)"]
    end

    subgraph ClosedForm ["Almgren-Chriss Analytical Benchmark"]
        Kappa["Hyperbolic Decay Parameter:<br/>&kappa; &approx; sqrt(&lambda; &sigma;^2 / &eta;)"]
        Trajectory["Optimal Trajectory:<br/>x_j = X * sinh(&kappa;(T - t_j)) / sinh(&kappa; T)"]
    end

    subgraph Comparison ["Execution Trade-Off"]
        Fast["Fast Liquidation (TWAP): Low Market Risk, High Impact Cost"]
        Slow["Slow Liquidation: Low Impact Cost, High Market Drift Risk"]
        AC_Opt["Almgren-Chriss / RL Agent: Pareto Optimal Frontier"]
    end

    ProblemSetup --> ImpactModels
    ImpactModels --> ClosedForm
    ClosedForm --> Comparison
```

---

## 5. Institutional Risk Circuit Breaker State Machine

The Risk Management Layer safeguards the portfolio against catastrophic drawdowns and regime changes:

```mermaid
stateDiagram-v2
    [*] --> ActiveTrading : Initialization (NAV = $100k)

    state ActiveTrading {
        [*] --> VolatilityMonitoring
        VolatilityMonitoring --> ScalePositions : Calculate Realized Vol &sigma;_t
        ScalePositions --> PositionConstraint : Cap Leverage <= 1.0
        PositionConstraint --> VolatilityMonitoring : Next Step
    }

    ActiveTrading --> CircuitBreakerTripped : Drawdown >= Max DD Limit (15%)
    
    state CircuitBreakerTripped {
        [*] --> LiquidateAllToCash
        LiquidateAllToCash --> CooloffPeriod : Hold 100% Cash Buffer
        CooloffPeriod --> CooloffPeriod : Await Market Stabilization
    }

    CircuitBreakerTripped --> ActiveTrading : Manual Operator Reset / Regime Normalized
```

---

## 6. Purged & Embargoed Walk-Forward Validation

Standard random K-Fold cross-validation causes fatal data leakage in financial machine learning due to overlapping return labels and serial correlation. Alpha-RL enforces **Purged and Embargoed Walk-Forward Splitting**:

```mermaid
gantt
    title Purged and Embargoed Walk-Forward Validation (Fold i)
    dateFormat  X
    axisFormat %s

    section Train Left
    Training Segment 1        :done, train1, 0, 35
    section Purge Gap 1
    Purged Overlap Window     :crit, purge1, 35, 38
    section Test Split
    Test Evaluation Fold      :active, test, 38, 60
    section Post-Test Embargo
    Embargo Buffer (2%)       :crit, embargo, 60, 64
    section Train Right
    Training Segment 2        :done, train2, 64, 100
```

- **Purging**: Removes training samples whose labels overlap with the test fold window.
- **Embargoing**: Inserts an embargo period immediately following the test fold to prevent autoregressive information leakage into subsequent training intervals.

---

## 7. Real-Time Telemetry & WebSocket Streaming Loop

The WebSocket architecture allows sub-millisecond dispatch of tick updates, live model weights, and order book states to the web terminal:

```mermaid
sequenceDiagram
    autonumber
    participant Browser as React 19 Quant Terminal
    participant WS as FastAPI WebSocket (/ws/live-trading)
    participant Engine as Alpha-RL Engine Loop
    participant LOB as L2 Order Book Depth

    Browser->>WS: Connect WebSocket
    WS-->>Browser: Connection Accepted (HTTP 101 Switching Protocols)
    
    loop Every 400ms (Adjustable via Speed Slider)
        Engine->>Engine: Sample next market tick & technical features
        Engine->>Engine: Actor-Critic Policy Inference (weights)
        Engine->>LOB: Query Top-5 Bid/Ask Levels & Queue Sizes
        Engine->>Engine: Update Portfolio Value & Drawdown Metrics
        Engine->>WS: Formulate JSON Telemetry Payload
        WS-->>Browser: Broadcast: {step, price, weights, orderbook, pnl, action}
        Browser->>Browser: Update Candlestick Chart (SVG Canvas)
        Browser->>Browser: Animate Asset Allocation Bars
        Browser->>Browser: Render Order Book Depth Ladder
    end

    Browser->>WS: User Injects Vol Shock (-6%)
    WS->>Engine: Mutate Price Level
    Engine->>Engine: Risk Manager Vol Targeting Triggered
    Engine-->>WS: Broadcast De-risked Portfolio State
```

---

## 8. Directory & Dependency Map

```
alpha-rl/
├── alpha_rl/
│   ├── agents/
│   │   └── ppo.py                 # PPO implementation with GAE and value clipping
│   ├── data/
│   │   ├── generators.py          # Heston, Merton Jump-Diffusion, and Multi-Asset data
│   │   └── features.py            # Garman-Klass, Parkinson, RSI, MACD, Bollinger Bands
│   ├── envs/
│   │   ├── trading_env.py         # Gymnasium multi-asset continuous trading env
│   │   └── execution_env.py       # Almgren-Chriss L2 order book execution env
│   ├── evaluation/
│   │   └── backtest.py            # Purged & Embargoed K-Fold split and tear-sheet
│   ├── models/
│   │   ├── policy.py              # Continuous Gaussian Actor & Value Critic
│   │   └── transformer.py         # Patch Temporal Transformer & Cross-Asset Attention
│   └── risk/
│       └── manager.py             # Max Drawdown circuit breaker & Volatility targeting
├── backend/
│   └── server.py                  # FastAPI REST and WebSocket streaming gateway
├── frontend/                      # React 19 + TypeScript + Vite quant web terminal
├── tests/                         # Pytest unit and integration test suite
├── .github/workflows/ci.yml       # GitHub Actions CI matrix
├── pyproject.toml                 # Package definition
├── requirements.txt
└── README.md
```
