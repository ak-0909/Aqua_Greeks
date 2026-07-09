<p align="center">
  <h1 align="center">AQUA Greeks</h1>
  <p align="center">
    <strong>Advanced Quantitative Underlying Analysis — Greek-Based Options Backtester</strong>
  </p>
  <p align="center">
    <em>NIFTY 50 Multi-Strategy Options Backtesting with HMM + GMM Regime Detection</em>
  </p>
  <p align="center">
    <a href="#results"><img src="https://img.shields.io/badge/Profit_Factor-2.34-brightgreen?style=for-the-badge" alt="PF"/></a>
    <a href="#results"><img src="https://img.shields.io/badge/Win_Rate-72.1%25-blue?style=for-the-badge" alt="Win Rate"/></a>
    <a href="#results"><img src="https://img.shields.io/badge/Max_DD--1.85%25-orange?style=for-the-badge" alt="Max DD"/></a>
    <a href="#results"><img src="https://img.shields.io/badge/Return-+7.79%25-green?style=for-the-badge" alt="Return"/></a>
  </p>
</p>

---

## Overview

AQUA Greeks is an **advanced options backtesting framework** for NIFTY 50 index options that combines:

- **Black-Scholes Greeks** — real-time delta, gamma, vega, theta computation for option pricing and risk management
- **Hidden Markov Model (HMM)** — temporal regime detection with state transition modeling
- **Gaussian Mixture Model (GMM)** — unsupervised market regime clustering
- **Walk-Forward Optimization** — quarterly model re-fitting to prevent look-ahead bias

The system runs **4 options strategies** (Iron Condor, Credit Put Spread, Long Straddle, Bull Put Spread on Dips), each gated to only execute in its optimal market regime — dramatically reducing losing trades.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │         NIFTY 50 Price Data (yfinance)      │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │         Feature Engineering                  │
                    │  Log Returns │ HV10/20/30 │ IV/HV Ratio     │
                    │  RSI(14) │ Bollinger Width │ Vol Z-Score     │
                    │  Realized Skewness │ Mean-Rev Z-Score        │
                    └──────────────────┬──────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────────┐
              │       Walk-Forward Regime Detection                  │
              │                                                      │
              │  ┌──────────────┐     ┌──────────────────────┐      │
              │  │     GMM      │────▶│        HMM           │      │
              │  │  (Discover   │     │  (Temporal Sequence   │      │
              │  │   Clusters)  │     │   + Transitions)      │      │
              │  └──────────────┘     └──────────┬───────────┘      │
              │                                   │                  │
              │  Train: 200 days │ Re-fit: Quarterly │ No Lookahead │
              └───────────────────────────────────┬────────────────┘
                                                  │
                         ┌────────────────────────▼───────────────┐
                         │         3 Market Regimes               │
                         │                                         │
                         │  🟢 LOW_VOL_TRENDING    (26% of days)  │
                         │  🔵 RANGE_BOUND         (40% of days)  │
                         │  🔴 HIGH_VOL_STRESS     (34% of days)  │
                         └────────────────────────┬───────────────┘
                                                  │
                    ┌─────────────────────────────▼──────────────┐
                    │         Strategy-Regime Gating              │
                    │                                              │
                    │  Iron Condor      → 🔵 RANGE_BOUND only    │
                    │  Credit Put Spread → 🟢 LOW_VOL_TRENDING   │
                    │  Long Straddle    → 🔴 HIGH_VOL transition │
                    │  Bull Put (Dip)   → Skip if 🔴 > 60%       │
                    └─────────────────────────────┬──────────────┘
                                                  │
                    ┌─────────────────────────────▼──────────────┐
                    │     Black-Scholes Greeks Engine              │
                    │  Option Pricing │ Delta │ Gamma │ Vega       │
                    │  Dynamic Stops │ Position Sizing │ Exits     │
                    └─────────────────────────────────────────────┘
```

---

## Strategies

### 1. Iron Condor (Range-Bound Regime)
Sells OTM call + put spreads when the market is **range-bound** (HMM confidence > 55%). Collects premium from theta decay in non-trending markets. Dynamic strike offsets adjust based on regime volatility.

### 2. Credit Put Spread (Bullish Trending Regime)
Sells OTM put spread when the market is in a **low-vol uptrend** (HMM confidence > 50%). Captures premium from bullish momentum with SMA confirmation. Wider strikes (350pt OTM) for safety.

### 3. Long Straddle (Volatility Expansion)
Buys ATM call + put when the HMM detects a **transition toward high-vol stress**. Profits from large moves in either direction. Includes leg-lock profit-taking and IV crush protection.

### 4. Bull Put Spread on Dips ⭐ (Mean-Reversion)
Enters after NIFTY dips > 2.5% in 5 days — but **only if high-vol stress probability < 60%** (avoids catching falling knives). Short-dated (15 DTE) for fast theta capture. The star strategy with 78.3% win rate and 2.67 profit factor.

---

## Results

<a name="results"></a>

### Backtest Period: Jan 2022 — Dec 2024 (3 years)

| Metric | Value |
|---|---|
| **Initial Capital** | ₹5,00,000 |
| **Total P&L** | **₹38,975** |
| **Total Return** | **+7.79%** |
| **Annualized Return** | **+2.53%** |
| **Win Rate** | **72.1%** |
| **Profit Factor** | **2.34** |
| **Max Drawdown** | **-1.85%** |
| **Total Trades** | 43 |
| **Stop Losses Hit** | 1 (out of 43 trades) |

### Strategy Breakdown

| Strategy | P&L | Trades | Win Rate | Profit Factor |
|---|---|---|---|---|
| Iron Condor | +₹2,714 | 9 | 55.6% | 1.53 |
| Credit Put Spread | -₹1,283 | 5 | 60.0% | 0.74 |
| Long Straddle | +₹16,036 | 6 | 83.3% | 3.56 |
| **Bull Put Spread (Dip)** | **+₹21,508** | **23** | **78.3%** | **2.67** |

### Improvement Over Baseline (v2.0 → v3.0)

| Metric | Before (No HMM/GMM) | After (HMM/GMM) | Improvement |
|---|---|---|---|
| Total P&L | ₹26,572 | **₹38,975** | **+47%** |
| Profit Factor | 1.27 | **2.34** | **+84%** |
| Win Rate | 62.7% | **72.1%** | +9.4pp |
| Max Drawdown | -8.22% | **-1.85%** | **77% less risk** |
| Stop Losses | 24 | **1** | **96% reduction** |
| Iron Condor P&L | -₹16,806 | **+₹2,714** | Turned profitable |
| Credit Put Spread P&L | -₹12,983 | **-₹1,283** | 90% loss reduction |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Options Pricing** | Black-Scholes model (scipy) |
| **Regime Detection** | HMM (hmmlearn) + GMM (scikit-learn) |
| **Market Data** | yfinance (NIFTY 50 spot data) |
| **Visualization** | Plotly (interactive HTML dashboards) |
| **Analysis** | pandas, numpy |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Falgun-Dadhich07/AQUA_Greeks.git
cd AQUA_Greeks

# Install dependencies
pip install yfinance pandas numpy scipy scikit-learn hmmlearn plotly
```

## Usage

### Run the Full Backtest

```bash
python "greeks_backtester (1).py"
```

This will:
1. Download NIFTY 50 data (2022–2024) from yfinance
2. Run walk-forward HMM/GMM regime detection (9 quarterly re-fits)
3. Execute all 4 strategies with regime gating
4. Print detailed trade log and summary statistics
5. Generate interactive Plotly dashboard → `multi_strategy_dashboard.html`
6. Save results to `multi_strategy_results.json`

### Custom Date Range

```python
from greeks_backtester import run_all_strategies

# Run with custom dates
trades, alerts, metrics, strat_metrics = run_all_strategies(
    from_date='2023-01-01',
    to_date='2024-12-31'
)
```

### View the Dashboard

Open `multi_strategy_dashboard.html` in any browser. The dashboard includes:
- Cumulative P&L by strategy
- Per-trade P&L waterfall
- Strategy P&L contribution (pie chart)
- Win rate comparison
- NIFTY price with regime overlay
- Regime probability time series

---

## Project Structure

```
AQUA_Greeks/
├── greeks_backtester (1).py        # Main backtester v3.0 (HMM/GMM)
├── greeks_backtester.py            # Legacy backtester
├── AQUA_1 (1).ipynb                # Jupyter analysis notebook
├── Backtester.ipynb                # Backtester notebook
├── multi_strategy_dashboard.html   # Interactive Plotly dashboard
├── multi_strategy_results.json     # Latest backtest results
├── NIFTY_Backtester_Combined_Report.pdf  # Report
└── README.md                       # This file
```

---

## How HMM/GMM Regime Detection Works

### 1. Feature Engineering
Six features are computed from raw NIFTY data (all backward-looking, no lookahead):
- **Log Returns** — daily price changes
- **HV20** — 20-day historical volatility (annualized)
- **IV/HV Ratio** — implied vs realized volatility gap
- **RSI(14)** — momentum oscillator
- **Volume Z-Score** — unusual volume detection
- **Bollinger Band Width** — normalized price dispersion

### 2. GMM Clustering
A 3-component Gaussian Mixture Model discovers natural market clusters:
- Clusters are sorted by mean volatility (lowest → TRENDING, mid → RANGE_BOUND, highest → STRESS)
- Uses full covariance matrices to capture feature correlations

### 3. HMM Temporal Modeling
A 3-state Gaussian HMM adds temporal dependence:
- Models state transitions (e.g., "93% chance calm market stays calm tomorrow")
- Provides regime confidence scores via `predict_proba()`
- Initialized with GMM parameters for faster convergence

### 4. Walk-Forward Training
- **Training window**: 200 trading days (~10 months)
- **Re-fit interval**: 63 trading days (~quarterly)
- **Context window**: 40 days for HMM sequence prediction
- Result: 9 re-fit windows, 538 predicted days, 87.76% average confidence

---

## Configuration

All strategy parameters are in the `CFG` dictionary at the top of the backtester. Key parameters:

```python
CFG = {
    'INITIAL_CAPITAL'    : 500_000,    # Starting capital (₹)
    'LOT_SIZE'           : 50,         # NIFTY lot size
    'MAX_LOTS'           : 2,          # Max lots per trade

    # HMM/GMM
    'HMM_N_REGIMES'      : 3,          # Number of market regimes
    'HMM_TRAIN_WINDOW'   : 200,        # Training window (trading days)
    'HMM_REFIT_INTERVAL' : 63,         # Re-fit every N days

    # Regime entry thresholds
    'REGIME_IC_CONFIDENCE'   : 0.55,   # Iron Condor min confidence
    'REGIME_CPS_CONFIDENCE'  : 0.50,   # Credit Put Spread min confidence
    'REGIME_BPS_HIGHVOL_SKIP': 0.60,   # Bull Put Spread skip threshold
    ...
}
```

---



