# Volatility vs Spread Alpha Heatmap — Multi-Symbol Validation
## Data: 2026-03-03 | 2321 simulated trades | 11 symbols
## Generated: 2026-03-06

---

## Methodology

Expanded from 20 BATL paper trades to **all ignition-passed signals across
11 symbols**. Trade outcomes simulated using tick-level quote data with
production parameters (trail=1.0%, max_hold=300s, size=5000 shares).

---

## 1. Volatility Marginal Performance

| Volatility | Trades | Win Rate | PF | Avg PnL | Total PnL | Avg MFE | Avg MAE |
|------------|--------|----------|-----|---------|-----------|---------|---------|
| **low** | 1316 | 34.2% | 0.59 | $-356 | $-468,215 | 1.057% | 0.905% |
| **medium** | 968 | 32.5% | 0.89 | $-72 | $-70,047 | 1.250% | 0.716% |
| **high** | 37 | 45.9% | 4.32 | $+1,222 | $+45,197 | 2.700% | 0.591% |

## 2. Spread Marginal Performance

| Spread | Trades | Win Rate | PF | Avg PnL | Total PnL |
|--------|--------|----------|-----|---------|-----------|
| **<0.3%** | 689 | 37.3% | 0.75 | $-121 | $-83,667 |
| **0.3-0.6%** | 813 | 32.3% | 0.75 | $-168 | $-136,527 |
| **0.6-1.0%** | 676 | 33.7% | 0.8 | $-222 | $-149,921 |
| **>1.0%** | 138 | 23.2% | 0.11 | $-892 | $-123,050 |

## 3. Volatility x Spread Heatmap — Profit Factor

### 

| Volatility \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% |
|---|---|---|---|---|
| **low** | PF=0.47 (n=420) | PF=0.6 (n=479) | PF=0.64 (n=384) | PF=0.57 (n=32) |
| **medium** | PF=1.17 (n=251) | PF=0.88 (n=324) | PF=1.24 (n=283) | PF=0.04 (n=106) |
| **high** | PF=1.95 (n=18) | PF=120.73 (n=10) | PF=0.0 (n=9) | - |

## 4. Volatility x Spread — Average PnL

### 

| Volatility \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% |
|---|---|---|---|---|
| **low** | $-272 (n=420) | $-304 (n=479) | $-524 (n=384) | $-238 (n=32) |
| **medium** | $+72 (n=251) | $-74 (n=324) | $+181 (n=283) | $-1,089 (n=106) |
| **high** | $+693 (n=18) | $+3,292 (n=10) | $-23 (n=9) | - |

## 5. Volatility x Spread — Win Rate

### 

| Volatility \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% |
|---|---|---|---|---|
| **low** | 35.7% (n=420) | 27.1% (n=479) | 39.8% (n=384) | 50.0% (n=32) |
| **medium** | 39.0% (n=251) | 38.6% (n=324) | 26.5% (n=283) | 15.1% (n=106) |
| **high** | 50.0% (n=18) | 80.0% (n=10) | 0.0% (n=9) | - |

---

## 6. Regime x Volatility — Profit Factor

### 

| Regime \\ Volatility | low | medium | high |
|---|---|---|---|
| **LOW_VOLATILITY** | PF=0.54 (n=1080) | PF=0.67 (n=384) | PF=1.91 (n=21) |
| **RANGE_BOUND** | PF=0.84 (n=236) | PF=1.01 (n=584) | PF=166.0 (n=16) |

---

*Data source: live_signals.json, *_quotes.json (READ-ONLY)*
