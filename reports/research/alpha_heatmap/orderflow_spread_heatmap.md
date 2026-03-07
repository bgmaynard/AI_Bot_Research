# Order Flow vs Spread Alpha Heatmap — Multi-Symbol Validation
## Data: 2026-03-03 | 2321 simulated trades | 11 symbols
## Generated: 2026-03-06

---

## 1. Order Flow Marginal Performance

| OFI | Trades | Win Rate | PF | Avg PnL | Total PnL |
|-----|--------|----------|-----|---------|-----------|
| **weak** | 389 | 25.4% | 0.27 | $-569 | $-221,460 |
| **moderate** | 1687 | 35.3% | 0.81 | $-157 | $-265,357 |
| **strong** | 240 | 35.4% | 0.94 | $-20 | $-4,874 |

## 2. OFI x Spread — Profit Factor

### 

| OFI \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% |
|---|---|---|---|---|
| **weak** | PF=0.12 (n=105) | PF=0.17 (n=115) | PF=0.74 (n=129) | PF=0.0 (n=40) |
| **moderate** | PF=0.93 (n=449) | PF=0.8 (n=625) | PF=0.82 (n=520) | PF=0.25 (n=89) |
| **strong** | PF=0.55 (n=132) | PF=2.29 (n=71) | PF=0.02 (n=27) | PF=0.0 (n=9) |

## 3. OFI x Spread — Avg PnL

### 

| OFI \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% |
|---|---|---|---|---|
| **weak** | $-418 (n=105) | $-727 (n=115) | $-167 (n=129) | $-1,810 (n=40) |
| **moderate** | $-38 (n=449) | $-135 (n=625) | $-232 (n=520) | $-485 (n=89) |
| **strong** | $-159 (n=132) | $+442 (n=71) | $-291 (n=27) | $-828 (n=9) |

## 4. Volatility x OFI — Profit Factor

### 

| Volatility \\ OFI | weak | moderate | strong |
|---|---|---|---|
| **low** | PF=0.43 (n=207) | PF=0.6 (n=949) | PF=1.14 (n=158) |
| **medium** | PF=0.13 (n=173) | PF=1.21 (n=720) | PF=0.04 (n=73) |
| **high** | PF=0.0 (n=9) | PF=63.52 (n=18) | PF=442.67 (n=9) |

## 5. Volatility x OFI — Avg PnL

### 

| Volatility \\ OFI | weak | moderate | strong |
|---|---|---|---|
| **low** | $-431 (n=207) | $-405 (n=949) | $+36 (n=158) |
| **medium** | $-696 (n=173) | $+131 (n=720) | $-598 (n=73) |
| **high** | $-1,308 (n=9) | $+1,400 (n=18) | $+3,681 (n=9) |

---

*Data source: live_signals.json, *_quotes.json (READ-ONLY)*
