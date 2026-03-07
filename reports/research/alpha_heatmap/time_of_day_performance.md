# Time-of-Day Performance — Multi-Symbol Validation
## Data: 2026-03-03 | 2321 simulated trades | 11 symbols
## Generated: 2026-03-06

---

## 1. Session Segment Performance

| Segment | Trades | Win Rate | PF | Avg PnL | Total PnL | Avg MFE | Avg MAE |
|---------|--------|----------|-----|---------|-----------|---------|---------|
| **premarket** | 1170 | 35.9% | 0.81 | $-171 | $-200,610 | 1.380% | 0.900% |
| **open** | 203 | 29.6% | 0.56 | $-330 | $-67,009 | 1.036% | 0.766% |
| **midday** | 829 | 33.9% | 0.73 | $-163 | $-134,980 | 0.970% | 0.714% |
| **power_hour** | 119 | 17.6% | 0.11 | $-760 | $-90,466 | 0.611% | 0.882% |

## 2. Session x Volatility — Profit Factor

### 

| Session \\ Volatility | low | medium | high |
|---|---|---|---|
| **premarket** | PF=0.53 (n=684) | PF=1.47 (n=459) | PF=37.88 (n=27) |
| **open** | PF=1.89 (n=68) | PF=0.37 (n=126) | PF=0.0 (n=9) |
| **midday** | PF=0.74 (n=503) | PF=0.71 (n=325) | PF=0.0 (n=1) |
| **power_hour** | PF=0.11 (n=61) | PF=0.11 (n=58) | - |

## 3. Session x Spread — Profit Factor

### 

| Session \\ Spread | <0.3% | 0.3-0.6% | 0.6-1.0% | >1.0% |
|---|---|---|---|---|
| **premarket** | PF=0.45 (n=262) | PF=1.12 (n=376) | PF=0.82 (n=423) | PF=0.2 (n=104) |
| **open** | PF=0.53 (n=101) | PF=1.1 (n=61) | PF=0.42 (n=33) | PF=0.0 (n=8) |
| **midday** | PF=1.16 (n=301) | PF=0.38 (n=336) | PF=1.16 (n=174) | PF=0.0 (n=18) |
| **power_hour** | PF=0.13 (n=25) | PF=0.19 (n=40) | PF=0.09 (n=46) | PF=0.0 (n=8) |

## 4. Per-Symbol Session Performance

| Symbol | Premarket | Open | Midday | Power Hour |
|--------|-----------|------|--------|------------|
| BATL | PF=0.82 n=984 | PF=0.23 n=120 | PF=0.81 n=552 | PF=0.07 n=88 |
| CRCD | PF=INF n=1 | PF=0.68 n=3 | PF=0.5 n=17 | - |
| DUST | PF=0.25 n=2 | - | PF=INF n=1 | - |
| IONZ | PF=0.0 n=7 | PF=2.12 n=49 | PF=0.33 n=161 | PF=0.61 n=21 |
| MSTZ | PF=0.5 n=17 | PF=1.82 n=7 | PF=1.39 n=21 | PF=3.74 n=3 |
| PLUG | PF=1.5 n=2 | PF=0.0 n=1 | - | - |
| SOXS | PF=INF n=1 | - | - | - |
| TMDE | PF=0.63 n=115 | PF=0.7 n=7 | PF=0.61 n=32 | PF=0.14 n=2 |
| USEG | PF=0.39 n=7 | - | PF=0.0 n=1 | - |
| UVIX | PF=0.46 n=30 | PF=11.79 n=8 | PF=0.36 n=33 | PF=0.92 n=3 |
| VG | PF=0.15 n=4 | PF=0.68 n=8 | PF=0.3 n=11 | PF=0.12 n=2 |

---

*Data source: live_signals.json, *_quotes.json (READ-ONLY)*
