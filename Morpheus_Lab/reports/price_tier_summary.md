# Price Tier & Friction Survivability Analysis
## Strategy: flush_reclaim_v1

### Dataset
- Total trades: 494
- Symbols: ANL, BOXL, CISS, BATL, ELPW
- Price range: $1.05 - $11.36 (avg $2.85)

### Trade Distribution by Price Tier

| Tier | Trades |
|------|--------|
| $1-$3 | 359 |
| $3-$5 | 103 |
| $5-$7 | 5 |
| $7-$10 | 21 |
| $10-$20 | 6 |
| $20+ | 0 |

### Scenario: Ideal (frictionless)
- Friction/share: $0.0000
- Friction/trade: $0.00

| Tier | Trades | Gross WR | Net WR | Gross PF | Net PF | AvgW $/sh | AvgL $/sh | Flip% | EBR | Verdict |
|------|--------|----------|--------|----------|--------|-----------|-----------|-------|-----|---------|
| $1-$3 | 359 | 51.8% | 51.8% | 1.56 | 1.56 | $0.042 | $-0.029 | 0% | inf | ROBUST (frictionless) |
| $3-$5 | 103 | 52.4% | 52.4% | 1.88 | 1.88 | $0.073 | $-0.043 | 0% | inf | ROBUST (frictionless) |
| $5-$7 | 5 | 40.0% | 40.0% | 0.67 | 0.67 | $0.110 | $-0.110 | 0% | inf | NO EDGE (even frictionless) |
| $7-$10 | 21 | 52.4% | 52.4% | 2.13 | 2.13 | $0.277 | $-0.143 | 0% | inf | ROBUST (frictionless) |
| $10-$20 | 6 | 33.3% | 33.3% | 0.66 | 0.66 | $0.175 | $-0.133 | 0% | inf | NO EDGE (even frictionless) |
| $20+ | 0 | -- | -- | -- | -- | -- | -- | -- | -- | NO DATA |

### Scenario: Realistic (commission-free broker)
- Friction/share: $0.0300
- Friction/trade: $3.00

| Tier | Trades | Gross WR | Net WR | Gross PF | Net PF | AvgW $/sh | AvgL $/sh | Flip% | EBR | Verdict |
|------|--------|----------|--------|----------|--------|-----------|-----------|-------|-----|---------|
| $1-$3 | 359 | 51.8% | 39.6% | 1.56 | 0.25 | $0.042 | $-0.029 | 24% | 1.4 | FRAGILE |
| $3-$5 | 103 | 52.4% | 52.4% | 1.88 | 0.65 | $0.073 | $-0.043 | 0% | 2.4 | VIABLE |
| $5-$7 | 5 | 40.0% | 40.0% | 0.67 | 0.38 | $0.110 | $-0.110 | 0% | 3.7 | ROBUST |
| $7-$10 | 21 | 52.4% | 52.4% | 2.13 | 1.57 | $0.277 | $-0.143 | 0% | 9.2 | ROBUST |
| $10-$20 | 6 | 33.3% | 33.3% | 0.66 | 0.45 | $0.175 | $-0.133 | 0% | 5.8 | ROBUST |
| $20+ | 0 | -- | -- | -- | -- | -- | -- | -- | -- | NO DATA |

### Scenario: Custom (slip=1t, lat=0t, spread=$0.0050, comm=$0.00)
- Friction/share: $0.0300
- Friction/trade: $3.00

| Tier | Trades | Gross WR | Net WR | Gross PF | Net PF | AvgW $/sh | AvgL $/sh | Flip% | EBR | Verdict |
|------|--------|----------|--------|----------|--------|-----------|-----------|-------|-----|---------|
| $1-$3 | 359 | 51.8% | 39.6% | 1.56 | 0.25 | $0.042 | $-0.029 | 24% | 1.4 | FRAGILE |
| $3-$5 | 103 | 52.4% | 52.4% | 1.88 | 0.65 | $0.073 | $-0.043 | 0% | 2.4 | VIABLE |
| $5-$7 | 5 | 40.0% | 40.0% | 0.67 | 0.38 | $0.110 | $-0.110 | 0% | 3.7 | ROBUST |
| $7-$10 | 21 | 52.4% | 52.4% | 2.13 | 1.57 | $0.277 | $-0.143 | 0% | 9.2 | ROBUST |
| $10-$20 | 6 | 33.3% | 33.3% | 0.66 | 0.45 | $0.175 | $-0.133 | 0% | 5.8 | ROBUST |
| $20+ | 0 | -- | -- | -- | -- | -- | -- | -- | -- | NO DATA |

### Scenario: Conservative (worst-case retail)
- Friction/share: $0.0650
- Friction/trade: $6.50

| Tier | Trades | Gross WR | Net WR | Gross PF | Net PF | AvgW $/sh | AvgL $/sh | Flip% | EBR | Verdict |
|------|--------|----------|--------|----------|--------|-----------|-----------|-------|-----|---------|
| $1-$3 | 359 | 51.8% | 4.5% | 1.56 | 0.02 | $0.042 | $-0.029 | 91% | 0.7 | DEAD |
| $3-$5 | 103 | 52.4% | 24.3% | 1.88 | 0.15 | $0.073 | $-0.043 | 54% | 1.1 | FRAGILE |
| $5-$7 | 5 | 40.0% | 20.0% | 0.67 | 0.19 | $0.110 | $-0.110 | 50% | 1.7 | FRAGILE |
| $7-$10 | 21 | 52.4% | 52.4% | 2.13 | 1.12 | $0.277 | $-0.143 | 0% | 4.3 | ROBUST |
| $10-$20 | 6 | 33.3% | 33.3% | 0.66 | 0.28 | $0.175 | $-0.133 | 0% | 2.7 | VIABLE |
| $20+ | 0 | -- | -- | -- | -- | -- | -- | -- | -- | NO DATA |

### Recommendation

- Scenario used: realistic
- Criteria: Net PF >= 1.2, Avg PnL > 0, EBR >= 2.0

**Recommended min-price: $7.00** (Tier: $7-$10, Confidence: HIGH)

### Tier Verdicts

| Tier | Net PF | Net Avg PnL | EBR | Verdict |
|------|--------|------------|-----|---------|
| $1-$3 | 0.25 | $-2.21 | 1.4 | FRAGILE |
| $3-$5 | 0.65 | $-1.22 | 2.4 | VIABLE |
| $5-$7 | 0.38 | $-5.20 | 3.7 | ROBUST |
| $7-$10 | 1.57 | $+4.69 | 9.2 | ROBUST <-- recommended |
| $10-$20 | 0.45 | $-6.00 | 5.8 | ROBUST |
| $20+ | 0.00 | $+0.00 | 0.0 | NO DATA |

### Why Low-Priced Stocks Collapse Under Friction

The flush_reclaim_v1 strategy detects a genuine microstructure pattern, 
but on sub-$5 stocks the average winning move is typically $0.03-$0.05/share. 
A single tick of slippage ($0.01) on entry AND exit consumes 40-66% of the average winner. 
Add spread and commission, and the edge is entirely consumed by execution costs.

On $5+ stocks, winning moves average $0.15-$0.35/share -- 1 tick of slippage is 
only 6-13% of the winner, leaving substantial room for the edge to survive.

### Risk of Ignoring This Filter

Deploying flush_reclaim_v1 without a min-price filter will result in 
systematic losses that scale linearly with trade frequency. The more trades 
the system takes on low-priced stocks, the faster capital erodes. 
This is not a "might lose" scenario -- the math is deterministic: 
when friction exceeds edge, every trade is expected-value negative.