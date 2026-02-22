# AI_Bot_Research
## Morpheus AI — Microstructure Pressure & Arbitrage Index (MPAI)

Research engine for detecting institutional pressure in equity microstructure data to improve momentum trade entry timing.

---

## Purpose

Morpheus trades momentum. By the time it detects ignition, some of the move is already gone. This research project investigates whether we can detect the **pressure building BEFORE momentum ignites** — and whether that detection produces measurably better entries (larger MFE, smaller MAE, better R:R).

MPAI is **not** a standalone trading system. It is an early warning pressure detection layer designed to make Morpheus faster and more confident.

```
Current:   Pressure builds → Price moves → Momentum develops → Morpheus detects → ENTRY
Goal:      Pressure builds → MPAI DETECTS → Morpheus confirms → EARLIER ENTRY
```

---

## Architecture

```
TRADING PC (Bob1) — NEVER TOUCHED BY RESEARCH
├── IBKR Morpheus (live execution)
├── Databento Live Feed
├── Historical Cache: D:\AI_BOT_DATA\databento_cache\
├── Enriched Replays: D:\AI_BOT_DATA\replays\
└── CPU/memory/processes dedicated to trading ONLY

RESEARCH PC (network mount Z:\) — THIS REPO
├── mrl/                  ← Research engine
├── docs/                 ← Whitepaper, findings, notes
├── data/                 ← Local data artifacts (gitignored)
├── results/              ← Output reports and analysis
└── tickers.txt           ← Active research symbols
```

**Research is invisible to the bots.** Zero write access to trading systems. Zero impact on trading performance. Only validated discoveries move to production.

---

## Research Status

### Validated
| ID | Hypothesis | Result |
|----|-----------|--------|
| HYP-003 | FADE pressure spikes in HIGH vol at 30s bars | **56.1% hit rate, 2.22 R:R** (n=435, 60s horizon) |

### Falsified
| ID | Hypothesis | Result |
|----|-----------|--------|
| HYP-001 | Raw pressure predicts continuation | 47-49% hit rate across all conditions |
| HYP-002 | DPI alignment improves accuracy | Adds noise, no improvement |

### Priority Queue (Untested)
| Priority | ID | Hypothesis |
|----------|----|-----------|
| **CRITICAL** | HYP-013 | Pressure buildup precedes Morpheus ignition |
| **CRITICAL** | HYP-018 | Trades with pressure precursor have better MFE/MAE |
| HIGH | HYP-008 | Spread widening amplifies fade signal |
| HIGH | HYP-017 | RSI + pressure agreement improves entry timing |

See [MPAI_whitepaper.md](docs/MPAI_whitepaper.md) for full hypothesis registry (22 hypotheses).

---

## Data Sources

| Source | Type | Location | Access |
|--------|------|----------|--------|
| Databento XNAS.ITCH | Trade-level .dbn.zst | `Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades\` | Read-only network mount |
| Morpheus Enriched | JSONL (spread_dynamics, absorption, l2_pressure, nofi, momentum_score) | `Z:\AI_BOT_DATA\replays\enriched_*.jsonl` | Read-only network mount |
| FINRA Short Volume | Daily CSV | TBD | Free download |

---

## Research Engine (MRL)

Located in `mrl/main.py`. Processes Databento trade files into time-aggregated bars with:

- Aggressor flow pressure (buy vs sell initiated volume)
- Rolling z-score normalization
- Volatility regime classification (LOW/MID/HIGH)
- Trend state detection (UP/DOWN/FLAT)
- Forward return computation at multiple horizons
- MAE/MFE analysis
- Morpheus enriched data merge (spread_dynamics, absorption, etc.)
- LSI (Liquidity Shock Index) flag computation

### Usage
```bash
cd C:\AI_Bot_Research
python mrl/main.py
```

### Configuration
Edit top of `mrl/main.py`:
```python
DATA_ROOT = Path(r"Z:\AI_BOT_DATA\databento_cache\XNAS.ITCH\trades")
BAR_SIZES = [30]           # seconds
PRESSURE_THRESHOLDS = [2.0] # z-score
FORWARD_HORIZONS = {"60s": 60, "180s": 180}
```

Ticker list: `C:\AI_Bot_Research\tickers.txt`

---

## Governing Principle

**Validate and correlate BEFORE implementing.**

```
IDEA → FORMALIZE → TEST AGAINST DATA → CORRELATE → VALIDATE OUT-OF-SAMPLE → IMPLEMENT
```

- n > 500 minimum for any claim
- No execution logic from untested hypotheses
- No adding signals to Morpheus based on theory alone
- Bootstrap confidence intervals and permutation tests required
- Every validation gate must pass before advancing

---

## Project Documents

| Document | Description |
|----------|------------|
| [MPAI_whitepaper.md](docs/MPAI_whitepaper.md) | Living brainstorm & research whitepaper (V3.0) |
| [tickers.txt](tickers.txt) | Active research symbols |

---

## Requirements

```
python >= 3.10
databento
pandas
numpy
```

---

## License

Private research project. Not for distribution.
