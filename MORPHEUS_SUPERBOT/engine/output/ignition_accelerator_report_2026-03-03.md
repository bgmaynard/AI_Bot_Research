# Ignition Entry Accelerator Report - 2026-03-03

**Generated:** 2026-03-04T05:13:43Z
**Mode:** READ-ONLY RESEARCH (SuperBot Engine)
**Method:** Limit-order replay at ignition events with exit model simulation

---

## Study Overview

| Parameter | Value |
|-----------|-------|
| Ignition events tested | 180 |
| Offsets tested | +0.00%, -0.20%, -0.50% |
| Fill window | 30s |
| Exit model | stop=-1.0%, trail=0.8%/0.4%, time=300s |
| Detector lockout | 120s |

## Offset Comparison

| Offset | N Total | Filled | Fill Rate | WR | Avg Return | PF | Avg Slippage | Worst DD | Avg Fill Delay |
|--------|---------|--------|-----------|-----|------------|-----|-------------|----------|----------------|
| +0.00% | 180 | 160 | 88.9% | 44.4% | -0.0448% | 0.928 | -0.0103% | -2.3452% | 3.8s |
| -0.20% | 180 | 119 | 66.1% | 39.5% | -0.1782% | 0.744 | -0.0014% | -2.3452% | 8.6s |
| -0.50% | 180 | 83 | 46.1% | 41.0% | -0.1656% | 0.782 | -0.0515% | -2.8545% | 11.7s |

## Exit Type Distribution

| Offset | STOP | TRAIL | TIME | EOD |
|--------|------|-------|------|-----|
| +0.00% | 51.9% (83) | 43.1% (69) | 5.0% (8) | 0.0% (0) |
| -0.20% | 57.1% (68) | 39.5% (47) | 3.4% (4) | 0.0% (0) |
| -0.50% | 57.8% (48) | 42.2% (35) | 0.0% (0) | 0.0% (0) |

## Per-Symbol Breakdown (by ignition count)

### Offset +0.00%

| Symbol | Ignitions | Filled | Fill Rate | WR | Avg Return | PF |
|--------|-----------|--------|-----------|-----|------------|-----|
| BATL | 140 | 124 | 88.6% | 41.9% | -0.0656% | 0.903 |
| TMDE | 32 | 28 | 87.5% | 60.7% | +0.0954% | 1.209 |
| SOXS | 4 | 4 | 100.0% | 50.0% | +0.1204% | 1.971 |
| UVIX | 2 | 2 | 100.0% | 0.0% | -0.6341% | 0.000 |
| MSTZ | 1 | 1 | 100.0% | 0.0% | -0.4880% | 0.000 |
| PLUG | 1 | 1 | 100.0% | 0.0% | -0.4396% | 0.000 |

### Offset -0.20%

| Symbol | Ignitions | Filled | Fill Rate | WR | Avg Return | PF |
|--------|-----------|--------|-----------|-----|------------|-----|
| BATL | 140 | 96 | 68.6% | 36.5% | -0.2610% | 0.655 |
| TMDE | 32 | 18 | 56.2% | 61.1% | +0.2697% | 1.574 |
| SOXS | 4 | 1 | 25.0% | 100.0% | +0.9828% | 999.000 |
| UVIX | 2 | 2 | 100.0% | 0.0% | -0.5736% | 0.000 |
| MSTZ | 1 | 1 | 100.0% | 0.0% | -0.4063% | 0.000 |
| PLUG | 1 | 1 | 100.0% | 0.0% | -0.4396% | 0.000 |

### Offset -0.50%

| Symbol | Ignitions | Filled | Fill Rate | WR | Avg Return | PF |
|--------|-----------|--------|-----------|-----|------------|-----|
| BATL | 140 | 69 | 49.3% | 39.1% | -0.2002% | 0.745 |
| TMDE | 32 | 13 | 40.6% | 46.2% | -0.0322% | 0.952 |
| SOXS | 4 | 0 | 0.0% | - | - | - |
| UVIX | 2 | 0 | 0.0% | - | - | - |
| MSTZ | 1 | 1 | 100.0% | 100.0% | +0.4856% | 999.000 |
| PLUG | 1 | 0 | 0.0% | - | - | - |

## Recommended Configuration

- **Recommended offset:** +0.00%
- **Expected fill rate:** 88.9%
- **Expected WR:** 44.4%
- **Expected PF:** 0.928
- **Average fill delay:** 3.8s after ignition
- **Spread constraint (PM):** < 0.9%
- **Spread constraint (RTH):** < 0.6%
- **Lockout:** 120s between ignitions per symbol

---

## Part E: Production Integration Design

### Proposed runtime_config Keys

```json
{
  "ignition_accelerator_enabled": false,
  "ignition_accelerator_offset_pct": 0.0,
  "ignition_accelerator_fill_window_seconds": 30,
  "ignition_accelerator_lockout_seconds": 120,
  "ignition_accelerator_pm_max_spread_pct": 0.9,
  "ignition_accelerator_rth_max_spread_pct": 0.6
}
```

### Integration Path

1. **IgnitionDetector** runs in Morpheus quote processing loop (per-symbol)
2. On IGNITION_DETECTED, accelerator proposes a LIMIT entry at offset price
3. If signal already exists in pipeline (RISK_APPROVED), entry_reference is overridden to proposed_limit
4. If no signal exists yet, candidate is queued — will be consumed if/when signal arrives within fill_window_seconds
5. All existing gates remain active: risk, meta, containment are NOT bypassed
6. Accelerator only improves entry price — it does not create new signals

### Safety Guarantees

- Accelerator is gated by `ignition_accelerator_enabled` (default: false)
- Does NOT bypass risk gate, meta gate, or containment filter
- Does NOT create new trading signals (only adjusts entry price of existing ones)
- Lockout prevents over-trading on repeated ignitions
- Spread constraints prevent entries in illiquid conditions
- LIMIT order means unfavorable fills are impossible (no market order slippage)

---

*This study is research-only. No production changes applied.*