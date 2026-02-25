# Funnel Before/After Comparison — 2026-02-20

## Summary

Simulates the impact of two fixes on the signal funnel:

1. **DAILY_LOSS_LIMIT units fix** (P0): converts config ratio (0.02) to percentage (2.0%)
2. **Extension momentum override** (P1): strong momentum downgrades hard veto

---

## Ignition Gate Funnel

```
                                    Before       After       Delta
------------------------------  ----------  ----------  ----------
Signals scored                       23669       23669          --
Ignition evaluations                 21747       21747          --
Ignition PASSED                          1         672        +671
Ignition FAILED                      21746       21075        -671
```

## Per-Check Failure Counts

```
Check                               Before       After       Delta
------------------------------  ----------  ----------  ----------
CONFLICTING_SIGNALS                    713         713          +0
DAILY_LOSS_LIMIT                     19181           0      -19181 <-- FIXED
DECLINING_SCORE                        388         388          +0
HIGH_SPREAD                           7497        7497          +0
LOW_CONFIDENCE                        8515        8515          +0
LOW_NOFI                             12609       12609          +0
LOW_SCORE                            17573       17573          +0
NEGATIVE_L2_PRESSURE                  4792        4792          +0
NO_MOMENTUM_DATA                      1053        1053          +0
RTH_COOLDOWN                           156         156          +0
```

## Daily Loss Limit Fix Detail

- DAILY_LOSS_LIMIT blocks (before): **19181**
- DAILY_LOSS_LIMIT as ONLY failure: **671**
- DAILY_LOSS_LIMIT blocks (after fix): **0**
- Signals newly passing: **671**

### Newly Passing Signals (sample, max 20)

  Symbol     Score    Confidence  Original Failures
--------  --------  ------------  ----------------------------------------
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    BJDX      65.2         0.624  DAILY_LOSS_LIMIT: -0.1311 < -0.02
    EVTV      60.8         0.789  DAILY_LOSS_LIMIT: -0.1311 < -0.02
  ... and 651 more

## Extension Gate Impact

- Extension VETOs (before): **1922**
- Extension SCORE_REDUCED (before): **0**
- Note: Momentum override impact requires momentum data at extension evaluation time,
  which will be available in future sessions with the pipeline changes deployed.

## Key Takeaway

The daily loss limit fix alone would have unblocked **671** additional signals.
Combined with the remaining filters, **672** signals would have passed ignition.