# Optimal Entry Window — 2026-03-03

**Generated:** 2026-03-04T05:05:48Z
**Mode:** READ-ONLY RESEARCH (SuperBot Engine)

---

## Recommendation

| Parameter | Value |
|-----------|-------|
| Optimal entry start | 5s after ignition |
| Optimal entry end | 305s after ignition |
| Current Morpheus median entry | 180s after ignition |
| Recommended offset | -175s vs current 180s median |
| Composite score | 0.5193 |

## Window Comparison

| Window | P(Continue) | Avg Dir Ret | WR | PF | Score |
|--------|-------------|-------------|-----|-----|-------|
| 15s | 42.7% | +0.1164% | 42.7% | 1.357 | 0.4765 |
| 30s | 45.0% | +0.1749% | 45.0% | 1.405 | 0.4986 |
| 60s | 43.1% | +0.1253% | 43.1% | 1.192 | 0.4570 |
| 90s | 44.6% | +0.2187% | 44.6% | 1.307 | 0.4909 |
| 120s | 45.6% | +0.2790% | 45.6% | 1.359 | 0.5099 |
| 180s | 46.3% | +0.1992% | 46.3% | 1.203 | 0.4792 |
| 300s | 49.0% | +0.3459% | 49.0% | 1.278 | 0.5193 **<-- OPTIMAL** |

## Profitable Zone

Momentum continuation produces positive edge (PF >= 1.0) between **5s** and **305s** after ignition.

Morpheus currently enters at ~180s median, which is within this zone.

## Conclusion

Moving entry 175s earlier (from 180s to 5s after ignition) would capture more momentum continuation edge.

---

*This study is research-only. No production changes applied.*