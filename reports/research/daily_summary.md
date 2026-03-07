# Daily Research Summary
## Trading Day: 2026-03-03
## Generated: 2026-03-06

---

## PIPELINE STATUS

| Module | Status | Key Metric |
|--------|--------|------------|
| Shadow Replay | OK | Best PF=99.000 (3840 configs) |
| Alpha Heatmap | OK | 2321 trades, Filter PASS PF=1.22 |
| Regime Filter | OK | PF 0.73->1.3, NFV=$+546,837 |

---

## HIGHLIGHTS

### Shadow Replay
- **Best config**: hold=120s, trail_start=0.10%, trail_offset=0.05%, spread=1.0%, cap=40
- **PF**: 99.000 | **WR**: 80.0% | **Trades**: 40
- **Signals evaluated**: 2323

### Regime Filter
- **Baseline**: 2321 trades, PF=0.73, WR=33.7%
- **Filtered**: 322 trades, PF=1.3, WR=42.5%
- **Trade reduction**: 86%
- **Net filter value**: $+546,837

### Alpha Heatmap
- **Trades analyzed**: 2321 across 11 symbols
- **Filter PASS**: n=522, PF=1.22
- **Filter FAIL**: n=1799, PF=0.63

---

## OUTPUT FILES

| Module | Directory |
|--------|-----------|
| Shadow Replay | `reports/research/replay/2026-03-03/` |
| Alpha Heatmap | `reports/research/alpha_heatmap/2026-03-03/` |
| Regime Filter | `reports/research/regime_paper_validation/2026-03-03/` |
| Scorecard | `reports/research/regime_paper_validation/regime_filter_daily_scorecard.md` |
| Dashboard | `reports/research/daily_summary.md` |

---
*Pipeline completed at 2026-03-06 17:51:42*
*All data READ-ONLY. NO production changes.*