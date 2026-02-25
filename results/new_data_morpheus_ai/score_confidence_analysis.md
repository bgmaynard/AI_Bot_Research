# Score/Confidence Analysis — 2026-02-20

Total ignition evaluations with scores: 20694

---

## Score vs Confidence Analysis

Total pairs: 20694

Quadrant analysis (score threshold=60, confidence threshold=0.557):

  Q1 (PASS zone: score>=60 & conf>=0.557):     968  (4.7%)
  Q2 (High score, low conf):                   2153  (10.4%)
  Q3 (Low score, high conf):                  11211  (54.2%)
  Q4 (FAIL zone: score<60 & conf<0.557):       6362  (30.7%)

  Correlation coefficient: -0.112

---

## Score Bucket Analysis

   Score Range    Count    Avg Conf    Min Conf    Max Conf    Avg NOFI
  ------------  -------  ----------  ----------  ----------  ----------
    0- 10             1       0.320       0.320       0.320     -1.0000
   10- 20            41       0.480       0.151       0.743     -0.9529
   20- 30          1129       0.544       0.167       0.893     -0.8501
   30- 40          3192       0.561       0.030       1.000     -0.4596
   40- 50          7203       0.676       0.030       1.000     -0.0636
   50- 60          6007       0.608       0.016       1.000      0.4597
   60- 70          2641       0.494       0.030       1.000      0.8942
   70- 80           476       0.383       0.130       0.702      0.9984
   80- 90             4       0.514       0.514       0.514      1.0000
   90-100                        --          --          --          --

---

## Threshold Sensitivity Analysis

What-if: signals that would pass ignition at various thresholds

    Score Thresh    Conf Thresh    Would Pass    % of Total
  --------------  -------------  ------------  ------------
              60          0.557           677         3.27% <-- current
              60          0.500           851         4.11%
              60          0.400          1403         6.78%
              60          0.300          1623         7.84%
              60          0.200          1816         8.78%
              55          0.557          1626         7.86%
              55          0.500          1859         8.98%
              55          0.400          2477        11.97%
              55          0.300          2872        13.88%
              55          0.200          3139        15.17%
              50          0.557          2650        12.81%
              50          0.500          2901        14.02%
              50          0.400          3605        17.42%
              50          0.300          4054        19.59%
              50          0.200          4359        21.06%
              45          0.557          3063        14.80%
              45          0.500          3321        16.05%
              45          0.400          4027        19.46%
              45          0.300          4542        21.95%
              45          0.200          4847        23.42%
              40          0.557          3114        15.05%
              40          0.500          3372        16.29%
              40          0.400          4121        19.91%
              40          0.300          4636        22.40%
              40          0.200          4941        23.88%

---

## Data Density vs Confidence

Confidence quartiles: Q25=0.399  Q50=0.674  Q75=0.785

  Low conf (n=5207): avg_score=49.4 min=7.8 max=79.2
  Mid conf (n=10349): avg_score=48.7 min=16.4 max=82.8
  High conf (n=5138): avg_score=46.9 min=24.1 max=68.4

### EQUS.MINI Limitation Note
EQUS.MINI provides only 1-level book depth (mbp-1) and trades.
Confidence is derived from data density (ticks per window).
Low-volume names may have high scores but low confidence due to
sparse tick flow, while high-volume names have high confidence
but average scores — creating a structural mismatch.

---

## Recommendations

Current thresholds (score>=60, conf>=0.557): 968 pass (4.68%)
Lower confidence to 0.3 (score>=60):         2579 pass (12.46%)
Lower score to 50 (conf>=0.557):             4732 pass (22.87%)
