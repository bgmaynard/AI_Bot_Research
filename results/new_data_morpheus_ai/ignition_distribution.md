# Ignition Score Distribution — 2026-02-20

Signal ledger entries: 73023
Ignition evaluations: 21747
Gating blocks: 23668

---

## Score Histogram

Total with scores: 20694 / 21747 entries

    0- 10:      1  
   10- 20:     41  
   20- 30:   1129  #######
   30- 40:   3192  ######################
   40- 50:   7203  ##################################################
   50- 60:   6007  #########################################
   60- 70:   2641  ##################
   70- 80:    476  ###
   80- 90:      4  
   90-100:      0  

  Mean: 48.4  Median: 48.4  Min: 7.8  Max: 82.8

---

## Per-Check Failure Counts

  DAILY_LOSS_LIMIT                 19181
  LOW_SCORE                        17573
  LOW_NOFI                         12609
  LOW_CONFIDENCE                    8515
  HIGH_SPREAD                       7497
  NEGATIVE_L2_PRESSURE              4792
  NO_MOMENTUM_DATA                  1053
  CONFLICTING_SIGNALS                713
  DECLINING_SCORE                    388
  RTH_COOLDOWN                       156

---

## Near-Miss Analysis (Score 50-59)

Count: 6007
Top symbols: CDIO(1347), OPEN(552), RXT(546), CORD(363), KOS(333), EVTV(287), KNRX(263), SATL(259), NAMM(256), SOFI(225)
Avg confidence: 0.608
Confidence range: 0.016 - 1.000

---

## Confidence Distribution

Total with confidence: 20694 / 21747 entries

  0.0-0.1:    924  ########
  0.1-0.2:    720  ######
  0.2-0.3:    711  ######
  0.3-0.4:   2852  ##########################
  0.4-0.5:   2235  ####################
  0.5-0.6:   1628  ###############
  0.6-0.7:   1743  ################
  0.7-0.8:   5398  ##################################################
  0.8-0.9:   2210  ####################
  0.9-1.0:   2273  #####################

  Mean: 0.601  Median: 0.674  Min: 0.016  Max: 1.000

---

## Overlap Analysis (Score >= 60 AND Confidence >= 0.557)

Score >= 60 only:      3121
Confidence >= 0.557:   12179
Both (overlap):        968

Overlap symbols: KOS(154), EVTV(121), RXT(106), MARA(95), OPEN(79), INDI(75), CDIO(72), SATL(71), BATL(38), SOFI(35)
Avg score: 62.9  Avg confidence: 0.714

---

## After Daily-Loss-Fix Simulation

Total ignition evaluations: 21747
Currently passing: 1
Currently failing: 21746

Entries with DAILY_LOSS_LIMIT failure: 19181
Entries where DAILY_LOSS_LIMIT was the ONLY failure: 671

After fix: these 671 entries would newly pass ignition
Entries with DAILY_LOSS_LIMIT + other failures: 18510 (still blocked)

New estimated pass count: 672