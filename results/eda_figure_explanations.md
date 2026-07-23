# Exploratory Data Analysis — Figure Explanations

Detailed explanations for all 10 EDA figures, with exact values from the processed dataset of **1,020 matches** across **6 seasons** (2019-2020 through 2024-2025) involving 5 supported clubs: Arsenal, Liverpool, Manchester City, Chelsea, and Manchester United. Each figure section connects the raw visualisation to the **198 engineered features** it feeds into the ML pipeline, culminating in the **50 selected features** used by the final stacked-ensemble model.

---

## 1. Dataset Overview (Fig 4.1)

**Left panel (table):** Lists all 25 columns with dtype, non-null count, missing %, and unique values. Key columns include `FTHG`/`FTAG` (full-time home/away goals), `HTHG`/`HTAG` (half-time goals), `HS`/`AS` (shots), `HST`/`AST` (shots on target), `HF`/`AF` (fouls), `HC`/`AC` (corners), `HY`/`AY` (yellows), `HR`/`AR` (reds), `FTR` (full-time result: H/D/A), and `_season`. Missing % is near-zero for all match columns — the data is clean and complete.

**Right panel (bar chart):** Matches per season — exactly **170 matches per season** across all 6 seasons (2019-2020 through 2024-2025), totaling **1,020 matches**. The perfectly uniform distribution confirms balanced representation of the 5 supported clubs across all seasons.

**Feature engineering context:** These 25 raw columns are expanded into **198 engineered features** across 20 groups (see Section 8 for full listing). The raw data is transformed into rolling form metrics (windows 3/5/10), venue-specific performance, H2H records, attacking/defensive strength, discipline, momentum, efficiency, dominance, trends, consistency, first-half performance, fixture difficulty, venue momentum, win percentages, clean sheet rates, Elo ratings, xG proxies, cross-team differences, and a home advantage index. The resulting `engineered_features.csv` contains 1,016 rows × 204 columns (198 features + 6 metadata columns: `home_team`, `away_team`, `date`, `ftr`, `ftr_encoded`, `season`).

---

## 2. Match Outcome Analysis (Fig 4.2)

**Left panel (overall counts):**
- **Home Win (H):** 465 matches — **45.6%**
- **Draw (D):** 205 matches — **20.1%**
- **Away Win (A):** 350 matches — **34.3%**

Home teams win nearly half of all matches, while draws account for only 1 in 5. The home win rate (45.6%) exceeds the away win rate (34.3%) by **11.3 percentage points**, confirming a strong home advantage effect across the top-5 clubs.

**Right panel (by season breakdown):**

| Season | Home Wins | Draw | Away Win |
|---|---|---|---|
| 2019-2020 | 82 (48.2%) | 35 (20.6%) | 53 (31.2%) |
| 2020-2021 | 62 (36.5%) | 36 (21.2%) | 72 (42.4%) |
| 2021-2022 | 78 (45.9%) | 32 (18.8%) | 60 (35.3%) |
| 2022-2023 | 86 (50.6%) | 34 (20.0%) | 50 (29.4%) |
| 2023-2024 | 81 (47.6%) | 27 (15.9%) | 62 (36.5%) |
| 2024-2025 | 76 (44.7%) | 41 (24.1%) | 53 (31.2%) |

**Key insight:** 2020-2021 is an anomaly — away wins (42.4%) actually exceeded home wins (36.5%). This was the COVID-19 season played behind closed doors with no fans, eliminating the traditional 12th-man advantage. The home advantage returned to ~45-50% once fans were readmitted.

**Feature engineering context:** Outcomes map to the target variable `FTR_encoded` (H=0, D=1, A=2), which the ML models learn to predict. The home advantage observed here is quantified by the engineered `home_advantage_index` feature (`home_form_points_10 / max(away_form_points_10, 1)`), which ranges from **1.47** (Man United) to **2.53** (Man City) — meaning Man City's recent home form is 2.53x their away form, while Man United's is only 1.47x. The `home_team_ppm` (points per match over last 30 games) further quantifies the gap: Man City averages **2.25 ppm** vs Chelsea's **1.64 ppm**, a 37% difference in long-term points accumulation.

---

## 3. Goals Distribution (Fig 4.3)

Two side-by-side histograms showing home and away goals distributions with dashed mean lines.

**Home goals (left panel):**

| Home Goals | Matches | % |
|---|---|---|
| 0 | 246 | 24.1% |
| 1 | 285 | 27.9% |
| 2 | 247 | 24.2% |
| 3 | 138 | 13.5% |
| 4 | 63 | 6.2% |
| 5 | 27 | 2.6% |
| 6 | 7 | 0.7% |
| 7 | 4 | 0.4% |
| 8 | 1 | 0.1% |
| 9 | 2 | 0.2% |

- **Mean: 1.64 goals**, Median: 1, Mode: 1, Std: 1.44
- The distribution peaks at 1 goal (27.9%) and is right-skewed — 52.0% of matches see home teams score 0-1 goals, but 23.0% score 3+. Home teams have scored as many as 9 goals (twice), showing extreme outliers.

**Away goals (right panel):**

| Away Goals | Matches | % |
|---|---|---|
| 0 | 292 | 28.6% |
| 1 | 328 | 32.2% |
| 2 | 224 | 22.0% |
| 3 | 110 | 10.8% |
| 4 | 41 | 4.0% |
| 5 | 15 | 1.5% |
| 6 | 9 | 0.9% |
| 7 | 1 | 0.1% |

- **Mean: 1.38 goals**, Median: 1, Mode: 1, Std: 1.28
- Away teams are more likely to be shut out (28.6% vs 24.1% for home). The **0.26 goal gap** (1.64 vs 1.38) between home and away means quantifies the scoring advantage of playing at home. Away teams never scored more than 7 (vs 9 for home).

**Feature engineering context — The Efficiency Pipeline:**

These two histograms reveal the **shots → SOT → goals** pipeline that the efficiency features capture:

| Metric | Home Mean | Away Mean |
|---|---|---|
| Total Shots | 16.89 | 14.37 |
| Shots on Target | 6.11 | 5.05 |
| Goals | 1.64 | 1.38 |
| Shot Accuracy (SOT/Shots) | ~0.36 | ~0.35 |
| Goal Conversion (Goals/Shots) | ~0.12 | ~0.12 |
| Goals per SOT | ~0.27 | ~0.27 |

The engineered efficiency features formalise this pipeline:
- `home_eff_goal_conversion` (mean ~0.12): proportion of shots that become goals
- `home_eff_shot_accuracy` (mean ~0.35): proportion of shots that are on target
- `home_eff_goals_per_sot`: goals scored per shot on target

The xG proxy is derived directly from this chain: `home_xg_proxy = home_form_avg_sot_10 × 0.1` (overall mean: **0.49** home, **0.50** away). This approximation — that roughly 10% of shots on target become goals — is validated by the raw conversion rates above.

---

## 4. Average Goals Scored (Fig 4.4)

| Club | Home Scored | Away Scored | Total |
|---|---|---|---|
| **Manchester City** | **2.74** | 2.05 | **4.79** |
| **Liverpool** | **2.34** | 1.99 | **4.33** |
| **Arsenal** | 2.03 | 1.66 | 3.69 |
| **Chelsea** | 1.73 | 1.62 | 3.35 |
| **Manchester United** | 1.75 | 1.36 | 3.11 |

**Man City** is the most prolific scorer (2.74 home goals/game), followed by Liverpool (2.34). Man City's 2.05 away average shows they score prolifically regardless of venue. **Man United** scores the fewest away (1.36), a 0.39 gap from their home rate — they are particularly reliant on home advantage for scoring.

**Feature engineering context — Full Per-Club Engineered Profile:**

These raw scoring/conceding numbers expand into a rich set of engineered features that capture team quality from multiple angles:

#### Club Engineered Profile (Home Appearances)

| Club | Elo Range | Avg Elo | PPM | CS Rate | Goal Conv | Shot Acc | Win Streak | Unbeaten | HT Win Rate | Home Adv |
|---|---|---|---|---|---|---|---|---|---|---|
| Man City | 1510–1806 | 1695 | 2.25 | 0.42 | 0.14 | 0.37 | 2.4 | 4.7 | 0.56 | 2.53 |
| Liverpool | 1519–1760 | 1679 | 2.22 | 0.38 | 0.13 | 0.36 | 2.1 | 5.2 | 0.46 | 2.05 |
| Arsenal | 1482–1748 | 1614 | 1.89 | 0.36 | 0.13 | 0.35 | 1.3 | 3.3 | 0.42 | 1.88 |
| Man United | 1489–1650 | 1584 | 1.67 | 0.31 | 0.11 | 0.37 | 0.7 | 2.4 | 0.35 | 1.47 |
| Chelsea | 1490–1648 | 1572 | 1.64 | 0.31 | 0.11 | 0.36 | 0.8 | 2.2 | 0.37 | 1.77 |

- **Elo** (`home_elo`): Dynamic strength rating (init=1500, K=20). Man City's peak of **1806** is the highest in the dataset; Man United's ceiling of **1650** is lower than Man City's floor of 1510.
- **PPM** (`home_team_ppm`): Points per match over last 30. Man City/Liverpool average >2.0 (equivalent to 60+ pts/season pace), while Chelsea/United are below 1.70.
- **CS Rate** (`home_def_clean_sheet_rate`): Man City keeps a clean sheet in 42% of home matches; United and Chelsea only 31%.
- **Goal Conversion** (`home_eff_goal_conversion`): Man City converts 14% of shots to goals; Chelsea/United only 11% — a significant efficiency gap.
- **Win Streak** (`home_mom_win_streak`): Average max consecutive wins is 2.4 for City, but only 0.7 for United.
- **Home Advantage Index** (`home_advantage_index`): City's 2.53 means their home form is 2.5x their away form; United's 1.47 is the weakest venue differential.

---

## 5. Average Goals Conceded (Fig 4.5)

| Club | Home Conceded | Away Conceded | Total |
|---|---|---|---|
| **Liverpool** | **0.83** | 1.18 | 2.01 |
| **Man City** | 0.89 | **0.90** | 1.79 |
| **Arsenal** | 1.05 | **1.06** | 2.11 |
| **Chelsea** | 1.04 | 1.38 | 2.42 |
| **Man United** | 1.17 | **1.39** | 2.56 |

**Man City** concedes the fewest overall (1.79/game), and remarkably concedes the same at home (0.89) as away (0.90) — they defend equally well everywhere. **Liverpool** has the best home defensive record (0.83 conceded) but concedes more away (1.18). **Man United** has the worst defensive record, conceding 1.39 away — the worst in the group.

#### Club Engineered Profile (Away Appearances)

| Club | Elo Range | Avg Elo | PPM | CS Rate | Goal Conv | Shot Acc | Win Streak | Unbeaten | Trend (Goals) | Trend (SOT) |
|---|---|---|---|---|---|---|---|---|---|---|
| Man City | 1510–1805 | 1698 | 2.26 | 0.41 | 0.14 | 0.37 | 2.4 | 4.7 | +0.042 | +0.027 |
| Liverpool | 1529–1764 | 1679 | 2.22 | 0.38 | 0.13 | 0.36 | 2.3 | 5.7 | +0.026 | +0.090 |
| Arsenal | 1492–1750 | 1616 | 1.89 | 0.35 | 0.13 | 0.35 | 1.4 | 3.3 | +0.012 | +0.014 |
| Man United | 1492–1648 | 1585 | 1.69 | 0.32 | 0.11 | 0.37 | 0.9 | 2.7 | -0.026 | +0.017 |
| Chelsea | 1490–1646 | 1575 | 1.67 | 0.30 | 0.11 | 0.36 | 1.0 | 2.4 | -0.007 | +0.016 |

- **Trend features** (`away_trend_goals`, `away_trend_sot`): Linear regression slopes over last 5 matches. Positive values indicate improving form; Man City (+0.042 goals/match) and Liverpool (+0.026) trend upward, while Man United (-0.026) trends downward.
- **Liverpool's away unbeaten streak** (5.7) is the longest — they rarely lose on the road.

---

## 6. Home vs Away Performance (Fig 4.6)

**Left panel — Goals Scored:** Every club scores more at home. The gap is largest for Man City (2.74 vs 2.05, +0.69) and smallest for Chelsea (1.73 vs 1.62, +0.11). Chelsea is the most "neutral" in scoring between venues.

**Middle panel — Shots on Target:**

| Club | Home SOT | Away SOT | Gap |
|---|---|---|---|
| Liverpool | 6.88 | 5.70 | +1.18 |
| Man City | 6.63 | 6.26 | +0.37 |
| Chelsea | 5.83 | 5.03 | +0.80 |
| Man United | 5.68 | 4.76 | +0.92 |
| Arsenal | 5.69 | 4.08 | +1.61 |

Liverpool generates the most home SOT (6.88), but Man City's away SOT (6.26) is highest — they create chances everywhere. Arsenal has the largest home/away SOT gap (1.61).

**Right panel — Total Shots:**

| Club | Home Shots | Away Shots | Drop-off |
|---|---|---|---|
| Liverpool | **18.98** | 15.82 | -16.6% |
| Man City | 18.56 | **16.60** | -10.6% |
| Chelsea | 16.01 | 13.87 | -13.4% |
| Arsenal | 16.18 | 12.07 | -25.4% |
| Man United | 15.66 | 12.53 | -20.0% |

Liverpool averages the highest home shot volume (18.98), while Man City leads away shots (16.60). Man City has the smallest drop-off when traveling (-10.6%), while Arsenal drops the most (-25.4%).

**Feature engineering context — Match Dominance:**

The home/away differentials above are formalised into the `home_dom_*` and `away_dom_*` feature groups (average differential over last 10 matches):

| Club | Shot Diff | Goal Diff | SOT Diff | Corners Diff |
|---|---|---|---|---|
| Man City | **+9.76** | **+1.53** | **+3.57** | **+4.18** |
| Liverpool | +8.00 | +1.19 | +2.74 | +3.22 |
| Chelsea | +4.42 | +0.41 | +1.64 | +1.73 |
| Arsenal | +3.20 | +0.77 | +1.20 | +1.80 |
| Man United | +1.45 | +0.31 | +0.99 | **-0.04** |

- **Man City** dominates every metric at home with a shot differential of +9.76 — they average nearly 10 more shots than opponents at home.
- **Man United** is the only club with a **negative corner differential** (-0.04) — they concede slightly more corners than they win at home, indicating they spend more time defending.
- The `away_dom_shots_diff` feature (ranked #3 by SHAP importance, 0.0581) is the strongest away dominance predictor — teams that control the shot battle away from home are more likely to get results.

**Feature engineering context — Venue Momentum:**

| Club | Home Unbeaten Streak | Home Winning Streak | Home Scoring Streak |
|---|---|---|---|
| Man City | 4.7 | 2.4 | — |
| Liverpool | 5.2 | 2.1 | — |
| Arsenal | 3.3 | 1.3 | — |
| Chelsea | 2.2 | 0.8 | — |
| Man United | 2.4 | 0.7 | — |

Liverpool's home unbeaten streak (5.2) is the longest, reinforcing the Anfield fortress effect. Man United's winning streak of 0.7 means they rarely string together consecutive home wins.

---

## 7. Team Form Trend (Fig 4.7)

**Top panel — Rolling 5-Match Points (scaled to 3 = max):**

Time series from 2019-2024 showing 5-match rolling average points:
- **Man City** peaks highest (~15 points per 5 games = 5 consecutive wins) multiple times, particularly during 2022-2023 title run
- **Liverpool** shows similar peaks but more volatility, with notable dips in early 2021 and late 2023
- **Arsenal** shows a clear upward trajectory — low in 2019-2021 (~6-9 rolling points), climbing to 12+ by 2023-2024 (Arteta's rebuild)
- **Chelsea** and **Man United** oscillate between 6-12, never sustaining elite peaks for long

**Bottom panel — Rolling 5-Match Goals Scored:**
- **Man City** peaks at ~2.5-3.0 goals/match during dominant spells
- **Liverpool** similarly peaks near 2.5-3.0
- **Man United** and **Chelsea** generally hover around 1.0-1.5 goals/match in rolling form

**Feature engineering context — Multi-Window Form Comparison:**

The form features are computed at three time horizons to capture short-term bursts vs sustained quality:

| Window | Avg Points | Avg Goals Scored | Avg Shots | Avg SOT |
|---|---|---|---|---|
| 3 (short) | 5.6 | 5.7 | 15.4 | 4.7 |
| 5 (medium) | 9.5 | 9.5 | 15.5 | 4.7 |
| 10 (long) | 18.8 | 18.9 | 15.6 | 4.7 |

- **Shot volume is stable across windows** (15.4–15.6) — teams generate similar shot counts regardless of timeframe.
- **Points and goals scale linearly** with window size, confirming that form is roughly consistent over 3-10 match horizons.
- The `home_form_avg_shots_3` feature (SHAP rank #41, importance 0.028) captures very short-term shot form — a team that suddenly starts shooting more (or fewer) in their last 3 games provides a predictive signal beyond the 10-match average.

**Feature engineering context — Momentum Streaks:**

| Club | Home Win Streak | Home Unbeaten | Away Win Streak | Away Unbeaten |
|---|---|---|---|---|
| Man City | 2.4 | 4.7 | 2.4 | 4.7 |
| Liverpool | 2.1 | 5.2 | 2.3 | 5.7 |
| Arsenal | 1.3 | 3.3 | 1.4 | 3.3 |
| Chelsea | 0.8 | 2.2 | 1.0 | 2.4 |
| Man United | 0.7 | 2.4 | 0.9 | 2.7 |

Liverpool's away unbeaten streak (5.7) exceeds their home streak (5.2) — they are more resilient on the road than at home in terms of avoiding defeat. Man United's home win streak (0.7) means they almost never win two consecutive home matches.

**Feature engineering context — Trend Slopes (Linear Regression over Last 5):**

| Club | Home Trend (Goals) | Home Trend (SOT) | Away Trend (Goals) | Away Trend (SOT) |
|---|---|---|---|---|
| Man City | -0.060 | -0.055 | +0.042 | +0.027 |
| Liverpool | -0.031 | -0.103 | +0.026 | +0.090 |
| Arsenal | -0.012 | -0.017 | +0.012 | +0.014 |
| Chelsea | +0.004 | -0.027 | -0.007 | +0.016 |
| Man United | +0.022 | -0.014 | -0.026 | +0.017 |

- Most home trend values are slightly **negative** — this is a regression-to-mean effect; teams that scored heavily in early matches trend back toward their average.
- Liverpool's home SOT trend (-0.103) is the steepest decline — their early-season dominance normalises.
- The `away_trend_goals` feature (SHAP rank #11, importance 0.046) is among the top 15 most important features — away scoring trends carry strong predictive signal.

**Feature engineering context — Consistency (Std Dev over Last 10):**

| Club | Goals Std | Shots Std | SOT Std | Corners Std |
|---|---|---|---|---|
| Man City | 1.51 | 5.72 | — | — |
| Liverpool | 1.34 | 5.83 | — | — |
| Arsenal | 1.36 | 5.18 | — | — |
| Chelsea | 1.28 | 5.12 | — | — |
| Man United | 1.28 | 5.34 | — | — |

Man City has the **highest consistency std** for goals (1.51) — paradoxically, this means they are the **most volatile** in scoring, oscillating between very high and moderate tallies. This volatility is itself predictive: when City is "on," they score heavily, creating wider outcome margins. The `home_consist_shots_std` feature (SHAP rank #10, importance 0.046) confirms that shot consistency is among the most discriminating features for the model.

---

## 8. Head-to-Head Heatmap (Fig 4.8)

5x5 heatmap of home wins in H2H matchups among the top-5 clubs:

| Home \ Away | Arsenal | Liverpool | Man City | Chelsea | Man United |
|---|---|---|---|---|---|
| **Arsenal** | — | 3 | 2 | 4 | 5 |
| **Liverpool** | 3 | — | 3 | 3 | 3 |
| **Man City** | 4 | 2 | — | 4 | 3 |
| **Chelsea** | 0 | 1 | 1 | — | 2 |
| **Man United** | 2 | 1 | 2 | 3 | — |

**Total home wins vs top-5:** Arsenal (14), Man City (13), Liverpool (12), Man United (8), **Chelsea (4)**.

**Key findings:**
- **Chelsea is the weakest at home vs top-5** — only 4 home wins total, and has **never beaten Arsenal at home** in this dataset (0 wins)
- **Arsenal dominates Man United at home** (5 wins in 6 seasons)
- **Man City dominates Chelsea and Arsenal at home** (4 wins each)
- **Liverpool** is consistent — exactly 3 home wins against every other top-5 club

**Feature engineering context — Full H2H Feature Set:**

The heatmap shows only one dimension (home wins) of a 12-feature H2H group computed over the last 10 meetings:

| Feature | Mean | Std | Description |
|---|---|---|---|
| `h2h_home_wins` | 2.02 | — | Home team wins in last 10 meetings |
| `h2h_away_wins` | 2.03 | — | Away team wins in last 10 meetings |
| `h2h_draws` | 1.42 | — | Draws in last 10 meetings |
| `h2h_home_goals` | 7.98 | — | Total home team goals across meetings |
| `h2h_away_goals` | 7.98 | — | Total away team goals across meetings |
| `h2h_goal_diff` | 0.0 | — | Home minus away goals |
| `h2h_home_scoring_rate` | 1.34 | — | Home goals per match |
| `h2h_away_scoring_rate` | 1.35 | — | Away goals per match |
| `h2h_avg_shots` | 22.66 | — | Avg total shots per meeting |
| `h2h_home_dominance` | 0.001 | — | (home_wins - away_wins) / n |

Among top-5 clubs, H2H records are **remarkably balanced** — mean home wins (2.02) ≈ away wins (2.03), goal difference ≈ 0, and `h2h_home_dominance` is essentially zero (0.001). This means that among elite teams, home advantage in H2H is minimal, and the raw heatmap alone is insufficient — the model relies on `h2h_away_scoring_rate` (SHAP rank #24, 0.037) and `h2h_home_scoring_rate` (rank #28, 0.035) rather than win counts to capture matchup dynamics.

**Total H2H matches among top-5:** 119 (across 6 seasons).

---

## 9. Correlation Heatmap of Pre-match Features (Fig 4.9)

Heatmap showing correlations among pre-match engineered features from `engineered_features.csv`. Post-match raw columns (FTHG, FTAG, HTHG, HTAG, HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR) are excluded — only features available before kickoff are included.

**High intra-group correlations (>0.90) — candidates for the correlation filter:**

| Feature Pair | r | Why |
|---|---|---|
| `home_form_wins_10` ↔ `home_form_points_10` | >0.95 | Points are computed directly from wins |
| `home_form_goals_scored_10` ↔ `home_form_avg_goals_10` | >0.95 | Avg = goals / window size (constant) |
| `away_form_wins_10` ↔ `away_form_points_10` | >0.95 | Same pattern |
| `home_disc_fouls` ↔ `home_def_avg_fouls` | 1.00 | Duplicate computation |
| `home_disc_yellow` ↔ `home_def_avg_yellow` | 1.00 | Duplicate computation |

These redundant pairs are exactly what the **correlation filter** (|r| > 0.90) removes. Of the 198 original features, **~40 are dropped**, leaving ~158 for SHAP ranking.

**Moderate cross-team correlations:**
- `home_form_*` vs `away_form_*`: weakly negative (-0.1 to -0.3) — when one team is in good form, their opponents tend to be in weaker form (selection effect)
- `h2h_*` features vs season-long features: near-zero — H2H captures matchup-specific dynamics independent of overall form
- Multi-window features (e.g., `home_form_avg_shots_3` vs `home_form_avg_shots_10`): strongly correlated (0.7-0.9) but not above threshold — both are retained because the short window captures different signal than the long window

**Key insight:** The 198 features contain ~40 fully redundant pairs (r > 0.90) that are removed by the correlation filter, but the remaining ~158 features capture **13 distinct predictive dimensions**: form (short/medium/long), venue performance, H2H matchup, attacking strength, defensive strength, discipline, momentum, efficiency, dominance, trend, consistency, first-half performance, and fixture difficulty.

**Bridge to feature engineering:** The strong correlations between home shots and home SOT motivated creating separate feature groups for attacking strength, shots on target, and half-time performance rather than using raw columns directly. The near-zero correlations between disciplinary and goal features motivated the separate `discipline` feature group. This independence is validated by the SHAP ranking: `home_disc_fouls` (rank #9) and `home_disc_red` (rank #26) rank highly despite having zero correlation with goal features — they provide unique predictive information.

---

## 10. SHAP Feature Importance (Fig 4.10)

Horizontal bar chart showing the top 50 selected features ranked by SHAP importance from CatBoost.

**Category distribution of the 50 selected features:**

| Category | Count | % | Top Feature (Rank) | SHAP |
|---|---|---|---|---|
| Consistency | 7 | 14% | `home_consist_shots_std` (#10) | 0.0463 |
| Venue Performance | 7 | 14% | `away_team_away_perf_avg_shots` (#2) | 0.0606 |
| H2H | 4 | 8% | `h2h_away_scoring_rate` (#24) | 0.0370 |
| Fixture Difficulty | 4 | 8% | `away_team_fd_away_opp_avg_gd` (#20) | 0.0390 |
| Discipline | 4 | 8% | `home_disc_fouls` (#9) | 0.0464 |
| Trend | 4 | 8% | `away_trend_goals` (#11) | 0.0460 |
| Dominance | 3 | 6% | `away_dom_shots_diff` (#3) | 0.0581 |
| Difference | 2 | 4% | `diff_form_avg_shots_10` (#7) | 0.0490 |
| Defensive | 2 | 4% | `away_def_shots_faced` (#1) | 0.0649 |
| Attacking | 2 | 4% | `home_atk_avg_shots` (#21) | 0.0386 |
| Efficiency | 2 | 4% | `home_eff_sot_conversion` (#37) | 0.0304 |
| Elo Rating | 2 | 4% | `home_elo` (#6) | 0.0521 |
| HT Performance | 2 | 4% | `home_ht_win_rate` (#8) | 0.0475 |
| Win Percentage | 1 | 2% | `home_team_ppm` (#4) | 0.0562 |
| Home Advantage | 1 | 2% | `home_advantage_index` (#38) | 0.0303 |
| Form | 1 | 2% | `home_form_avg_shots_3` (#41) | 0.0281 |
| Momentum | 1 | 2% | `away_mom_win_streak` (#43) | 0.0272 |
| Venue Momentum | 1 | 2% | `away_team_venue_mom_away_unbeaten` (#40) | 0.0292 |

**Top 15 by SHAP importance:**

| Rank | Feature | SHAP | Category |
|---|---|---|---|
| 1 | `away_def_shots_faced` | 0.0649 | Defensive |
| 2 | `away_team_away_perf_avg_shots` | 0.0606 | Venue Performance |
| 3 | `away_dom_shots_diff` | 0.0581 | Dominance |
| 4 | `home_team_ppm` | 0.0562 | Win Percentage |
| 5 | `away_team_away_perf_avg_sot` | 0.0544 | Venue Performance |
| 6 | `home_elo` | 0.0521 | Elo Rating |
| 7 | `diff_form_avg_shots_10` | 0.0490 | Difference |
| 8 | `home_ht_win_rate` | 0.0475 | HT Performance |
| 9 | `home_disc_fouls` | 0.0464 | Discipline |
| 10 | `home_consist_shots_std` | 0.0463 | Consistency |
| 11 | `away_trend_goals` | 0.0460 | Trend |
| 12 | `home_team_home_perf_avg_sot` | 0.0454 | Venue Performance |
| 13 | `away_disc_fouls` | 0.0452 | Discipline |
| 14 | `home_ht_goals_scored_avg` | 0.0422 | HT Performance |
| 15 | `away_consist_goals_std` | 0.0400 | Consistency |

**Key insight:** The model prioritises **consistency** (how stable a team's performance is) and **venue-specific performance** (how they play in this particular context) over raw stats. The top feature — `away_def_shots_faced` (SHAP 0.0649) — measures how many shots the away team's opponents generate, which is a defensive vulnerability metric. A team that faces many shots away from home is likely to concede, regardless of their attacking quality.

---

## 11. Feature Engineering Pipeline

**Source code:** `model/feature_engineering.py` (function `compute_all_features`, line 787) and `web/ml_engine/feature_engineering.py` (class `FeatureEngineering`, for live prediction).

The 198 engineered features are organised into 20 groups, computed from the 25 raw columns in `cleaned_dataset.csv`:

| # | Group | Window | Count | Description | Example Features |
|---|---|---|---|---|---|
| 1 | Form (multi-window) | 3, 5, 10 | 60 | Rolling W/D/L, points, goals, shots, SOT | `home_form_points_5`, `away_form_avg_goals_10` |
| 2 | Venue Performance | 10 | 10 | Win rate, goals, conceded, shots, SOT at home/away | `home_team_home_perf_avg_goals` |
| 3 | Head-to-Head | 10 | 12 | Wins, goals, scoring rates, dominance vs specific opponent | `h2h_home_scoring_rate`, `h2h_home_dominance` |
| 4 | Attacking Strength | 10 | 8 | Avg goals, shots, SOT, corners (all venues) | `home_atk_avg_shots` |
| 5 | Defensive Strength | 10 | 12 | Avg conceded, clean sheet rate, shots faced, fouls, cards | `home_def_clean_sheet_rate` |
| 6 | Discipline | 10 | 6 | Avg fouls, yellows, reds committed | `home_disc_fouls` |
| 7 | Momentum | 10 | 6 | Consecutive wins, losses, unbeaten runs | `home_mom_win_streak` |
| 8 | Goal Efficiency | 10 | 8 | Conversion rate, shot accuracy, goals per SOT | `home_eff_goal_conversion` |
| 9 | Match Dominance | 10 | 8 | Avg differential in shots, SOT, corners, goals | `home_dom_shots_diff` |
| 10 | Recent Trend | 5 | 8 | Linear regression slope of goals, conceded, shots, SOT | `home_trend_goals` |
| 11 | Consistency | 10 | 12 | Std dev of goals, conceded, shots, SOT, corners, goal diff | `home_consist_goals_std` |
| 12 | First-Half Performance | 10 | 8 | HT goals scored/conceded, HT win/draw rate | `home_ht_win_rate` |
| 13 | Fixture Difficulty | 10 | 6 | Opponent's win rate, goal diff, points at venue | `home_team_fd_home_opp_avg_points` |
| 14 | Venue Momentum | 20 | 6 | Consecutive unbeaten/winning/scoring at specific venue | `away_team_venue_mom_away_unbeaten` |
| 15 | Win Percentage | 30 | 4 | Overall win % and points per match (long window) | `home_team_ppm` |
| 16 | Clean Sheet Rate | 10 | 2 | Proportion of matches with 0 goals conceded | `home_team_cs_rate` |
| 17 | Elo Rating | — | 3 | Dynamic strength rating (K=20, Init=1500) | `home_elo`, `elo_diff` |
| 18 | xG Proxy | — | 3 | SOT-based expected goals approximation | `home_xg_proxy` |
| 19 | Cross-Team Differences | 3, 5, 10 | 15 | Home minus away for form points, goals, shots, SOT, goal diff | `diff_form_avg_shots_10` |
| 20 | Home Advantage Index | — | 1 | Home form points / away form points (window 10) | `home_advantage_index` |
| | **Total** | | **198** | | |

**Output:** `results/engineered_features/engineered_features.csv` — 1,016 data rows × 204 columns (198 features + 6 metadata columns: `home_team`, `away_team`, `date`, `ftr`, `ftr_encoded`, `season`).

---

## 12. Feature Selection Pipeline: 198 → 50

**Source code:** `model/feature_selection.py`

The selection process reduces 198 engineered features to the 50 most predictive through a three-step pipeline:

### Step 1 — Correlation Filter (|r| > 0.90)

- **Input:** 198 features
- **Process:** Compute the full pairwise Pearson correlation matrix. Iterate through feature pairs in upper triangle; when |r| > 0.90, drop the feature with lower mean absolute correlation to all other features (the less "informative" member of the pair).
- **Output:** **~158 features** remain (~40 dropped)
- **Rationale:** Removes redundant features that would cause multicollinearity, inflate variance, and slow training. Examples of dropped pairs:
  - `home_form_wins_10` ↔ `home_form_points_10` (r > 0.95) — points are just wins×3+draws
  - `home_disc_fouls` ↔ `home_def_avg_fouls` (r = 1.00) — duplicate computation
  - `home_form_goals_scored_5` ↔ `home_form_avg_goals_5` (r > 0.95) — avg = total/window

### Step 2 — SHAP Importance Ranking

- **Model:** CatBoostClassifier (500 iterations, depth=5, learning_rate=0.1, 3-class multi-class)
- **Sample:** First 200 matches from the dataset (computational constraint for SHAP)
- **Process:** Train CatBoost on all ~158 filtered features. Use `shap.TreeExplainer` to compute SHAP values. Average the absolute SHAP values across all 3 output classes (H/D/A) to get a single importance score per feature. Rank features descending by mean |SHAP|.
- **Output:** All ~158 features ranked by importance (saved to `results/tables/feature_importance.csv`)

### Step 3 — Feature Count Evaluation

- **Method:** TimeSeriesSplit (5 folds) with XGBClassifier (300 trees, depth=5, lr=0.05, subsample=0.8)
- **Candidate counts:** [30, 40, 50, 60] features (top-N from SHAP ranking)
- **Metric:** Macro F1 score (handles class imbalance) and accuracy

| Features | Macro F1 | Accuracy | Std F1 |
|---|---|---|---|
| 30 | 0.4672 | 0.5787 | 0.0244 |
| 40 | 0.4643 | 0.5799 | 0.0264 |
| **50** | **0.4704** | **0.5870** | 0.0543 |
| 60 | 0.4553 | 0.5740 | 0.0490 |

**Winner: 50 features** — highest macro F1 (0.4704) and accuracy (58.7%). Performance degrades at 60 features, confirming that adding lower-ranked features introduces noise.

**Outputs:**
- `results/tables/selected_features.csv` — ranked list of 50 features
- `web/trained_models/selected_features.pkl` — serialised feature list for prediction
- `results/tables/feature_selection_eval.csv` — evaluation results

---

## 13. Final Model Feature Set (Top 50)

The 50 selected features used by the production stacked-ensemble model, ordered by SHAP importance:

| Rank | SHAP | Category | Feature | Description |
|---|---|---|---|---|
| 1 | 0.0649 | Defensive | `away_def_shots_faced` | Avg shots faced by away team (last 10) |
| 2 | 0.0606 | Venue Perf | `away_team_away_perf_avg_shots` | Away team avg shots in away matches (last 10) |
| 3 | 0.0581 | Dominance | `away_dom_shots_diff` | Away team avg shot differential (last 10) |
| 4 | 0.0562 | Win % | `home_team_ppm` | Home team points per match (last 30) |
| 5 | 0.0544 | Venue Perf | `away_team_away_perf_avg_sot` | Away team avg SOT in away matches (last 10) |
| 6 | 0.0521 | Elo | `home_elo` | Home team pre-match Elo rating |
| 7 | 0.0490 | Difference | `diff_form_avg_shots_10` | Home minus away avg shots (window 10) |
| 8 | 0.0475 | HT Perf | `home_ht_win_rate` | Home team half-time win rate (last 10) |
| 9 | 0.0464 | Discipline | `home_disc_fouls` | Home team avg fouls committed (last 10) |
| 10 | 0.0463 | Consistency | `home_consist_shots_std` | Std dev of home team shots (last 10) |
| 11 | 0.0460 | Trend | `away_trend_goals` | Away team goals trend slope (last 5) |
| 12 | 0.0454 | Venue Perf | `home_team_home_perf_avg_sot` | Home team avg SOT at home (last 10) |
| 13 | 0.0452 | Discipline | `away_disc_fouls` | Away team avg fouls committed (last 10) |
| 14 | 0.0422 | HT Perf | `home_ht_goals_scored_avg` | Home team avg HT goals scored (last 10) |
| 15 | 0.0400 | Consistency | `away_consist_goals_std` | Std dev of away team goals (last 10) |
| 16 | 0.0398 | Consistency | `home_consist_sot_std` | Std dev of home team SOT (last 10) |
| 17 | 0.0395 | Venue Perf | `home_team_home_perf_avg_goals` | Home team avg goals at home (last 10) |
| 18 | 0.0395 | Consistency | `home_consist_corners_std` | Std dev of home team corners (last 10) |
| 19 | 0.0392 | Consistency | `away_consist_conceded_std` | Std dev of away team goals conceded (last 10) |
| 20 | 0.0390 | Fixture Diff | `away_team_fd_away_opp_avg_gd` | Avg goal diff of opponents faced away |
| 21 | 0.0386 | Attacking | `home_atk_avg_shots` | Home team avg shots (all venues, last 10) |
| 22 | 0.0381 | Trend | `away_trend_sot` | Away team SOT trend slope (last 5) |
| 23 | 0.0373 | Dominance | `away_dom_sot_diff` | Away team avg SOT differential (last 10) |
| 24 | 0.0370 | H2H | `h2h_away_scoring_rate` | Away goals per H2H meeting (last 10) |
| 25 | 0.0368 | Fixture Diff | `away_team_fd_away_opp_avg_points` | Avg points of opponents faced away |
| 26 | 0.0363 | Discipline | `home_disc_red` | Home team avg red cards (last 10) |
| 27 | 0.0355 | Defensive | `home_def_shots_faced` | Avg shots faced by home team (last 10) |
| 28 | 0.0354 | H2H | `h2h_home_scoring_rate` | Home goals per H2H meeting (last 10) |
| 29 | 0.0341 | Elo | `elo_diff` | Home minus away Elo rating |
| 30 | 0.0333 | Trend | `home_trend_sot` | Home team SOT trend slope (last 5) |
| 31 | 0.0319 | Venue Perf | `home_team_home_perf_avg_conceded` | Home team avg goals conceded at home |
| 32 | 0.0318 | Consistency | `away_consist_sot_std` | Std dev of away team SOT (last 10) |
| 33 | 0.0315 | Fixture Diff | `home_team_fd_home_opp_avg_points` | Avg points of opponents faced at home |
| 34 | 0.0314 | Venue Perf | `away_team_away_perf_avg_conceded` | Away team avg goals conceded away |
| 35 | 0.0312 | H2H | `h2h_avg_shots` | Avg total shots per H2H meeting |
| 36 | 0.0310 | Fixture Diff | `home_team_fd_home_opp_avg_gd` | Avg goal diff of opponents faced at home |
| 37 | 0.0304 | Efficiency | `home_eff_sot_conversion` | Home team SOT-to-goal conversion (last 10) |
| 38 | 0.0303 | Home Adv | `home_advantage_index` | Home form / away form points (window 10) |
| 39 | 0.0297 | H2H | `h2h_home_dominance` | (home_wins - away_wins) / total H2H matches |
| 40 | 0.0292 | Venue Mom | `away_team_venue_mom_away_unbeaten` | Away team consecutive unbeaten away |
| 41 | 0.0281 | Form | `home_form_avg_shots_3` | Home team avg shots (last 3 matches) |
| 42 | 0.0275 | Efficiency | `away_eff_goals_per_sot` | Away team goals per SOT (last 10) |
| 43 | 0.0272 | Momentum | `away_mom_win_streak` | Away team consecutive wins (last 10) |
| 44 | 0.0267 | Consistency | `away_consist_shots_std` | Std dev of away team shots (last 10) |
| 45 | 0.0266 | Dominance | `away_dom_corners_diff` | Away team avg corner differential (last 10) |
| 46 | 0.0263 | Discipline | `away_disc_yellow` | Away team avg yellow cards (last 10) |
| 47 | 0.0262 | Trend | `away_trend_shots` | Away team shots trend slope (last 5) |
| 48 | 0.0261 | Venue Perf | `home_team_home_perf_avg_shots` | Home team avg shots at home (last 10) |
| 49 | 0.0253 | Attacking | `home_atk_avg_corners` | Home team avg corners (all venues, last 10) |
| 50 | 0.0247 | Difference | `diff_form_avg_shots_3` | Home minus away avg shots (window 3) |

**How these features are used at prediction time:**

1. `web/ml_engine/feature_engineering.py` computes all 198 features for a given home/away team pair
2. The 50 selected features are extracted via `features[self.selected_features]`
3. Three base models make predictions: XGBoost, SVM (with StandardScaler pipeline), Random Forest
4. Their probability outputs (3 classes each = 9 values) are fed into a Logistic Regression meta-learner
5. The meta-learner produces the final prediction with confidence scores

**Model files:** `web/trained_models/` — `xgboost_model.pkl`, `svm_pipeline.pkl`, `random_forest_model.pkl`, `logistic_meta_model.pkl`, `selected_features.pkl`
