---
title: Prem Lab
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: static
pinned: false
---

**Prem Lab** - Premier League match outcome predictions from three independently trained,
walk-forward validated models (Logistic Regression, Random Forest, XGBoost)
- none of them trained on bookmaker odds. See predictions for the next
gameweek, browse historical out-of-fold results back to 2015/16, and dig
into the methodology (calibration, feature importance, how the models
compare to the betting market). Static HTML/CSS/JS - no server, no live
model, ships with pre-computed exports (see the project's ADR-020/024 for
why).

Full project, methodology, and code: https://github.com/mateusz-gob1/football-outcome-mlops
