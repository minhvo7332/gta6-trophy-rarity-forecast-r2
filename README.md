# Logit-space CQR quantile fan v1: realised-move-scaled trophy earn-rate forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minhvo7332/gta6-trophy-rarity-forecast-r2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minhvo7332/gta6-trophy-rarity-forecast-r2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

make.fun bounty `gta6-trophy-rarity-forecast-01` ($250, closes 21/11 18:30 +07).

Forecasts each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00Z:
five quantiles (5/25/50/75/95%) per trophy, from `node main.ts` (Node 22,
stdlib only, offline, deterministic).

## Method

- Per trophy, work in logit space.
- Two forward signals: a damped logit-rate trend, and an earners/players
  decomposition (separate OLS lines for log earners and log players tracked).
- Each signal is fitted on two clocks — `ln(1+h)` (saturating) and plain
  hours (keeps growing) — and a walk-forward backtest on the case's own
  history (every prefix predicts 24h ahead) picks each signal's clock, the
  signal blend, and the blend with persistence.
- CQR-style calibration: the spread comes from the 90th percentile of the
  blended backtest error, floored at a random-walk `0.5*sqrt(days)`; bias
  correction clamped to ±0.5 logit.
- Flat history (launch night, no Exophase refresh yet): refresh-shock prior —
  median logit shifts down 1.4 (players 638 → 3,868, ~6x denominator), sigma 0.5.
- A trophy at 100% stays at 100%. The final case is never in any fit: every
  fit uses only the history inside the case file.

## Local probe (NOT a server grade)

Practice cases p00/p01, answers reconstructed from the 04:30 refresh
(players 638 → 3,868; earners growth factor g):

| g | this entry | house baseline |
|---|---:|---:|
| 1.0 | 0.206 | 1.688 |
| 1.5 | 0.102 | 1.289 |
| 2.0 | 0.157 | 1.006 |

Synthetic final-like case (flat night + refresh jump + growth, known truth):
0.392 vs house baseline 0.465. 3 runs byte-identical, ~0.2 s/run, 55/55 rows valid.
