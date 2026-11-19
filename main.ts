// gta6-trophy-rarity-forecast-01 — v1
// "Logit-space CQR quantile fan v1: realised-move-scaled trophy earn-rate forecast"
//
// Method (deterministic, offline, stdlib only). Per trophy, work in logit
// space on the earn rate.
//  * Two forward signals: (a) a logit-rate trend, slope damped 0.75; (b) an
//    earners/players split — log earners and log players tracked each get
//    their own OLS line (slopes damped 0.75 / 0.5) and recombine into a rate.
//    Each signal is fitted on TWO clocks — ln(1 + hours since
//    2026-11-19T00:00Z), which saturates, and plain hours, which keeps
//    growing — and a walk-forward backtest on the case's own history picks
//    each signal's clock by its 24-hour-ahead error. The two signals then
//    blend by their backtest errors, and the blend mixes with persistence
//    by w = errPersist / (errPersist + errCore), clamped to [0.2, 0.8].
//  * Calibration (CQR-style): the same backtest measures the 90th percentile
//    of the blended predictor's 24-hour-ahead absolute error and sets the
//    spread from it, floored at a random-walk 0.5*sqrt(days); a bias
//    correction (clamped to +/-0.5 logit) is applied. The final case is one
//    unseen case and every fit above uses only the history inside the case
//    file — nothing is fitted to the practice cases.
//  * Flat history (every read identical, as on launch night): Exophase had
//    not refreshed yet, so persistence is the wrong centre. Use the
//    refresh-shock prior from the rules: players 638 -> 3,868 (~6x the
//    denominator), so the median logit shifts down by 1.4 with sigma 0.5.
//  * A trophy at 100% stays at 100%.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
// Launch-night epoch: the trend clock runs from 2026-11-19T00:00:00Z.
const EPOCH_MS = Date.parse('2026-11-19T00:00:00Z');

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logitOf = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const pctOfLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const clampNum = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));

interface TrophyDef {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophyDef[];
  history: Read[];
}

/** Least-squares line y ~ a + b*x; returns [a, b]. Degenerate x -> [mean(y), 0]. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx < 1e-12) return [my, 0];
  const b = sxy / sxx;
  return [my - b * mx, b];
}

/** Predict at xPred from an OLS fit on (xs, ys) with the slope damped. */
function dampedPredict(xs: number[], ys: number[], xPred: number, damp: number): number {
  const [a, b] = ols(xs, ys);
  return a + damp * b * xPred;
}

/** One trophy's time series extracted from a case. */
interface Series {
  xLn: number[]; // ln(1 + hours since epoch)
  xDay: number[]; // hours since epoch
  tH: number[]; // hours since epoch
  logit: number[];
  logEarners: number[];
  logPlayers: number[];
  lastRate: number;
  flat: boolean;
}

function extractSeries(c: CaseFile, idx: number): Series {
  const xLn: number[] = [];
  const xDay: number[] = [];
  const tH: number[] = [];
  const logit: number[] = [];
  const logEarners: number[] = [];
  const logPlayers: number[] = [];
  let lastRate = 0;
  for (const r of c.history) {
    const rate = r.rates[idx];
    if (rate === null || rate === undefined) continue;
    lastRate = rate;
    const h = (Date.parse(r.at) - EPOCH_MS) / 3600000;
    xLn.push(Math.log1p(Math.max(0, h)));
    xDay.push(Math.max(0, h));
    tH.push(h);
    logit.push(logitOf(rate));
    const players = Math.max(1, r.players_tracked || 1);
    const earners = Math.max(0.001, (rate / 100) * players);
    logEarners.push(Math.log(earners));
    logPlayers.push(Math.log(players));
  }
  let flat = true;
  for (let i = 1; i < logit.length; i++) {
    if (Math.abs(logit[i] - logit[0]) > 1e-12) {
      flat = false;
      break;
    }
  }
  return { xLn, xDay, tH, logit, logEarners, logPlayers, lastRate, flat };
}

/** First index with tH[i] >= t, or -1. */
function firstAtOrAfter(tH: number[], t: number): number {
  for (let i = 0; i < tH.length; i++) {
    if (tH[i] >= t) return i;
  }
  return -1;
}

/** Decomposition prediction: log earners / log players fits -> logit rate. */
function decompPredict(
  s: Series,
  k: number,
  xPredLn: number,
  xPredDay: number,
  useLn: boolean,
): number {
  const xs = useLn ? s.xLn.slice(0, k) : s.xDay.slice(0, k);
  const xPred = useLn ? xPredLn : xPredDay;
  const ePred = Math.exp(dampedPredict(xs, s.logEarners.slice(0, k), xPred, 0.75));
  const pPred = Math.exp(dampedPredict(xs, s.logPlayers.slice(0, k), xPred, 0.5));
  return logitOf(clampPct((100 * ePred) / Math.max(1, pPred)));
}

interface Backtest {
  /** Weight on the ln-clock for the trend signal. */
  wLnTrend: number;
  /** Weight on the ln-clock for the decomposition signal. */
  wLnDecomp: number;
  /** Weight on the trend signal inside the core. */
  aTrend: number;
  /** Weight on the core vs persistence. */
  wCore: number;
  bias: number;
  blendErrs: number[];
}

/**
 * Walk-forward backtest: for every prefix of length k >= 3, fit both signals
 * on both clocks and predict the logit 24 hours after the prefix's last read
 * (the first read at or after that time); persistence predicts the prefix's
 * last logit. Clock and signal weights come from the 24-hour-ahead errors;
 * the spread inputs are the errors of the fully blended predictor. A 24-hour
 * horizon is used because a one-step backtest rewards persistence and says
 * nothing about a multi-day forecast. Null when too few reads or no
 * 24-hour-ahead point exists.
 */
function backtest(s: Series): Backtest | null {
  const n = s.xLn.length;
  if (n < 5) return null;
  // Per-prefix predictions, stored to derive weights then blended errors.
  const rows: Array<{
    actual: number;
    p: number;
    tLn: number;
    tDay: number;
    dLn: number;
    dDay: number;
  }> = [];
  for (let k = 3; k < n; k++) {
    const j = firstAtOrAfter(s.tH, s.tH[k - 1] + 24);
    if (j < 0) continue;
    const xsLn = s.xLn.slice(0, k);
    const xsDay = s.xDay.slice(0, k);
    const ys = s.logit.slice(0, k);
    rows.push({
      actual: s.logit[j],
      p: ys[ys.length - 1],
      tLn: dampedPredict(xsLn, ys, s.xLn[j], 0.75),
      tDay: dampedPredict(xsDay, ys, s.xDay[j], 0.75),
      dLn: decompPredict(s, k, s.xLn[j], s.xDay[j], true),
      dDay: decompPredict(s, k, s.xLn[j], s.xDay[j], false),
    });
  }
  if (rows.length === 0) return null;
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  const err = (f: (r: (typeof rows)[number]) => number) =>
    mean(rows.map((r) => Math.abs(r.actual - f(r))));
  const errP = err((r) => r.p);
  const errTLn = err((r) => r.tLn);
  const errTDay = err((r) => r.tDay);
  const errDLn = err((r) => r.dLn);
  const errDDay = err((r) => r.dDay);
  const wLnTrend = clampNum(errTDay / (errTLn + errTDay + 1e-9), 0.25, 0.75);
  const wLnDecomp = clampNum(errDDay / (errDLn + errDDay + 1e-9), 0.25, 0.75);
  const errTrend = wLnTrend * errTLn + (1 - wLnTrend) * errTDay;
  const errDecomp = wLnDecomp * errDLn + (1 - wLnDecomp) * errDDay;
  const aTrend = clampNum(errDecomp / (errTrend + errDecomp + 1e-9), 0.25, 0.75);
  const core = (r: (typeof rows)[number]) => {
    const t = wLnTrend * r.tLn + (1 - wLnTrend) * r.tDay;
    const d = wLnDecomp * r.dLn + (1 - wLnDecomp) * r.dDay;
    return aTrend * t + (1 - aTrend) * d;
  };
  const errCore = err(core);
  const wCore = clampNum(errP / (errP + errCore + 1e-9), 0.2, 0.8);
  const blendErrs: number[] = [];
  const signed: number[] = [];
  for (const r of rows) {
    const pred = wCore * core(r) + (1 - wCore) * r.p;
    blendErrs.push(Math.abs(r.actual - pred));
    signed.push(r.actual - pred);
  }
  return {
    wLnTrend,
    wLnDecomp,
    aTrend,
    wCore,
    bias: clampNum(mean(signed), -0.5, 0.5),
    blendErrs,
  };
}

/** Forecast one trophy: returns five quantiles in percent. */
function forecastTrophy(c: CaseFile, idx: number): number[] {
  const s = extractSeries(c, idx);
  const days = c.horizon_minutes / 1440;
  const targetH = (Date.parse(c.target) - EPOCH_MS) / 3600000;
  const xTargetLn = Math.log1p(Math.max(0, targetH));
  const xTargetDay = Math.max(0, targetH);

  // A trophy every tracked player already has stays at 100%.
  if (s.lastRate >= 100) return [100, 100, 100, 100, 100];
  if (s.logit.length === 0) return [0.01, 0.01, 0.01, 0.01, 0.01];

  const lastL = s.logit[s.logit.length - 1];

  // --- Flat history: refresh-shock prior (see header). ---
  if (s.flat || s.logit.length < 3) {
    const med = lastL - 1.4;
    const sig = Math.sqrt(0.25 + 0.25 * days);
    return Z.map((z) => round4(clampPct(pctOfLogit(med + z * sig))));
  }

  // --- Forward signals on both clocks. ---
  const n = s.xLn.length;
  const mTrendLn = dampedPredict(s.xLn, s.logit, xTargetLn, 0.75);
  const mTrendDay = dampedPredict(s.xDay, s.logit, xTargetDay, 0.75);
  const mDecompLn = decompPredict(s, n, xTargetLn, xTargetDay, true);
  const mDecompDay = decompPredict(s, n, xTargetLn, xTargetDay, false);

  // --- Walk-forward calibration (CQR-style). ---
  const bt = backtest(s);
  let med = 0.5 * (0.5 * (mTrendLn + mTrendDay) + 0.5 * (mDecompLn + mDecompDay)) + 0.5 * lastL;
  let spread = 0.5 * Math.sqrt(days) + 0.3;
  if (bt !== null) {
    const mTrend = bt.wLnTrend * mTrendLn + (1 - bt.wLnTrend) * mTrendDay;
    const mDecomp = bt.wLnDecomp * mDecompLn + (1 - bt.wLnDecomp) * mDecompDay;
    const mCore = bt.aTrend * mTrend + (1 - bt.aTrend) * mDecomp;
    med = bt.wCore * mCore + (1 - bt.wCore) * lastL + bt.bias;
    // CQR spread: the 90th percentile of the blended backtest errors is the
    // half-width of the 90% band, so spread = q90 / 1.645. Floor at the
    // random-walk spread so a quiet history never claims false certainty.
    const sorted = [...bt.blendErrs].sort((a, b) => a - b);
    const qi = Math.min(sorted.length - 1, Math.floor(0.9 * sorted.length));
    const q90 = sorted[qi];
    spread = Math.max(0.5 * Math.sqrt(days), q90 / 1.645, 0.15);
  }
  med = clampNum(med, logitOf(0.01), logitOf(99.99));
  const qs = Z.map((z) => round4(clampPct(pctOfLogit(med + z * spread))));
  // Enforce non-decreasing (already true by construction; be safe).
  for (let i = 1; i < qs.length; i++) {
    if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = c.trophies.map((t, i) => ({ id: t.id, q: forecastTrophy(c, i) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
