"""GTA VI trophy earn-rate forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads one case file (makefun/gta6-trophies-case/v1) per run from the input
directory and writes, for every trophy, five quantiles (5/25/50/75/95%) of its
earn rate at the case's target, as percent.

Method, in one paragraph: the tracker page only changes at irregular
refreshes, so the read history is first collapsed into distinct refresh
states.  Each trophy's rate is smoothed with its players-tracked count
(Jeffreys), mapped to the logit scale, and fitted with a weighted least
squares line against log time since release.  The level is anchored on the
last observed state (the printed rates are exact counts, not samples) and
only the slope is extrapolated, damped for the long final horizon
(phi = 1 / (1 + H/D), with extra shrinkage for falling slopes).  The fit
window and the damping constant, plus the spread
curve sigma(h) = sigma0 + sigma1 * sqrt(h in days), are chosen at run time by
backtesting the same forecaster on pseudo-cutoffs inside the case's own
history, scored with the contest's own pinball loss on the logit scale.
Everything is deterministic, standard library only, and needs no network.

Run:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Candidate grids for the run-time backtest calibration.
WINDOWS_H = [24.0, 48.0, 96.0, 1e18]       # fit windows, hours before a cutoff
DAMPING_DAYS = [1.0, 2.0, 4.0, 1e18]       # trend damping constant D, days
SPREAD_SCALES = [0.85, 1.0, 1.2]           # global multiplier on sigma(h)
BACKTEST_HORIZONS_H = [6.0, 12.0, 24.0, 48.0]

# Fallbacks used when a history is too short to backtest (e.g. one state).
DEFAULT_WINDOW_H = 48.0
DEFAULT_DAMPING_DAYS = 2.0
DEFAULT_SIGMA0 = 0.25
DEFAULT_SIGMA1 = 0.60

MAX_MOVE_LOGIT = 2.0        # cap on |damped trend move| over the horizon
MEDIAN_CLAMP = 7.5          # median logit clamp (percent 0.055 .. 99.945)


def parse_time(text):
    """ISO 8601 (with Z) to epoch seconds."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def clamp_pct(p):
    return min(99.99, max(0.01, p))


def logit_pct(p):
    p = clamp_pct(p)
    return math.log(p / (100.0 - p))


def logistic_pct(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (x is always >= 0 here)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def x_of(t, release_epoch):
    """Log-time coordinate: ln(1 + hours since release / 24)."""
    h = (t - release_epoch) / 3600.0
    if h < 0.0:
        h = 0.0
    return math.log1p(h / 24.0)


def build_states(history):
    """Collapse consecutive identical reads into distinct refresh states.

    A state keeps the time of its first read, the players-tracked count and
    the rate vector.  Returns a list of (t, players, rates) oldest first.
    """
    states = []
    last_key = None
    for read in history:
        rates = read.get("rates") or []
        players = read.get("players_tracked") or 0
        key = (players, tuple(rates))
        if key == last_key:
            continue
        last_key = key
        t = parse_time(read.get("slot") or read.get("at"))
        states.append((t, players, rates))
    return states


def trophy_series(states, idx):
    """One trophy's (t, z, players) points, z the smoothed logit rate."""
    pts = []
    prev_players = 0
    for t, players, rates in states:
        if idx >= len(rates) or rates[idx] is None:
            prev_players = players
            continue
        r = rates[idx]
        if players and players > 0:
            # Jeffreys smoothing with the tracked pool as the sample size.
            prop = (r / 100.0 * players + 0.5) / (players + 1.0)
            z = math.log(prop / (1.0 - prop))
        else:
            z = logit_pct(r)
        dn = players - prev_players if players else 0
        pts.append((t, z, dn))
        prev_players = players
    return pts


def fit_line(xs, zs, ws):
    """Weighted least squares z ~ a + b*x.  Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0.0:
        return (zs[-1] if zs else 0.0), 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    mz = sum(w * z for z, w in zip(zs, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        return mz, 0.0
    sxy = sum(w * (x - mx) * (z - mz) for x, z, w in zip(xs, zs, ws))
    b = sxy / sxx
    return mz - b * mx, b


def predict_median(pts, t_cut, t_target, release_epoch, window_h, damping_days):
    """Damped log-time trend forecast of one trophy's median logit."""
    if not pts:
        return 0.0
    lo = t_cut - window_h * 3600.0
    sel = [p for p in pts if p[0] >= lo]
    if len(sel) < 2:
        sel = pts[-2:] if len(pts) >= 2 else pts
    xs = [x_of(p[0], release_epoch) for p in sel]
    zs = [p[1] for p in sel]
    # Recency weights, with a mild bonus for states that arrived with a
    # large jump in the tracked pool (a real refresh, not a partial read).
    ws = []
    for p in sel:
        age_h = max(0.0, (t_cut - p[0]) / 3600.0)
        bonus = math.sqrt(min(max(p[2], 0), 10000) + 1.0) / 8.0
        ws.append(math.exp(-age_h / 36.0) * min(bonus, 3.0))
    _a, b = fit_line(xs, zs, ws)
    x_c = x_of(t_cut, release_epoch)
    x_t = x_of(t_target, release_epoch)
    horizon_days = max(0.0, (t_target - t_cut) / 86400.0)
    if damping_days >= 1e17:
        phi = 1.0
    else:
        phi = 1.0 / (1.0 + horizon_days / damping_days)
    if b < 0.0:
        phi *= 0.5
    move = phi * b * (x_t - x_c)
    if move > MAX_MOVE_LOGIT:
        move = MAX_MOVE_LOGIT
    elif move < -MAX_MOVE_LOGIT:
        move = -MAX_MOVE_LOGIT
    # Anchor the level on the last observed state, not on the fitted line:
    # the printed rates are exact counts, so the last state is the true
    # current level, and the line is only trusted for its slope.
    m = pts[-1][1] + move
    if m > MEDIAN_CLAMP:
        m = MEDIAN_CLAMP
    elif m < -MEDIAN_CLAMP:
        m = -MEDIAN_CLAMP
    return m


def pinball(z_obs, m, sigma):
    """Mean pinball loss on the logit scale for one forecast, all levels."""
    total = 0.0
    for tau, zt in zip(LEVELS, Z):
        q = m + zt * sigma
        d = z_obs - q
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def backtest(all_series, states, t_cut, release_epoch):
    """Pick (window, damping, sigma0, sigma1, scale) from the history itself.

    Pseudo-cutoffs are placed on refresh states that leave at least 12 h of
    earlier history and some later states to score against.  Returns None
    when there is not enough material, so the caller uses the defaults.
    """
    if len(states) < 6:
        return None
    t_first = states[0][0]
    eligible = []
    for i, (t, _n, _r) in enumerate(states):
        if t - t_first >= 12.0 * 3600.0 and t_cut - t >= 6.0 * 3600.0:
            eligible.append(i)
    if len(eligible) < 3:
        return None
    # At most 10 pseudo-cutoffs, evenly spaced over the eligible states.
    step = max(1, len(eligible) // 10)
    cut_idx = eligible[::step][:10]

    # Observations: (trophy, cutoff_t, horizon_h, z_obs).
    obs = []
    for ci in cut_idx:
        t0 = states[ci][0]
        for hh in BACKTEST_HORIZONS_H:
            t_want = t0 + hh * 3600.0
            if t_want > t_cut:
                continue
            # First state at or after t_want is the answering read.
            ans = None
            for (t, _n, _r) in states:
                if t >= t_want:
                    ans = t
                    break
            if ans is None:
                continue
            for ti, series in enumerate(all_series):
                z_obs = None
                for (t, z, _dn) in series:
                    if t == ans:
                        z_obs = z
                        break
                if z_obs is not None:
                    obs.append((ti, t0, hh, z_obs))
    if len(obs) < 30:
        return None

    def residuals(window_h, damping_days):
        res = []  # (horizon_days, error)
        for (ti, t0, hh, z_obs) in obs:
            past = [p for p in all_series[ti] if p[0] <= t0]
            if len(past) < 1:
                continue
            m = predict_median(past, t0, t0 + hh * 3600.0, release_epoch,
                               window_h, damping_days)
            res.append((hh / 24.0, z_obs - m))
        return res

    # Stage 1: choose window and damping with a provisional spread curve.
    best = None
    for w in WINDOWS_H:
        for d in DAMPING_DAYS:
            res = residuals(w, d)
            if len(res) < 30:
                continue
            # Pinball of the residual distribution around 0 with sigma(h).
            tot = 0.0
            for (hd, e) in res:
                s = 0.30 + 0.60 * math.sqrt(hd)
                tot += pinball(e, 0.0, s)
            loss = tot / len(res)
            if best is None or loss < best[0]:
                best = (loss, w, d, res)
    if best is None:
        return None
    _loss, win_w, win_d, res = best

    # Stage 2: fit sigma(h) = sigma0 + sigma1*sqrt(h_days) to the residuals
    # per horizon bucket, using a robust per-bucket scale (MAD).
    buckets = {}
    for (hd, e) in res:
        buckets.setdefault(round(hd, 4), []).append(e)
    pts = []
    for hd, es in sorted(buckets.items()):
        if len(es) < 5:
            continue
        es_sorted = sorted(es)
        med = es_sorted[len(es_sorted) // 2]
        devs = sorted(abs(e - med) for e in es)
        mad = devs[len(devs) // 2]
        sigma = 1.4826 * mad
        if sigma < 0.05:
            sigma = 0.05
        pts.append((math.sqrt(hd), sigma))
    if len(pts) >= 2:
        a, b = fit_line([p[0] for p in pts], [p[1] for p in pts],
                        [1.0] * len(pts))
        sigma0, sigma1 = a, b
    elif len(pts) == 1:
        sigma0, sigma1 = pts[0][1], 0.0
    else:
        sigma0, sigma1 = DEFAULT_SIGMA0, DEFAULT_SIGMA1
    if sigma1 < 0.0:
        sigma1 = 0.0
    if sigma0 < 0.05:
        sigma0 = 0.05
    if sigma0 + sigma1 > 3.0:
        sigma1 = max(0.0, 3.0 - sigma0)

    # Stage 3: one global spread multiplier, by pinball on the residuals.
    best_scale, best_loss = 1.0, None
    for scale in SPREAD_SCALES:
        tot = 0.0
        for (hd, e) in res:
            s = (sigma0 + sigma1 * math.sqrt(hd)) * scale
            tot += pinball(e, 0.0, s)
        loss = tot / len(res)
        if best_loss is None or loss < best_loss:
            best_loss, best_scale = loss, scale
    return (win_w, win_d, sigma0, sigma1, best_scale)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    t_cut = parse_time(case["at"])
    t_target = parse_time(case["target"])
    release_epoch = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    horizon_days = max(0.0, (t_target - t_cut) / 86400.0)

    states = build_states(history)
    all_series = [trophy_series(states, i) for i in range(len(trophies))]

    calib = backtest(all_series, states, t_cut, release_epoch)
    if calib is None:
        window_h, damping, sigma0, sigma1, scale = (
            DEFAULT_WINDOW_H, DEFAULT_DAMPING_DAYS,
            DEFAULT_SIGMA0, DEFAULT_SIGMA1, 1.0)
    else:
        window_h, damping, sigma0, sigma1, scale = calib
    sigma = (sigma0 + sigma1 * math.sqrt(horizon_days)) * scale
    if sigma < 0.08:
        sigma = 0.08

    rows = []
    for i, tr in enumerate(trophies):
        series = all_series[i]
        if series:
            m = predict_median(series, t_cut, t_target, release_epoch,
                               window_h, damping)
        else:
            m = logit_pct(0.01)
        qs = []
        prev = 0.0
        for zt in Z:
            q = logistic_pct(m + zt * sigma)
            q = round4(q)
            if q < prev:
                q = prev
            if q > 100.0:
                q = 100.0
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (case["id"] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
