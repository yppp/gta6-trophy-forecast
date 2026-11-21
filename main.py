"""GTA VI trophy earn-rate forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads one case file (makefun/gta6-trophies-case/v1) per run from the input
directory and writes, for every trophy, five quantiles (5/25/50/75/95%) of its
earn rate at the case's target, as percent.

Method, in one paragraph: the tracker page only changes at irregular
refreshes, so the read history is first collapsed into distinct refresh
states.  Two views of each state are modelled.  The rate view maps a trophy's
rate (Jeffreys-smoothed with the players-tracked count) to the logit scale
and extrapolates a damped log-time trend.  The count view decomposes the rate
into earners E = rate * players / 100 and the tracked pool N, extrapolates
both in log space with the same damped-trend machinery, and recombines them;
this separates genuine uptake from pool dilution, which a rate-only trend
conflates.  The two medians are blended on the logit scale.  The fit window,
the trend damping constant, the blend weight and the spread curve
sigma(h) = sigma0 + sigma1 * sqrt(h in days) are all chosen at run time by
backtesting the same forecaster on pseudo-cutoffs inside the case's own
history, scored with the contest's own pinball loss on the logit scale.
Three guards keep that calibration honest at the real horizon, which is
days beyond any backtest horizon: the choice is scored on the longest
backtest horizon alone, the damping grid holds only finite constants and
the blend never lets the count view dominate, the recombined count median
is clamped to the rate view's move cap, and the final spread has a floor
that grows with the horizon.
Levels are anchored on the last observed state (the printed rates are exact
counts, not samples) and only slopes are extrapolated.  Everything is
deterministic, standard library only, and needs no network.

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
DAMPING_DAYS = [1.0, 2.0, 4.0]             # trend damping constant D, days
BLEND_W = [0.5, 0.75, 1.0]                 # weight on the rate-trend median
SPREAD_SCALES = [0.85, 1.0, 1.2]           # global multiplier on sigma(h)
BACKTEST_HORIZONS_H = [6.0, 12.0, 24.0, 48.0]

# Fallbacks used when a history is too short to backtest (e.g. one state).
DEFAULT_WINDOW_H = 48.0
DEFAULT_DAMPING_DAYS = 2.0
DEFAULT_BLEND_W = 0.5
DEFAULT_SIGMA0 = 0.25
DEFAULT_SIGMA1 = 0.60

MAX_MOVE_LOGIT = 2.0        # cap on |damped trend move| for rates, logit
MAX_MOVE_LOG = 3.0          # cap on |damped trend move| for ln counts
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
    """One trophy's (t, z, dN) points, z the smoothed logit rate."""
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


def earner_series(states, idx):
    """One trophy's (t, ln(E + 0.5), dN) points; E = rate/100 * players."""
    pts = []
    prev_players = 0
    for t, players, rates in states:
        if idx >= len(rates) or rates[idx] is None or not players:
            prev_players = players
            continue
        earners = rates[idx] / 100.0 * players
        dn = players - prev_players if players else 0
        pts.append((t, math.log(earners + 0.5), dn))
        prev_players = players
    return pts


def player_series(states):
    """The tracked pool's (t, ln N, dN) points."""
    pts = []
    prev_players = 0
    for t, players, _rates in states:
        if not players:
            continue
        dn = players - prev_players
        pts.append((t, math.log(players), dn))
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


def series_slope(pts, t_cut, release_epoch, window_h):
    """WLS slope of a series over the window ending at t_cut.

    pts must already be truncated to t <= t_cut.  Returns (slope, last_y).
    Weights: recency decay, with a mild bonus for states that arrived with
    a large jump in the tracked pool (a real refresh, not a partial read).
    """
    if not pts:
        return 0.0, 0.0
    lo = t_cut - window_h * 3600.0
    sel = [p for p in pts if p[0] >= lo]
    if len(sel) < 2:
        sel = pts[-2:] if len(pts) >= 2 else pts
    xs = [x_of(p[0], release_epoch) for p in sel]
    ys = [p[1] for p in sel]
    ws = []
    for p in sel:
        age_h = max(0.0, (t_cut - p[0]) / 3600.0)
        bonus = math.sqrt(min(max(p[2], 0), 10000) + 1.0) / 8.0
        ws.append(math.exp(-age_h / 36.0) * min(bonus, 3.0))
    _a, b = fit_line(xs, ys, ws)
    return b, pts[-1][1]


def damped_move(b, dx, horizon_days, damping_days, cap):
    """Damped, capped trend move; falling slopes are damped twice as hard."""
    if damping_days >= 1e17:
        phi = 1.0
    else:
        phi = 1.0 / (1.0 + horizon_days / damping_days)
    if b < 0.0:
        phi *= 0.5
    move = phi * b * dx
    if move > cap:
        move = cap
    elif move < -cap:
        move = -cap
    return move


def trend_median(slope_last, t_cut, t_target, release_epoch, damping_days):
    """Rate-view median logit from a cached (slope, last z)."""
    b, z_last = slope_last
    dx = x_of(t_target, release_epoch) - x_of(t_cut, release_epoch)
    horizon_days = max(0.0, (t_target - t_cut) / 86400.0)
    m = z_last + damped_move(b, dx, horizon_days, damping_days, MAX_MOVE_LOGIT)
    if m > MEDIAN_CLAMP:
        m = MEDIAN_CLAMP
    elif m < -MEDIAN_CLAMP:
        m = -MEDIAN_CLAMP
    return m


def count_median(e_slope_last, n_slope_last, e_last_raw, n_last_raw,
                 t_cut, t_target, release_epoch, damping_days, z_last):
    """Count-view median logit: forecast earners and pool, recombine.

    e_slope_last / n_slope_last are (slope, last ln value) pairs for
    ln(E + 0.5) and ln N.  Earners are floored near their last count (the
    pool churns slowly) and capped by the forecast pool.  The recombined
    median is finally clamped to the same move cap as the rate view: near
    saturation a small relative error in E or N otherwise explodes on the
    logit scale.
    """
    dx = x_of(t_target, release_epoch) - x_of(t_cut, release_epoch)
    horizon_days = max(0.0, (t_target - t_cut) / 86400.0)
    b_e, ln_e = e_slope_last
    b_n, ln_n = n_slope_last
    e_hat = math.exp(ln_e + damped_move(b_e, dx, horizon_days,
                                        damping_days, MAX_MOVE_LOG)) - 0.5
    n_hat = math.exp(ln_n + damped_move(b_n, dx, horizon_days,
                                        damping_days, MAX_MOVE_LOG))
    if n_hat < 0.98 * n_last_raw:
        n_hat = 0.98 * n_last_raw
    if e_hat < 0.90 * e_last_raw:
        e_hat = 0.90 * e_last_raw
    if e_hat > n_hat:
        e_hat = n_hat
    if e_hat < 0.0:
        e_hat = 0.0
    if n_hat <= 0.0:
        return None
    z = logit_pct(100.0 * e_hat / n_hat)
    if z > z_last + MAX_MOVE_LOGIT:
        z = z_last + MAX_MOVE_LOGIT
    elif z < z_last - MAX_MOVE_LOGIT:
        z = z_last - MAX_MOVE_LOGIT
    return z


def pinball(z_obs, m, sigma):
    """Mean pinball loss on the logit scale for one forecast, all levels."""
    total = 0.0
    for tau, zt in zip(LEVELS, Z):
        q = m + zt * sigma
        d = z_obs - q
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def backtest(all_series, all_earners, pool_series, states, t_cut,
             release_epoch):
    """Pick (window, damping, blend, sigma0, sigma1, scale) from history.

    Pseudo-cutoffs are placed on refresh states that leave at least 12 h of
    earlier history and some later states to score against.  Slopes for
    every (trophy, cutoff, window) are fitted once and cached; the grid
    over damping and blend weight is then pure arithmetic.  Returns None
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
    step = max(1, len(eligible) // 10)
    cut_idx = eligible[::step][:10]

    # Raw last counts per (trophy, cutoff) for the count view's floors.
    def last_counts(ti, t0):
        e_raw = n_raw = None
        for (t, players, rates) in states:
            if t > t0:
                break
            n_raw = players
            if ti < len(rates) and rates[ti] is not None and players:
                e_raw = rates[ti] / 100.0 * players
        return e_raw, n_raw

    # Observations with cached slopes:
    # (z_obs, horizon_days, dx, rate_slope_last, e_slope_last, n_slope_last,
    #  e_raw, n_raw) grouped by window.
    obs_by_window = {w: [] for w in WINDOWS_H}
    n_slopes = {}
    for ci in cut_idx:
        t0 = states[ci][0]
        pool_past = [p for p in pool_series if p[0] <= t0]
        for w in WINDOWS_H:
            n_slopes[(ci, w)] = series_slope(pool_past, t0, release_epoch, w)
        for hh in BACKTEST_HORIZONS_H:
            t_want = t0 + hh * 3600.0
            if t_want > t_cut:
                continue
            ans = None
            for (t, _n, _r) in states:
                if t >= t_want:
                    ans = t
                    break
            if ans is None:
                continue
            dx = x_of(t0 + hh * 3600.0, release_epoch) - x_of(t0, release_epoch)
            for ti in range(len(all_series)):
                z_obs = None
                for (t, z, _dn) in all_series[ti]:
                    if t == ans:
                        z_obs = z
                        break
                if z_obs is None:
                    continue
                e_raw, n_raw = last_counts(ti, t0)
                rate_past = [p for p in all_series[ti] if p[0] <= t0]
                earn_past = [p for p in all_earners[ti] if p[0] <= t0]
                for w in WINDOWS_H:
                    r_sl = series_slope(rate_past, t0, release_epoch, w)
                    e_sl = series_slope(earn_past, t0, release_epoch, w)
                    obs_by_window[w].append(
                        (z_obs, hh / 24.0, dx, r_sl, e_sl, n_slopes[(ci, w)],
                         e_raw, n_raw, t0, hh))
    total_obs = sum(len(v) for v in obs_by_window.values())
    if total_obs < 30:
        return None

    def combo_median(ob, damping_days, blend_w):
        (z_obs, hd, dx, r_sl, e_sl, n_sl, e_raw, n_raw, t0, hh) = ob
        m_tr = trend_median(r_sl, t0, t0 + hh * 3600.0, release_epoch,
                            damping_days)
        if e_raw is None or n_raw is None or not n_raw:
            return m_tr
        m_ct = count_median(e_sl, n_sl, e_raw, n_raw, t0,
                            t0 + hh * 3600.0, release_epoch, damping_days,
                            r_sl[1])
        if m_ct is None:
            return m_tr
        return blend_w * m_tr + (1.0 - blend_w) * m_ct

    # Stage 1: choose window, damping and blend with a provisional spread.
    # The real horizon (days) is far beyond any backtest horizon, so the
    # choice is scored on the longest backtest horizon alone when there is
    # enough of it: short-horizon scores reward undamped extrapolation
    # that the long final horizon punishes.
    best = None
    for w in WINDOWS_H:
        obs_all = obs_by_window[w]
        obs = [ob for ob in obs_all if ob[9] >= max(BACKTEST_HORIZONS_H)]
        if len(obs) < 30:
            obs = obs_all
        if len(obs) < 30:
            continue
        for d in DAMPING_DAYS:
            for bw in BLEND_W:
                tot = 0.0
                for ob in obs:
                    m = combo_median(ob, d, bw)
                    s = 0.30 + 0.60 * math.sqrt(ob[1])
                    tot += pinball(ob[0], m, s)
                loss = tot / len(obs)
                if best is None or loss < best[0]:
                    best = (loss, w, d, bw, obs)
    if best is None:
        return None
    _loss, win_w, win_d, win_bw, win_obs = best

    # Stage 2: fit sigma(h) = sigma0 + sigma1*sqrt(h_days) to the residuals
    # of the winning combo, per horizon bucket, robust scale (MAD).
    res = [(ob[1], ob[0] - combo_median(ob, win_d, win_bw)) for ob in win_obs]
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
    return (win_w, win_d, win_bw, sigma0, sigma1, best_scale)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    t_cut = parse_time(case["at"])
    t_target = parse_time(case["target"])
    release_epoch = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    horizon_days = max(0.0, (t_target - t_cut) / 86400.0)

    states = build_states(history)
    all_series = [trophy_series(states, i) for i in range(len(trophies))]
    all_earners = [earner_series(states, i) for i in range(len(trophies))]
    pool_series = player_series(states)

    calib = backtest(all_series, all_earners, pool_series, states, t_cut,
                     release_epoch)
    if calib is None:
        window_h, damping, blend_w, sigma0, sigma1, scale = (
            DEFAULT_WINDOW_H, DEFAULT_DAMPING_DAYS, DEFAULT_BLEND_W,
            DEFAULT_SIGMA0, DEFAULT_SIGMA1, 1.0)
    else:
        window_h, damping, blend_w, sigma0, sigma1, scale = calib
    sigma = (sigma0 + sigma1 * math.sqrt(horizon_days)) * scale
    # Floor the spread for long horizons: in-history backtests only reach
    # 48 h and can pick a near-zero sigma that a multi-day extrapolation
    # does not earn.
    sigma_floor = 0.10 + 0.10 * math.sqrt(horizon_days)
    if sigma < sigma_floor:
        sigma = sigma_floor

    n_sl = series_slope(pool_series, t_cut, release_epoch, window_h)
    n_last_raw = states[-1][1] if states else 0
    rows = []
    for i, tr in enumerate(trophies):
        series = all_series[i]
        if not series:
            m = logit_pct(0.01)
        else:
            r_sl = series_slope(series, t_cut, release_epoch, window_h)
            m_tr = trend_median(r_sl, t_cut, t_target, release_epoch, damping)
            m = m_tr
            e_series = all_earners[i]
            last_rates = states[-1][2] if states else []
            if e_series and n_last_raw and i < len(last_rates) \
                    and last_rates[i] is not None:
                e_raw = last_rates[i] / 100.0 * n_last_raw
                e_sl = series_slope(e_series, t_cut, release_epoch, window_h)
                m_ct = count_median(e_sl, n_sl, e_raw, n_last_raw, t_cut,
                                    t_target, release_epoch, damping,
                                    series[-1][1])
                if m_ct is not None:
                    m = blend_w * m_tr + (1.0 - blend_w) * m_ct
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
