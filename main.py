#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Reads every case file in the input directory (schema
makefun/gta6-trophies-case/v1) and writes one forecast file per case: five
quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate at the case
target, in percent.

Method, per trophy, in plain stdlib Python, fully deterministic:

  * Split every observed rate into its two moving parts: the earned count
    E = rate * players_tracked / 100 and the tracked base N.  Both are
    fitted as log-linear in log(1 + hours since the first read), with
    recency weights; the slopes are damped because launch-week growth
    decelerates.  The forecast rate is E(target) / N(target).
  * The split is the point.  New players joining pull every rate down
    (dilution) while existing players earning pull it up.  A flat forecast
    misses both, and one straight line through the logit rate confounds
    them.  Earned counts only ever rise; the tracked base is shared by
    every trophy, so it is fitted once and well determined.
  * Spreads come from each trophy's fit residuals, a base term, a term
    that grows with the horizon, a refresh allowance when the history
    shows no refresh yet (a refresh is then due and moves every rate),
    and a small-count term for rare trophies.  Quantiles are the median
    shifted by standard-normal z-scores on the logit scale.
"""

import json
import math
import os
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Weak ceilings, in percent, by tier: only bind when a fitted line blows
# up.  Nobody finishes a ~200-hour platinum in one week; 100% completion
# is barely started; grindy silvers stay low; early bronze story trophies
# are the only ones that can run high.
TIER_CAP = {"platinum": 0.1, "gold": 2.0, "silver": 15.0, "bronze": 80.0}
TIER_FALLBACK = {"platinum": 0.01, "gold": 0.05, "silver": 0.5, "bronze": 2.0}

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Blend weight for the direct logit-rate line in the median: the median's
# logit is BLEND_LINE * logit(line) + (1 - BLEND_LINE) * logit(E/N).
# 0.5 is the honest middle; the synthetic tests tune it.
BLEND_LINE = 0.8
# Slope damping for the direct logit-rate line (1.0 = none).
LINE_DAMP = 1.0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def logit_pct(rate_pct):
    p = min(max(rate_pct / 100.0, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l > 30.0:
        return 99.99
    if l < -30.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-l))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b). Degenerate
    input (all x equal, one point, zero weight) gives slope 0 and the
    weighted mean as intercept."""
    sw = sum(ws)
    if sw <= 0.0 or not xs:
        return 0.0, 0.0
    xm = sum(w * x for w, x in zip(ws, xs)) / sw
    ym = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        return ym, 0.0
    sxy = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return ym - b * xm, b


def wls_quad(xs, ys, ws):
    """Weighted least squares y ~ a + b*x + c*x^2. Returns (a, b, c) or
    None when the system is degenerate.  Solved by Gaussian elimination
    on the 3x3 normal equations."""
    if len(xs) < 5:
        return None
    # Normal-equation sums.
    s = [0.0] * 5  # sum w*x^k, k = 0..4
    t = [0.0] * 3  # sum w*x^k*y, k = 0..2
    for x, y, w in zip(xs, ys, ws):
        xp = 1.0
        for k in range(5):
            s[k] += w * xp
            if k < 3:
                t[k] += w * xp * y
            xp *= x
    # Augmented matrix, 3 rows.
    m = [[s[0], s[1], s[2], t[0]],
         [s[1], s[2], s[3], t[1]],
         [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(3):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [v - f * u for v, u in zip(m[r], m[col])]
    return m[0][3], m[1][3], m[2][3]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    target = parse_time(case["target"])

    if not history:
        return [{"id": t["id"], "q": quantiles_for(TIER_FALLBACK[t["tier"]], 1.5)}
                for t in trophies]

    # Time axis: hours since the first read; x = log(1 + hours).  Weights
    # decay with a 12-hour half-life so recent reads matter most.
    times = [parse_time(r["at"]) for r in history]
    t0 = times[0]
    hours = [(t - t0).total_seconds() / 3600.0 for t in times]
    xs = [math.log1p(max(0.0, h)) for h in hours]
    t_last = times[-1]
    weights = [0.5 ** (max(0.0, (t_last - t).total_seconds() / 3600.0) / 12.0)
               for t in times]
    x_target = math.log1p(max(0.0, (target - t0).total_seconds() / 3600.0))
    span_days = max((t_last - t0).total_seconds() / 86400.0, 0.25)
    horizon_factor = max(1.0, horizon_days / span_days)

    # --- tracked base N: one fit shared by every trophy -------------------
    ns = [r["players_tracked"] for r in history]
    n_last = ns[-1]
    distinct_n = len(set(ns))
    log_n = [math.log(max(1, n)) for n in ns]
    a_n, b_n = wls(xs, log_n, weights)
    n_target = math.exp(a_n + 0.9 * b_n * x_target)
    n_target = min(max(n_target, float(n_last)), 10.0 * n_last)

    # --- per-trophy earned counts -----------------------------------------
    # Pooled residual scale, for trophies too flat or too short to measure
    # their own.  Filled in a first pass below.
    pooled_resid = 0.35
    fits = []
    for ti, trophy in enumerate(trophies):
        pts = []
        for j, r in enumerate(history):
            rate = r["rates"][ti]
            if rate is None:
                continue
            e = max(rate * ns[j] / 100.0, 0.0)
            pts.append((xs[j], weights[j], math.log(max(e, 0.5)), e))
        fits.append(pts)

    resid_pool = []
    for pts in fits:
        if len(pts) < 3:
            continue
        px = [p[0] for p in pts]
        py = [p[2] for p in pts]
        pw = [p[1] for p in pts]
        a, b = wls(px, py, pw)
        for x, w, y, _e in pts:
            resid_pool.append(y - (a + b * x))
    if len(resid_pool) >= 10:
        mean_r = sum(resid_pool) / len(resid_pool)
        pooled_resid = math.sqrt(
            sum((v - mean_r) ** 2 for v in resid_pool) / len(resid_pool))
        pooled_resid = min(max(pooled_resid, 0.1), 1.5)

    rows = []
    for ti, trophy in enumerate(trophies):
        pts = fits[ti]
        last_rate = None
        for r in reversed(history):
            if r["rates"][ti] is not None:
                last_rate = r["rates"][ti]
                break
        if last_rate is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles_for(TIER_FALLBACK[trophy["tier"]], 1.5)})
            continue
        if last_rate >= 99.99:
            # Everybody holds it; it stays (nearly) universal.
            rows.append({"id": trophy["id"],
                         "q": [99.0, 99.7, 99.9, 99.97, 99.99]})
            continue

        e_last = max(last_rate * n_last / 100.0, 0.0)
        if e_last <= 0.0 and all(p[3] <= 0.0 for p in pts):
            # Nobody has earned it yet in any read: it stays ~zero, with a
            # small upward tail in case the first earners show up.
            rows.append({"id": trophy["id"],
                         "q": [0.01, 0.01, 0.01, 0.02, 0.10]})
            continue

        if len(pts) >= 2:
            px = [p[0] for p in pts]
            py = [p[2] for p in pts]
            pw = [p[1] for p in pts]
            a, b = wls(px, py, pw)
            log_e_lin = a + 0.8 * b * x_target
            resid = [y - (a + b * x) for x, w, y, _e in pts]
            # Earned counts accelerate as the bulk of players reaches a
            # trophy, then slow down: a quadratic in log-log space catches
            # the acceleration a straight line misses.  Blend it 50/50
            # with the line (a hedge against overshoot), only when there
            # is enough spread in time to identify curvature, and only
            # allow non-negative curvature (a decelerating count falls
            # back to the line).
            log_e_use = log_e_lin
            quad = None
            if len(pts) >= 5 and (max(px) - min(px)) >= 1.0:
                quad = wls_quad(px, py, pw)
            if quad is not None:
                qa, qb, qc = quad
                qc = min(max(qc, 0.0), 0.3)
                log_e_quad = qa + 0.8 * qb * x_target + qc * x_target ** 2
                log_e_use = 0.5 * (log_e_lin + log_e_quad)
                resid = [y - (qa + qb * x + qc * x * x)
                         for x, w, y, _e in pts]
            e_target = math.exp(log_e_use)
            rmean = sum(resid) / len(resid)
            resid_var = sum((v - rmean) ** 2 for v in resid) / len(resid)
        else:
            e_target = e_last
            resid_var = pooled_resid ** 2
        e_target = min(max(e_target, e_last), n_target)

        median_en = 100.0 * e_target / n_target
        median_en = min(median_en, TIER_CAP.get(trophy["tier"], 80.0))
        median_en = min(max(median_en, 0.01), 99.99)

        # Second view: a straight line through the logit rate itself, the
        # house trend baseline's shape but fitted on every read with
        # recency weights and a mild slope damping.  It reads saturating
        # rate curves well where the earned-count split, damped twice
        # over, undershoots.  Blend the two medians in logit space: when
        # the models agree the blend is either; when they disagree it is
        # the honest middle, and the disagreement widens the spread.
        lx = []
        ly = []
        lw = []
        for j, r in enumerate(history):
            rate = r["rates"][ti]
            if rate is None:
                continue
            lx.append(xs[j])
            ly.append(logit_pct(rate))
            lw.append(weights[j])
        if len(lx) >= 2:
            la, lb = wls(lx, ly, lw)
            median_line = inv_logit_pct(la + LINE_DAMP * lb * x_target)
            lresid = [y - (la + lb * x) for x, y in zip(lx, ly)]
            lrmean = sum(lresid) / len(lresid)
            lresid_var = sum((v - lrmean) ** 2 for v in lresid) / len(lresid)
        else:
            median_line = last_rate
            lresid_var = pooled_resid ** 2
        median_line = min(median_line, TIER_CAP.get(trophy["tier"], 80.0))
        median_line = min(max(median_line, 0.01), 99.99)

        lo_en = logit_pct(median_en)
        lo_line = logit_pct(median_line)
        median = inv_logit_pct(BLEND_LINE * lo_line + (1.0 - BLEND_LINE) * lo_en)

        var = (0.5 * (resid_var + lresid_var) * horizon_factor
               + 0.25 + 0.09 * horizon_days
               + 0.25 * (0.5 * (lo_en - lo_line)) ** 2)
        if distinct_n == 1:
            # No refresh seen in the whole history: one is due, and the
            # first refresh moves every rate a long way (launch night's
            # first refresh cut the tracked base's rates by ~1.5 logit).
            var += 0.81
        if e_last < 10.0:
            # Small counts are noisy: a couple of earners move the rate.
            var += 0.25
        spread = math.sqrt(var)
        rows.append({"id": trophy["id"],
                     "q": quantiles_for(median, spread)})

    rows.sort(key=lambda row: row["id"])
    return rows


def quantiles_for(median_pct, spread):
    centre = logit_pct(median_pct)
    out = []
    prev = 0.01
    for z in Z:
        q = inv_logit_pct(centre + z * spread)
        q = min(max(q, 0.01), 99.99)
        q = max(q, prev)
        out.append(round(q, 4))
        prev = q
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump({"trophies": rows}, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
