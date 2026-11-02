"""Illustrative case study: mechanistic, machine-learning and hybrid forecasters
on synthetic SEIR data with time-varying transmission.

Reproducible:  python3 experiment.py            (main study + latent-period scenarios)
               python3 experiment.py main       (main study only)
               python3 experiment.py scenarios  (latent-period scenarios only)
Requirements:  see requirements.txt (numba optional, only for speed)
Outputs:       results.json, scenario_results.json, results_table.tex, prob_table.tex,
               calib_table.tex, variant_table.tex, sens_table.tex, scenario_table.tex,
               fig_data.pdf, fig_forecast.pdf
All random seeds are fixed (realisation r uses seed 1000 + r).

Observation target: the daily E->I transition flow, rho * int_d^{d+1} eps E(s) ds (eps = latent rate epsilon), used as a
proxy for newly reportable cases (not the S->E infection incidence).

Information given to the SEIR-based models (not estimated): N, eps (latent rate epsilon; variable SIGMA in the code), gamma, rho, the ratio
I(0)/E(0) = 0.4 used to generate the data, and the seeding time t = 0; the negative binomial
variant and the observation-noise term of the predictive distribution also use the dispersion
k = 50. The network input is t / T with T = 100 days (earlier runs used the series length, 260). The optimiser starts at E(0) = 10 (true value 50).
"""
import json, sys, time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
except ImportError:  # identical results, only slower
    def njit(f=None, **kw):
        return f if f is not None else (lambda g: g)

# ---------------- settings (all fixed before the experiment was run) ----------------
N, SIGMA, GAMMA, RHO, KDISP = 1e6, 1 / 5, 1 / 7, 0.4, 50.0
E0_TRUE, I0_TRUE = 50.0, 20.0
SIGMA_MIS, RHO_MIS, GAMMA_MIS = 1 / 3, 0.3, 1 / 5   # deliberately wrong values (misspecification tests)
I0_RATIO = I0_TRUE / E0_TRUE   # fitted models assume I0 = 0.4 E0 (the value used in the simulation)
LE0_START = np.log(10.0)       # optimiser start for log E0 (true value log 50)
T_TOTAL, H = 260, 21
ORIGINS = [60, 90, 105, 175, 190, 205]   # trough, early growth, growth, before / at / after the peak
N_REP = 50
DT = 0.25
STEPS = int(round(1 / DT))
LAGS = 14
T_SCALE = 100.0                # time scale of the network input t / T_SCALE (days); earlier runs used T_TOTAL
CAL_WIN = 28                   # pseudo-origin offset for the common uncertainty calibration
PW_BLOCK = 14                  # block length (days) of the piecewise-constant beta baseline
TREND_WIN = 14                 # window of the log-linear trend baseline
LE0_LO, LE0_HI = -5.0, 12.0    # bounds of log E0
LAM = 1e-2                     # weight penalty of the hybrid model (log-scale criterion)
QLEVELS = np.array([0.01, 0.025] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.975, 0.99])
ALPHAS = np.array([0.02, 0.05] + [round(0.1 * i, 1) for i in range(1, 10)])  # central intervals
HZ = [1, 7, 14, 21]            # horizons for horizon-specific coverage
# latent-period scenarios use the original 150-day window and origins
T_SCEN, ORIGINS_SCEN, N_REP_SCEN = 150, [60, 75, 90, 105], 30


def sig(x):
    return 1 / (1 + np.exp(-x))


def beta_true(t):
    return 0.45 - 0.34 * sig((t - 30) / 3) + 0.14 * sig((t - 85) / 5)


# ---------------- SEIR solver (RK4, fixed step) ----------------
@njit(cache=True)
def _seir_rk4(bgrid, E0, I0, days, sigma, gamma, n, dt, steps):
    """bgrid holds beta on a half-step grid t = j*dt/2, j = 0..days*2*steps.
    Returns the daily E->I flow over [d, d+1] and S at the END of day d (i.e. S(d+1))."""
    S, E, I = n - E0 - I0, E0, I0
    inc = np.zeros(days)
    Ss = np.zeros(days)
    for d in range(days):
        flow = 0.0
        for k in range(steps):
            j = 2 * (d * steps + k)
            b0, b1, b2 = bgrid[j], bgrid[j + 1], bgrid[j + 2]
            l = b0 * S * I / n
            a0, a1, a2, a3 = -l, l - sigma * E, sigma * E - gamma * I, sigma * E
            S1, E1, I1 = S + dt / 2 * a0, E + dt / 2 * a1, I + dt / 2 * a2
            l = b1 * S1 * I1 / n
            c0, c1, c2, c3 = -l, l - sigma * E1, sigma * E1 - gamma * I1, sigma * E1
            S2, E2, I2 = S + dt / 2 * c0, E + dt / 2 * c1, I + dt / 2 * c2
            l = b1 * S2 * I2 / n
            e0, e1, e2, e3 = -l, l - sigma * E2, sigma * E2 - gamma * I2, sigma * E2
            S3, E3, I3 = S + dt * e0, E + dt * e1, I + dt * e2
            l = b2 * S3 * I3 / n
            g0, g1, g2, g3 = -l, l - sigma * E3, sigma * E3 - gamma * I3, sigma * E3
            S += dt / 6 * (a0 + 2 * c0 + 2 * e0 + g0)
            E += dt / 6 * (a1 + 2 * c1 + 2 * e1 + g1)
            I += dt / 6 * (a2 + 2 * c2 + 2 * e2 + g2)
            flow += dt / 6 * (a3 + 2 * c3 + 2 * e3 + g3)
        inc[d] = flow
        Ss[d] = S
    return inc, Ss


def simulate(beta_fn, E0, I0, days, sigma=SIGMA, gamma=GAMMA):
    """Daily E->I flow and S(d+1). beta_fn must accept arrays."""
    tg = np.arange(days * 2 * STEPS + 1) * (DT / 2)
    bgrid = np.broadcast_to(np.asarray(beta_fn(tg), dtype=float), tg.shape).copy()
    return _seir_rk4(bgrid, float(E0), float(I0), days, sigma, gamma, N, DT, STEPS)


def observe(mu, rng, k=KDISP):
    # negative binomial with mean mu and variance mu + mu^2/k
    p = k / (k + mu)
    return rng.negative_binomial(k, p).astype(float)


# ---------------- probabilistic forecasts and scoring ----------------
def calib_scale(pred, y):
    """Common uncertainty calibration: pred is a CAL_WIN-day multi-step forecast made at the
    pseudo-origin n - CAL_WIN with data up to that point; y are the outcomes that followed.
    Assuming log-scale errors with variance s^2 h + obs_var, the moment estimate is
    s^2 = mean((e_h^2 - obs_var_h) / h), truncated at zero."""
    h = np.arange(1, len(pred) + 1)
    e = np.log1p(np.maximum(pred, 0)) - np.log1p(y)
    return float(np.sqrt(max(np.mean((e ** 2 - obs_var(pred)) / h), 0.0)))


def obs_var(point, k=KDISP):
    """Delta-method variance of log(1+Y) for Y ~ NegBin(mean point, size k): observation noise that
    is present even for a perfect forecast (the dispersion k is given to all models)."""
    m = np.maximum(point, 0)
    return (m + m ** 2 / k) / (1 + m) ** 2


def quantiles(point, s):
    """Predictive quantiles: log(1+Y_h) ~ Normal(log(1+point_h), s^2 h + obs_var(point_h))."""
    h = np.arange(1, len(point) + 1)
    z = norm.ppf(QLEVELS)[None, :]
    sd = np.sqrt(s ** 2 * h + obs_var(point))
    return np.maximum(np.expm1(np.log1p(np.maximum(point, 0))[:, None] + sd[:, None] * z), 0)


def wis_parts(q, y):
    """Weighted interval score (Bracher et al. 2021) per horizon, split into dispersion,
    under-prediction (y above the interval) and over-prediction (y below it)."""
    med = q[:, QLEVELS == 0.5][:, 0]
    K = len(ALPHAS)
    disp = np.zeros(len(y)); under = 0.5 * np.maximum(y - med, 0); over = 0.5 * np.maximum(med - y, 0)
    for a in ALPHAS:
        lo = q[:, np.isclose(QLEVELS, a / 2)][:, 0]
        hi = q[:, np.isclose(QLEVELS, 1 - a / 2)][:, 0]
        disp += a / 2 * (hi - lo)
        under += a / 2 * 2 / a * (y - hi) * (y > hi)
        over += a / 2 * 2 / a * (lo - y) * (y < lo)
    c = 1 / (K + 0.5)
    return c * disp, c * under, c * over


def wis(q, y):
    d, u, o = wis_parts(q, y)
    return float(np.mean(d + u + o))


def covered(q, y, level):
    a = 1 - level
    lo = q[:, np.isclose(QLEVELS, a / 2)][:, 0]
    hi = q[:, np.isclose(QLEVELS, 1 - a / 2)][:, 0]
    return (y >= lo) & (y <= hi)


def safe_inc(inc):
    return np.nan_to_num(np.clip(inc, 0, 10 * N), nan=10 * N)


# ---------------- mechanistic, constant beta ----------------
def fit_seir_const(y, sigma=SIGMA):
    n = len(y)

    def res(th):
        b, lE0 = th
        inc, _ = simulate(lambda t: b + 0 * t, np.exp(lE0), I0_RATIO * np.exp(lE0), n, sigma=sigma)
        return np.log1p(RHO * safe_inc(inc)) - np.log1p(y)

    best = None
    for b0 in (0.2, 0.3, 0.45):
        r = least_squares(res, [b0, LE0_START], bounds=([0.01, LE0_LO], [2, LE0_HI]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def fc_seir_const(th, n, h, sigma=SIGMA):
    b, lE0 = th
    inc, _ = simulate(lambda t: b + 0 * t, np.exp(lE0), I0_RATIO * np.exp(lE0), n + h, sigma=sigma)
    return RHO * inc[n:]


# ---------------- mechanistic, piecewise-constant beta (no neural network) ----------------
PW_LAM = 0.5   # penalty on jumps of log beta between consecutive blocks (random-walk penalty)


def pw_beta(lb, t):
    """Piecewise-constant beta on PW_BLOCK-day blocks; the last block extends to the right."""
    idx = np.minimum((np.asarray(t, dtype=float) // PW_BLOCK).astype(int), len(lb) - 1)
    return np.exp(lb[idx])


def fit_seir_pw(y, sigma=SIGMA):
    n = len(y)
    nb = int(np.ceil(n / PW_BLOCK))

    def res(th):
        lb, lE0 = th[:nb], th[nb]
        inc, _ = simulate(lambda t: pw_beta(lb, t), np.exp(lE0), I0_RATIO * np.exp(lE0), n, sigma=sigma)
        fit = np.log1p(RHO * safe_inc(inc)) - np.log1p(y)
        return np.concatenate([fit, np.sqrt(PW_LAM) * np.diff(lb)])

    th0 = np.append(np.full(nb, np.log(0.3)), LE0_START)
    lo = np.append(np.full(nb, np.log(0.01)), LE0_LO)
    hi = np.append(np.full(nb, np.log(3.0)), LE0_HI)
    r = least_squares(res, th0, bounds=(lo, hi), x_scale="jac", max_nfev=200 * (nb + 2))
    return r.x


def fc_seir_pw(th, n, h, sigma=SIGMA):
    nb = int(np.ceil(n / PW_BLOCK))
    lb, lE0 = th[:nb], th[nb]
    b_ahead = float(np.mean(pw_beta(lb, np.arange(n - 7, n))))   # same 7-day rule as the hybrid
    bf = lambda t: np.where(t < n, pw_beta(lb, t), b_ahead)
    inc, _ = simulate(bf, np.exp(lE0), I0_RATIO * np.exp(lE0), n + h, sigma=sigma)
    return RHO * inc[n:]


# ---------------- hybrid: SEIR with neural transmission (UDE) ----------------
NH = 6
NPAR = 3 * NH + 1
LOGB_LO, LOGB_HI = -8.0, 2.0   # guard: log beta is clipped to [-8, 2] (beta <= 7.4)


TS = [T_SCALE]   # current time scale; changed only for the time-scale sensitivity variants


def mlp_beta(w, t, scale=None):
    """beta(t) = exp(NN(t / T)). Because the weight penalty acts on the weights, T also controls how
    fast beta(t) can change, so it is a hyperparameter; it is set to a round value, not tied to the length of the data."""
    W1, b1, W2, b2 = w[:NH], w[NH:2 * NH], w[2 * NH:3 * NH], w[3 * NH]
    x = np.atleast_1d(np.asarray(t, dtype=float)) / (TS[0] if scale is None else scale)
    out = np.exp(np.clip(np.tanh(np.outer(x, W1) + b1) @ W2 + b2, LOGB_LO, LOGB_HI))
    return out if np.ndim(t) else float(out[0])


def nb_deviance_resid(m, y, k=KDISP):
    """Signed square-root deviance residuals of NegBin(mean m, size k); their sum of squares is
    minus twice the log-likelihood up to a constant (k fixed). With the weight penalty added, the
    fit is a penalised likelihood fit, not an unpenalised MLE."""
    m = np.maximum(m, 1e-8)
    t1 = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / m), 0.0)
    d = 2 * (t1 - (y + k) * np.log((y + k) / (m + k)))
    return np.sign(y - m) * np.sqrt(np.maximum(d, 0))


def hybrid_data_resid(th, y, sigma=SIGMA, rho=RHO, gamma=GAMMA, loss="log"):
    n = len(y)
    w, lE0 = th[:NPAR], th[NPAR]
    inc, _ = simulate(lambda t: mlp_beta(w, t), np.exp(lE0), I0_RATIO * np.exp(lE0), n,
                      sigma=sigma, gamma=gamma)
    inc = safe_inc(inc)
    return nb_deviance_resid(rho * inc, y) if loss == "nb" else np.log1p(rho * inc) - np.log1p(y)


def hybrid_inits(rng, starts=3):
    out = []
    for s in range(starts):
        w0 = rng.normal(0, 1, NPAR)
        w0[3 * NH] = np.log(0.3)
        w0[2 * NH:3 * NH] *= 0.3
        out.append(np.append(w0, LE0_START))
    return out


CONV = {"fits": 0, "converged": 0}


def fit_hybrid(y, inits, lam=LAM, sigma=SIGMA, rho=RHO, gamma=GAMMA, loss="log"):
    """Penalised fit of the network weights and log E0 through the ODE solver; the output-layer
    intercept (baseline level of log beta) is not penalised. The same initial points are used for
    every variant (common random numbers). Returns (theta, converged)."""

    def res(th):
        fit = hybrid_data_resid(th, y, sigma, rho, gamma, loss)
        return np.concatenate([fit, np.sqrt(lam) * th[:3 * NH]])

    lo = np.append(np.full(NPAR, -np.inf), LE0_LO)
    hi = np.append(np.full(NPAR, np.inf), LE0_HI)
    best = None
    for th0 in inits:
        r = least_squares(res, th0, bounds=(lo, hi), max_nfev=400 * (NPAR + 1), x_scale="jac")
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    return best.x, bool(best.status > 0)


def beta_ahead(w, n, rule):
    """How the learned beta(t) is extended from the forecast origin t = n onwards."""
    if rule == "last":
        return lambda t: mlp_beta(w, n - 1)
    if rule == "mean7":
        b = float(np.mean(mlp_beta(w, np.arange(n - 7, n))))
        return lambda t: b
    if rule == "mean14":
        b = float(np.mean(mlp_beta(w, np.arange(n - 14, n))))
        return lambda t: b
    if rule == "trend14":  # log-linear trend of the last 14 estimated values
        tt = np.arange(n - 14, n)
        c = np.polyfit(tt, np.log(mlp_beta(w, tt)), 1)
        return lambda t: np.exp(np.clip(np.polyval(c, t), LOGB_LO, LOGB_HI))
    if rule == "oracle":   # true future transmission (not available in practice; reference)
        return beta_true
    raise ValueError(rule)


def fc_hybrid(th, n, h, rule="mean7", sigma=SIGMA, rho=RHO, gamma=GAMMA):
    w, lE0 = th[:NPAR], th[NPAR]
    ahead = beta_ahead(w, n, rule)
    # observation d is the flow over [d, d+1]; the last observed day is n-1, so the learned
    # beta(t) is used on [0, n) and the extension rule only from the forecast origin t = n
    bf = lambda t: np.where(t < n, mlp_beta(w, t), ahead(t))
    inc, _ = simulate(bf, np.exp(lE0), I0_RATIO * np.exp(lE0), n + h, sigma=sigma, gamma=gamma)
    return rho * inc[n:]


# ---------------- purely data-driven ----------------
def lagmat(z, L):
    X = np.array([z[i - L:i] for i in range(L, len(z))])
    return X, z[L:]


def fc_recursive(model, z, h, L):
    hist = list(z)
    out = []
    for _ in range(h):
        x = np.array(hist[-L:])[None, :]
        nxt = float(model.predict(x)[0])
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)


def fc_ar(make_model, y, h):
    z = np.log1p(y)
    X, t = lagmat(z, LAGS)
    model = make_model().fit(X, t)
    return np.maximum(np.expm1(fc_recursive(model, z, h, LAGS)), 0)


def fc_naive(y, h):
    return np.full(h, y[-7:].mean())


def fc_trend(y, h):
    """Log-linear trend of log(1+y) over the last TREND_WIN days, extended h days."""
    tt = np.arange(TREND_WIN)
    c = np.polyfit(tt, np.log1p(y[-TREND_WIN:]), 1)
    return np.maximum(np.expm1(np.polyval(c, TREND_WIN + np.arange(h))), 0)


ridge = lambda: Ridge(alpha=1.0)
gbm = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=5, random_state=0)

BASE = ["Naive", "Trend-14", "SEIR-const", "SEIR-piecewise", "Ridge-AR", "GBM-AR", "Hybrid SEIR-NN"]
MODELS = BASE + ["Ensemble", "Median ensemble"]
VARIANTS = {  # name: (sigma, rho, gamma, loss, time scale T of the network input)
    "Hybrid, wrong latent period": (SIGMA_MIS, RHO, GAMMA, "log", T_SCALE),
    "Hybrid, wrong reporting fraction": (SIGMA, RHO_MIS, GAMMA, "log", T_SCALE),
    "Hybrid, wrong infectious period": (SIGMA, RHO, GAMMA_MIS, "log", T_SCALE),
    "Hybrid, NegBin likelihood": (SIGMA, RHO, GAMMA, "nb", T_SCALE),
    "Hybrid, time scale T = 150": (SIGMA, RHO, GAMMA, "log", 150.0),
    "Hybrid, time scale T = 260": (SIGMA, RHO, GAMMA, "log", 260.0),
}
TSCALE = {m: v[4] for m, v in VARIANTS.items()}
TSCALE["Hybrid SEIR-NN"] = T_SCALE
EXTRA = list(VARIANTS)
HYB = ["Hybrid SEIR-NN"] + EXTRA
EARLY = 25   # early growth phase (days 0-24) used for the bias of the learned beta and R
RULES = ["last", "mean7", "mean14", "trend14", "oracle"]
INC_TRUE, S_TRUE = simulate(beta_true, E0_TRUE, I0_TRUE, T_TOTAL)
MU = RHO * INC_TRUE


def gmean(a, b):
    """Geometric mean of 1+y, minus 1 (point forecasts and quantiles)."""
    return np.expm1(0.5 * (np.log1p(a) + np.log1p(b)))


def forecast_all(ytr, n, h, rng, sigma=SIGMA, variants=True, base=BASE):
    """Point forecasts of every model from data ytr (length n)."""
    P, TH, info = {}, {}, {}
    P["Naive"] = fc_naive(ytr, h)
    P["Trend-14"] = fc_trend(ytr, h)
    P["SEIR-const"] = fc_seir_const(fit_seir_const(ytr, sigma=sigma), n, h, sigma=sigma)
    P["SEIR-piecewise"] = fc_seir_pw(fit_seir_pw(ytr, sigma=sigma), n, h, sigma=sigma)
    P["Ridge-AR"] = fc_ar(ridge, ytr, h)
    P["GBM-AR"] = fc_ar(gbm, ytr, h)
    inits = hybrid_inits(rng)
    th, ok = fit_hybrid(ytr, inits, sigma=sigma)
    CONV["fits"] += 1; CONV["converged"] += ok
    TH["Hybrid SEIR-NN"] = th
    P["Hybrid SEIR-NN"] = fc_hybrid(th, n, h, sigma=sigma)
    if variants:
        # scale the weight penalty of the likelihood fit by the ratio of the two data terms at the
        # log-scale solution, so that both criteria are regularised to a comparable degree
        ss_log = float(np.sum(hybrid_data_resid(th, ytr) ** 2))
        ss_nb = float(np.sum(hybrid_data_resid(th, ytr, loss="nb") ** 2))
        info["nb_ratio"] = ss_nb / ss_log
        for m, (sg, rh, gm, loss, ts) in VARIANTS.items():
            lam = LAM * info["nb_ratio"] if loss == "nb" else LAM
            TS[0] = ts
            thm, ok = fit_hybrid(ytr, inits, lam=lam, sigma=sg, rho=rh, gamma=gm, loss=loss)
            CONV["fits"] += 1; CONV["converged"] += ok
            TH[m] = thm
            P[m] = fc_hybrid(thm, n, h, sigma=sg, rho=rh, gamma=gm)
            TS[0] = T_SCALE
    P["Ensemble"] = gmean(P["Hybrid SEIR-NN"], P["GBM-AR"])
    P["Median ensemble"] = np.median(np.array([P[m] for m in base]), axis=0)
    return P, TH, info


def quantiles_all(P, Pc, y_after, base=BASE):
    Q = {m: quantiles(P[m], calib_scale(Pc[m], y_after)) for m in P if m not in ("Ensemble", "Median ensemble")}
    Q["Ensemble"] = gmean(Q["Hybrid SEIR-NN"], Q["GBM-AR"])
    Q["Median ensemble"] = np.median(np.array([Q[m] for m in base]), axis=0)
    return Q


def run_rep(rep):
    mu = MU
    rng = np.random.default_rng(1000 + rep)
    y = observe(mu, rng)
    CONV["fits"], CONV["converged"] = 0, 0
    allm = MODELS + EXTRA
    out = {k: {m: {} for m in allm} for k in ("err", "wis", "disp", "under", "over", "cov50", "cov90", "bias")}
    out.update({"covh90": {m: {} for m in allm}, "covh50": {m: {} for m in allm},
                "rule": {r: {} for r in RULES}, "beta_rmse": {m: {} for m in HYB},
                "beta_bias_early": {m: {} for m in HYB}, "r_bias_early": {m: {} for m in HYB},
                "endpoint": {}, "nb_ratio": {}, "gbm_exceeds": {}, "example": {}, "b_end": None})
    gam = {m: (VARIANTS[m][2] if m in VARIANTS else GAMMA) for m in HYB}
    for o in ORIGINS:
        ytr = y[:o]
        mute, yte = mu[o:o + H], y[o:o + H]
        P, TH, info = forecast_all(ytr, o, H, rng)
        out["nb_ratio"][o] = info["nb_ratio"]
        # common uncertainty calibration: refit every model at the pseudo-origin o - CAL_WIN
        p0 = o - CAL_WIN
        Pc, _, _ = forecast_all(y[:p0], p0, CAL_WIN, rng)
        Q = quantiles_all(P, Pc, y[p0:o])
        out["gbm_exceeds"][o] = bool(P["GBM-AR"].max() > ytr.max())
        for r in RULES:
            out["rule"][r][o] = float(np.mean(np.abs(fc_hybrid(TH["Hybrid SEIR-NN"], o, H, rule=r) - mute)))
        for m in allm:
            out["err"][m][o] = float(np.mean(np.abs(P[m] - mute)))
            out["bias"][m][o] = float(np.mean(P[m] - mute))
            d, u, ov = wis_parts(Q[m], yte)
            out["wis"][m][o] = float(np.mean(d + u + ov))
            out["disp"][m][o], out["under"][m][o], out["over"][m][o] = float(d.mean()), float(u.mean()), float(ov.mean())
            c50, c90 = covered(Q[m], yte, 0.5), covered(Q[m], yte, 0.9)
            out["cov50"][m][o], out["cov90"][m][o] = float(c50.mean()), float(c90.mean())
            out["covh50"][m][o] = [float(c50[hh - 1]) for hh in HZ]
            out["covh90"][m][o] = [float(c90[hh - 1]) for hh in HZ]
        tt = np.arange(o)
        w = TH["Hybrid SEIR-NN"][:NPAR]
        # value used for the forecast (mean of the last 7 estimates) relative to the true beta on the
        # first forecast day t = o
        out["endpoint"][o] = float(np.mean(mlp_beta(w, np.arange(o - 7, o))) / beta_true(o) - 1)
        for m in HYB:
            bh = mlp_beta(TH[m][:NPAR], tt, scale=TSCALE[m])
            out["beta_rmse"][m][o] = float(np.sqrt(np.mean((bh - beta_true(tt)) ** 2)))
            out["beta_bias_early"][m][o] = float(np.mean(bh[:EARLY] / beta_true(tt[:EARLY])) - 1)
            # early phase: S ~ N, so R ~ beta / gamma
            out["r_bias_early"][m][o] = float(np.mean((bh[:EARLY] / gam[m]) / (beta_true(tt[:EARLY]) / GAMMA)) - 1)
        if o in (90, 175):
            out["example"][o] = dict(preds={k: v.tolist() for k, v in P.items()},
                                     q05=Q["Hybrid SEIR-NN"][:, np.isclose(QLEVELS, 0.05)][:, 0].tolist(),
                                     q95=Q["Hybrid SEIR-NN"][:, np.isclose(QLEVELS, 0.95)][:, 0].tolist())
        if o == ORIGINS[-1]:
            out["b_end"] = mlp_beta(w, tt).tolist()
    out["y"] = y.tolist()
    out["conv"] = dict(CONV)
    print(f"rep {rep} done", flush=True)
    return out


def fmt(x, d=1):
    return f"{x:.{d}f}"


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def summarise(results, models, origins, naive="Naive"):
    """Pooled and geometric-mean relative scores."""
    A = lambda key, m, o: np.array([r[key][m][o] for r in results])
    S = {}
    for m in models:
        d = {o: dict(mae=float(A("err", m, o).mean()), sd=float(A("err", m, o).std(ddof=1)),
                     wis=float(A("wis", m, o).mean())) for o in origins}
        d["overall_mae"] = float(np.mean([d[o]["mae"] for o in origins]))
        d["overall_wis"] = float(np.mean([d[o]["wis"] for o in origins]))
        S[m] = d
    for m in models:
        d = S[m]
        d["rel_mae"] = d["overall_mae"] / S[naive]["overall_mae"]
        d["rel_wis"] = d["overall_wis"] / S[naive]["overall_wis"]
        d["rel_mae_o"] = {o: d[o]["mae"] / S[naive][o]["mae"] for o in origins}
        d["rel_wis_o"] = {o: d[o]["wis"] / S[naive][o]["wis"] for o in origins}
        d["geo_mae"] = float(np.exp(np.mean(np.log(list(d["rel_mae_o"].values())))))
        d["geo_wis"] = float(np.exp(np.mean(np.log(list(d["rel_wis_o"].values())))))
    return S


def main():
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(2) as pool:
        results = pool.map(run_rep, range(N_REP))
    allm = MODELS + EXTRA
    A = lambda key, m, o: np.array([r[key][m][o] for r in results])
    S = {"models": summarise(results, allm, ORIGINS), "rules": {}}
    MS = S["models"]
    for m in allm:
        MS[m]["cov50"] = float(np.mean([A("cov50", m, o).mean() for o in ORIGINS]))
        MS[m]["cov90"] = float(np.mean([A("cov90", m, o).mean() for o in ORIGINS]))
        MS[m]["covh90"] = {hh: float(np.mean([[r["covh90"][m][o][i] for o in ORIGINS] for r in results]))
                           for i, hh in enumerate(HZ)}
        MS[m]["covh50"] = {hh: float(np.mean([[r["covh50"][m][o][i] for o in ORIGINS] for r in results]))
                           for i, hh in enumerate(HZ)}
        tot = {k: float(np.mean([A(k, m, o).mean() for o in ORIGINS])) for k in ("disp", "under", "over")}
        s = sum(tot.values())
        MS[m]["wis_share"] = {k: v / s for k, v in tot.items()}
        MS[m]["bias_o"] = {o: float(A("bias", m, o).mean()) for o in ORIGINS}
        MS[m]["over_frac_o"] = {o: float(np.mean(A("bias", m, o) > 0)) for o in ORIGINS}
    naive_o = {o: MS["Naive"][o]["mae"] for o in ORIGINS}
    for r in RULES:
        v = {o: float(np.mean([res["rule"][r][o] for res in results])) for o in ORIGINS}
        v["rel_mae"] = float(np.mean([v[o] for o in ORIGINS]) / MS["Naive"]["overall_mae"])
        v["geo_mae"] = float(np.exp(np.mean([np.log(v[o] / naive_o[o]) for o in ORIGINS])))
        S["rules"][r] = v
    S["beta"] = {}
    for m in HYB:
        g = lambda key: {o: np.array([r[key][m][o] for r in results]) for o in ORIGINS}
        br, bb, rb = g("beta_rmse"), g("beta_bias_early"), g("r_bias_early")
        S["beta"][m] = dict(rmse={o: [float(br[o].mean()), float(br[o].std(ddof=1))] for o in ORIGINS},
                            rmse_mean=float(np.mean([br[o].mean() for o in ORIGINS])),
                            rmse_median=float(np.median(np.mean([br[o] for o in ORIGINS], axis=0))),
                            early_bias_mean=float(np.mean([bb[o].mean() for o in ORIGINS])),
                            r_bias_mean=float(np.mean([rb[o].mean() for o in ORIGINS])))
    ep = {o: np.array([r["endpoint"][o] for r in results]) for o in ORIGINS}
    S["endpoint"] = {o: dict(median=float(np.median(ep[o])), lo=float(ep[o].min()), hi=float(ep[o].max()),
                             corr_mae=float(np.corrcoef(ep[o], A("err", "Hybrid SEIR-NN", o))[0, 1]))
                     for o in ORIGINS}
    nbr = np.concatenate([[r["nb_ratio"][o] for o in ORIGINS] for r in results])
    S["nb_ratio"] = dict(median=float(np.median(nbr)), lo=float(nbr.min()), hi=float(nbr.max()))
    per = lambda key, m: np.array([np.mean([r[key][m][o] for o in ORIGINS]) for r in results])
    S["paired"] = {}
    pairs = [("Hybrid SEIR-NN", "SEIR-piecewise"), ("Hybrid SEIR-NN", "Trend-14"), ("Hybrid SEIR-NN", "Ensemble"),
             ("Hybrid SEIR-NN", "Naive"), ("SEIR-piecewise", "Naive"), ("SEIR-piecewise", "Trend-14"), ("Ensemble", "GBM-AR"),
             ("Hybrid SEIR-NN", "Hybrid, wrong reporting fraction"),
             ("Hybrid SEIR-NN", "Hybrid, wrong latent period"),
             ("Hybrid SEIR-NN", "Hybrid, wrong infectious period"),
             ("Hybrid SEIR-NN", "Hybrid, NegBin likelihood"),
             ("Hybrid SEIR-NN", "Hybrid, time scale T = 150"),
             ("Hybrid SEIR-NN", "Hybrid, time scale T = 260")]
    for a, b in pairs:
        for key in ("err", "wis"):
            dlt = per(key, a) - per(key, b)
            S["paired"][f"{a} - {b} [{key}]"] = dict(
                mean=float(dlt.mean()), se=float(dlt.std(ddof=1) / np.sqrt(len(dlt))),
                frac_a_better=float(np.mean(dlt < 0)))
    # paired comparisons restricted to the peak-phase origins
    late = [o for o in ORIGINS if o >= 175]
    perl = lambda key, m: np.array([np.mean([r[key][m][o] for o in late]) for r in results])
    S["paired_late"] = {}
    for a, b in [("Hybrid SEIR-NN", "Trend-14"), ("SEIR-piecewise", "Trend-14"), ("Hybrid SEIR-NN", "SEIR-piecewise"),
                 ("Hybrid SEIR-NN", "Hybrid, time scale T = 260"),
                 ("Hybrid SEIR-NN", "Hybrid, wrong reporting fraction")]:
        dlt = perl("err", a) - perl("err", b)
        S["paired_late"][f"{a} - {b}"] = dict(mean=float(dlt.mean()), se=float(dlt.std(ddof=1) / np.sqrt(len(dlt))),
                                              frac_a_better=float(np.mean(dlt < 0)))
    S["gbm_exceeds_frac"] = {o: float(np.mean([r["gbm_exceeds"][o] for r in results])) for o in ORIGINS}
    S["convergence"] = {k: sum(r["conv"][k] for r in results) for k in ("fits", "converged")}
    S["peak"] = dict(day=int(np.argmax(MU)), value=float(MU.max()),
                     attack_at_peak=float(1 - S_TRUE[int(np.argmax(MU))] / N))
    S["attack_rate"] = float(1 - S_TRUE[-1] / N)
    S["n_rep"] = N_REP
    S["runtime_min"] = (time.time() - t0) / 60
    with open("results.json", "w") as f:
        json.dump(S, f, indent=1)

    # ---------------- tables ----------------
    best = lambda key_fn, models: min(models, key=key_fn)
    oh = " & ".join(str(o) for o in ORIGINS)
    no = len(ORIGINS)
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Mean absolute error (MAE) of 21-day point forecasts against the expected reported cases, by forecast origin (mean over {N_REP} realisations). Origins 60--105 lie in the trough and early second wave, 175, 190 and 205 before, at and after its peak. The last two columns give the MAE relative to the naive baseline, pooled over origins (dominated by the origins with the largest counts) and as the geometric mean of the origin-specific ratios. Bold marks the best model in each column.}}\label{{tab:results}}",
         r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l" + "r" * no + r"rr@{}}", r"\toprule",
         rf" & \multicolumn{{{no}}}{{c}}{{Forecast origin (day)}} & \multicolumn{{2}}{{c}}{{Relative MAE}} \\",
         rf"\cmidrule(lr){{2-{no + 1}}}\cmidrule(l){{{no + 2}-{no + 3}}}",
         rf"Model & {oh} & Pooled & Geom. \\", r"\midrule"]
    bests = {o: best(lambda m: MS[m][o]["mae"], MODELS) for o in ORIGINS}
    bp = best(lambda m: MS[m]["rel_mae"], MODELS); bg = best(lambda m: MS[m]["geo_mae"], MODELS)
    B = lambda s, cond: rf"\textbf{{{s}}}" if cond else s
    for m in MODELS:
        cells = [B(fmt(MS[m][o]["mae"]), m == bests[o]) for o in ORIGINS]
        L.append(f"{m} & " + " & ".join(cells) + f" & {B(fmt(MS[m]['rel_mae'], 2), m == bp)} & {B(fmt(MS[m]['geo_mae'], 2), m == bg)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("results_table.tex", L)

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Probabilistic accuracy: weighted interval score (WIS) of 21-day forecasts against the observed reported cases (lower is better), WIS relative to the naive baseline (pooled and geometric mean over origins; bold: best), and empirical coverage of the central 50\% and 90\% prediction intervals pooled over horizons and origins (nominal values 0.50 and 0.90). All intervals use the common calibration protocol described in the text.}\label{tab:prob}",
         r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l" + "r" * no + r"rrrr@{}}", r"\toprule",
         rf" & \multicolumn{{{no}}}{{c}}{{WIS by origin (day)}} & \multicolumn{{2}}{{c}}{{Rel.\ WIS}} & \multicolumn{{2}}{{c}}{{Coverage}} \\",
         rf"\cmidrule(lr){{2-{no + 1}}}\cmidrule(lr){{{no + 2}-{no + 3}}}\cmidrule(l){{{no + 4}-{no + 5}}}",
         rf"Model & {oh} & Pooled & Geom. & 50\% & 90\% \\", r"\midrule"]
    bp = best(lambda m: MS[m]["rel_wis"], MODELS); bg = best(lambda m: MS[m]["geo_wis"], MODELS)
    for m in MODELS:
        L.append(f"{m} & " + " & ".join(fmt(MS[m][o]["wis"]) for o in ORIGINS)
                 + f" & {B(fmt(MS[m]['rel_wis'], 2), m == bp)} & {B(fmt(MS[m]['geo_wis'], 2), m == bg)}"
                 + f" & {fmt(MS[m]['cov50'], 2)} & {fmt(MS[m]['cov90'], 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("prob_table.tex", L)

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Where the probabilistic scores come from: coverage of the central 90\% prediction interval at horizons of 1, 7, 14 and 21 days (nominal 0.90), and the shares of the WIS due to dispersion (interval width), under-prediction and over-prediction, averaged over origins and realisations.}\label{tab:calib}",
         r"\small", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{90\% coverage at horizon (days)} & \multicolumn{3}{c}{Share of WIS} \\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-8}",
         r"Model & 1 & 7 & 14 & 21 & Dispersion & Under & Over \\", r"\midrule"]
    for m in MODELS:
        sh = MS[m]["wis_share"]
        L.append(f"{m} & " + " & ".join(fmt(MS[m]['covh90'][hh], 2) for hh in HZ)
                 + f" & {100 * sh['disp']:.0f}\\% & {100 * sh['under']:.0f}\\% & {100 * sh['over']:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("calib_table.tex", L)

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Hybrid model variants: point and probabilistic accuracy relative to the naive baseline (geometric mean over origins), coverage of the 90\% interval, root-mean-square error of the learned $\hat\beta(t)$ over the training window, and the mean relative bias of $\hat\beta(t)$ and of the implied reproduction number $\hat\beta(t)/\gamma$ in the early growth phase (days 0--24), averaged over origins and realisations. Wrong values: latent period 3 instead of 5 days; reporting fraction 0.3 instead of 0.4; infectious period 5 instead of 7 days. The NegBin row replaces least squares on the $\log(1+y)$ scale by the negative binomial deviance ($k=50$ fixed) with a correspondingly rescaled weight penalty. The last two rows change only the time scale $T$ of the network input $t/T$ (default 100 days). RMSE is the median over realisations of the RMSE averaged over origins.}\label{tab:variants}",
         r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Variant & Rel.\ MAE & Rel.\ WIS & 90\% cov. & RMSE $\hat\beta$ & $\hat\beta$ bias & $\hat{\mathcal{R}}$ bias \\", r"\midrule"]
    vname = {"Hybrid SEIR-NN": "Correct parameters (reference)", "Hybrid, wrong latent period": "Wrong latent period",
             "Hybrid, wrong reporting fraction": "Wrong reporting fraction",
             "Hybrid, wrong infectious period": "Wrong infectious period",
             "Hybrid, NegBin likelihood": "Penalised NegBin likelihood",
             "Hybrid, time scale T = 150": "Time scale $T=150$ days", "Hybrid, time scale T = 260": "Time scale $T=260$ days"}
    for m in HYB:
        L.append(f"{vname[m]} & {fmt(MS[m]['geo_mae'], 2)} & {fmt(MS[m]['geo_wis'], 2)} & "
                 f"{fmt(MS[m]['cov90'], 2)} & {fmt(S['beta'][m]['rmse_median'], 3)} & "
                 f"{100 * S['beta'][m]['early_bias_mean']:+.0f}\\% & {100 * S['beta'][m]['r_bias_mean']:+.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("variant_table.tex", L)

    names = {"last": r"Last estimated value $\hat\beta(n-1)$", "mean7": "Mean of last 7 days (default)",
             "mean14": "Mean of last 14 days", "trend14": r"Log-linear trend of last 14 days",
             "oracle": r"Oracle: true future $\beta(t)$"}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Sensitivity of the hybrid forecast to the rule used to extend the learned transmission rate from the forecast origin onwards: MAE by origin and relative to the naive baseline (geometric mean over origins). The fitted model is identical in all rows. The oracle row uses the true future transmission rate, which is never available in practice.}\label{tab:sens}",
         r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l" + "r" * no + r"r@{}}", r"\toprule",
         rf"Extension rule & {oh} & Rel. \\", r"\midrule"]
    for r in RULES:
        if r == "oracle":
            L.append(r"\midrule")
        L.append(f"{names[r]} & " + " & ".join(fmt(S['rules'][r][o]) for o in ORIGINS)
                 + f" & {fmt(S['rules'][r]['geo_mae'], 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("sens_table.tex", L)

    # ---------------- figures ----------------
    r0 = results[0]
    days = np.arange(T_TOTAL)
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(days, beta_true(days), "k-", label=r"$\beta(t)$")
    ax0b = ax[0].twinx()
    S_start = np.concatenate([[N - E0_TRUE - I0_TRUE], S_TRUE[:-1]])   # S at the start of each day
    ax0b.plot(days, beta_true(days) / GAMMA * S_start / N, "C3--", label=r"$\mathcal{R}_t$")
    ax0b.axhline(1, color="C3", lw=0.6, ls=":")
    for o in ORIGINS:
        ax[0].axvline(o, color="0.75", lw=0.6, ls="--")
        ax[1].axvline(o, color="0.75", lw=0.6, ls="--")
    ax[0].set_xlabel("day"); ax[0].set_ylabel(r"$\beta(t)$"); ax0b.set_ylabel(r"$\mathcal{R}_t$", color="C3")
    ax[0].set_title("(a) transmission and reproduction number")
    ax[1].plot(days, r0["y"], ".", ms=2.5, color="0.5", label="observed")
    ax[1].plot(days, MU, "k-", lw=1, label="expected")
    ax[1].set_xlabel("day"); ax[1].set_ylabel("reported cases"); ax[1].legend(frameon=False, loc="upper left")
    ax[1].set_title("(b) synthetic reported cases")
    fig.tight_layout(); fig.savefig("fig_data.pdf"); plt.close(fig)

    sty = {"Naive": ("#7f7f7f", (0, (1, 1))), "Trend-14": ("#8c564b", (0, (5, 2))),
           "SEIR-const": ("#1f77b4", "-"), "SEIR-piecewise": ("#17becf", (0, (3, 1, 1, 1))),
           "GBM-AR": ("#ff7f0e", ":"), "Hybrid SEIR-NN": ("#2ca02c", "-")}
    fig = plt.figure(figsize=(7.2, 5.6))
    axs = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 1, 2)]
    for k, (o, xlo) in enumerate([(90, 40), (175, 120)]):
        a = axs[k]
        ex = r0["example"][o]
        a.plot(days[:o + H], r0["y"][:o + H], ".", ms=2.5, color="0.6", label="observed")
        a.plot(days[:o + H], MU[:o + H], "k-", lw=1, label="expected")
        fd = np.arange(o, o + H)
        a.fill_between(fd, ex["q05"], ex["q95"], color="#2ca02c", alpha=0.18, lw=0, label="Hybrid 90% PI")
        for m, (c, ls) in sty.items():
            a.plot(fd, ex["preds"][m], color=c, ls=ls, lw=1.4, label=m)
        a.axvline(o, color="0.5", ls="--", lw=0.6)
        a.set_xlim(xlo, o + H); a.set_ylim(0, None); a.set_xlabel("day"); a.set_ylabel("reported cases")
        a.set_title(f"({'ab'[k]}) 21-day forecasts from day {o}")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
    tt = np.arange(ORIGINS[-1])
    axs[2].plot(tt, beta_true(tt), "k-", label=r"true $\beta(t)$")
    axs[2].plot(tt, r0["b_end"], "C2--", label=r"hybrid estimate $\hat\beta(t)$")
    axs[2].set_xlabel("day"); axs[2].set_ylabel(r"$\beta$"); axs[2].legend(frameon=False, fontsize=8)
    axs[2].set_title(rf"(c) recovered $\beta(t)$, origin day {ORIGINS[-1]}")
    fig.tight_layout(); fig.savefig("fig_forecast.pdf"); plt.close(fig)
    print(json.dumps({k: v for k, v in S.items() if k != "models"}, indent=1))


# ---------------- latent-period scenarios (original 150-day window) ----------------
# Same transmission scenario beta(t); only the mean latent period changes. The initial seed is
# scaled so that the first wave peaks at a similar number of reported cases (about 110 per day),
# so that differences reflect the timing of the dynamics rather than the size of the counts.
SCENARIOS = {"short": (2.0, 0.14), "base": (5.0, 1.0), "long": (12.0, 6.0)}  # latent days, seed factor


def run_rep_scenario(args):
    name, rep = args
    latent, f = SCENARIOS[name]
    sg = 1 / latent
    inc, _ = simulate(beta_true, E0_TRUE * f, I0_TRUE * f, T_SCEN, sigma=sg)
    mu = RHO * inc
    rng = np.random.default_rng(1000 + rep)
    y = observe(mu, rng)
    out = {"err": {m: {} for m in MODELS}, "wis": {m: {} for m in MODELS}, "beta_rmse": {}}
    for o in ORIGINS_SCEN:
        ytr = y[:o]
        mute, yte = mu[o:o + H], y[o:o + H]
        P, TH, _ = forecast_all(ytr, o, H, rng, sigma=sg, variants=False)
        p0 = o - CAL_WIN
        Pc, _, _ = forecast_all(y[:p0], p0, CAL_WIN, rng, sigma=sg, variants=False)
        Q = quantiles_all(P, Pc, y[p0:o])
        for m in MODELS:
            out["err"][m][o] = float(np.mean(np.abs(P[m] - mute)))
            out["wis"][m][o] = wis(Q[m], yte)
        tt = np.arange(o)
        out["beta_rmse"][o] = float(np.sqrt(np.mean((mlp_beta(TH["Hybrid SEIR-NN"][:NPAR], tt) - beta_true(tt)) ** 2)))
    print(f"scenario {name} rep {rep} done", flush=True)
    return out


def scenarios_main():
    from multiprocessing import Pool
    jobs = [(nm, r) for nm in SCENARIOS for r in range(N_REP_SCEN)]
    with Pool(2) as pool:
        res = pool.map(run_rep_scenario, jobs)
    R = {nm: [r for (n2, _), r in zip(jobs, res) if n2 == nm] for nm in SCENARIOS}
    S = {}
    for nm, rs in R.items():
        S[nm] = summarise(rs, MODELS, ORIGINS_SCEN)
        S[nm]["beta_rmse"] = float(np.mean([np.mean(list(r["beta_rmse"].values())) for r in rs]))
        for a in ("Ensemble", "SEIR-piecewise", "Trend-14"):
            d = np.array([np.mean([r["err"]["Hybrid SEIR-NN"][o] for o in ORIGINS_SCEN]) -
                          np.mean([r["err"][a][o] for o in ORIGINS_SCEN]) for r in rs])
            S[nm][f"hyb_minus_{a}"] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                                           frac_hyb_better=float(np.mean(d < 0)))
    with open("scenario_results.json", "w") as f:
        json.dump(S, f, indent=1)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Latent-period scenarios on the original 150-day window (origins 60, 75, 90 and 105): point (MAE) and probabilistic (WIS) accuracy relative to the naive baseline, geometric mean over origins, " + str(N_REP_SCEN) + r" realisations, for a mean latent period of 2, 5 (base case) and 12 days. The transmission scenario $\beta(t)$ is the same in all three. The initial seed is scaled so that the first wave peaks at about 110 reported cases per day. The last row gives the root-mean-square error of the learned $\hat\beta(t)$.}\label{tab:scen}",
         r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Relative MAE} & \multicolumn{3}{c}{Relative WIS} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Model & 2 days & 5 days & 12 days & 2 days & 5 days & 12 days \\", r"\midrule"]
    for m in MODELS:
        vals = [S[nm][m]["geo_mae"] for nm in SCENARIOS] + [S[nm][m]["geo_wis"] for nm in SCENARIOS]
        L.append(f"{m} & " + " & ".join(fmt(v, 2) for v in vals) + r" \\")
    L.append(r"\midrule")
    L.append(r"RMSE of $\hat\beta(t)$ & " + " & ".join(fmt(S[nm]["beta_rmse"], 3) for nm in SCENARIOS)
             + r" & & & \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("scenario_table.tex", L)
    print(json.dumps({nm: {k: v for k, v in S[nm].items() if k.startswith("hyb") or k == "beta_rmse"} for nm in S}, indent=1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "main"):
        main()
    if what in ("all", "scenarios"):
        scenarios_main()
