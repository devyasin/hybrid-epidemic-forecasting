# Hybrid mechanistic–ML epidemic forecasting: case study code

Code for the case study in the chapter
*Hybrid Mechanistic–Machine Learning Models for Epidemiological Forecasting: A Practical Guide Built on Compartmental Models*
(Yasin Ucakan, İstanbul Rumeli University).

## What it does
- Simulates a 260-day SEIR epidemic with a time-varying transmission rate (a small first wave and a second wave
  that peaks and declines through susceptible depletion) and negative-binomial reporting of the daily E→I flow.
- Fits seven forecasters and two ensembles at six forecast origins (days 60, 90, 105, 175, 190, 205; 21-day horizon)
  on 50 noisy realisations: naive, 14-day log-linear trend, constant-β SEIR, piecewise-constant-β SEIR,
  ridge autoregression, gradient-boosted autoregression, hybrid SEIR whose β(t) is a small neural network
  (universal differential equation), a hybrid + gradient-boosting ensemble and a median ensemble of all seven.
- Refits the hybrid model with a wrong latent period, a wrong reporting fraction, a wrong infectious period,
  by penalised negative binomial likelihood (penalty rescaled to the deviance scale), and with two other
  time scales T of the network input.
- Scores point forecasts (MAE) and probabilistic forecasts (weighted interval score and its decomposition,
  50%/90% coverage pooled and by horizon); intervals for all models are calibrated with a common pseudo-origin protocol.
  Relative scores are reported as ratios of pooled scores and as geometric means of origin-specific ratios.
- Compares five rules for extending the learned β(t) (including an oracle) and repeats the comparison for latent
  periods of 2, 5 and 12 days (150-day window, 30 realisations).

Information given to the SEIR-based models (not estimated): N, ε (latent rate), γ, ρ, the ratio I(0)/E(0) = 0.4 and the seeding
time t = 0; the negative binomial variant and the observation-noise term of all predictive distributions
also use k = 50. The network input is t / T with T = 100 days (sensitivity runs: T = 150, 260; earlier versions used T = 260). The optimiser starts at E(0) = 10 (true value 50).

## Run
```
pip install -r requirements.txt
python3 experiment.py            # everything
python3 experiment.py main       # main study only
python3 experiment.py scenarios  # latent-period scenarios only
```
Runtime depends on the hardware: with two parallel workers the main study took about 1.5 to 4 hours on the two computers tested, and the latent-period scenarios about half as long. Results may differ in the last digits across platforms (floating-point arithmetic); the values reported in the chapter are unaffected.
Outputs: `results.json`, `scenario_results.json`, `results_table.tex`, `prob_table.tex`, `calib_table.tex`,
`variant_table.tex`, `sens_table.tex`, `scenario_table.tex`, `fig_data.pdf`, `fig_forecast.pdf`.
All seeds are fixed (realisation r uses seed 1000 + r). `numba` is optional and only speeds up the ODE solver;
results are identical without it.

Tested with Python 3.13 and the versions pinned in `requirements.txt`.

## Licence
MIT
