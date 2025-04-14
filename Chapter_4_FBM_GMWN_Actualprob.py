import numpy as np
import pandas as pd
from scipy.stats import norm
from math import factorial, exp

np.random.seed(1)

def present_value(cash_flows, time_ids, interest_rate):
    return np.sum(cash_flows / (1 + interest_rate) ** np.array(time_ids))

def GMWB(delta, H, T, prob_file):
    lambda_ = 1.25
    r = 0.03
    alpha = 0.005
    gamma = 0.01
    sigma_j = 0.1
    m = np.exp(alpha + gamma**2 / 2) - 1
    S0 = 100
    w0 = S0
    g = w0 / T
    x = 55
    simulations = 5000

    prob = pd.read_csv(prob_file)
    qx = prob.iloc[0:(110 - x), 1].values
    px = prob.iloc[0:(110 - x), 2].values
    tpx = np.ones(len(px) + 1)
    tpx[1:] = px
    tpx0 = np.cumprod(tpx)

    r_t = np.zeros((simulations, T + 1))
    s_t = np.zeros((simulations, T + 2))
    s_t[:, 0] = w0

    for i in range(1, T + 2):
        for n in range(1, 101):
            mu = (r - delta - m * lambda_) * i - 0.5 * sigma_j ** 2 * (i + i ** (2 * H)) + n * alpha
            sigma = np.sqrt(sigma_j ** 2 * (i + i ** (2 * H)) + n * gamma ** 2)
            term = norm.rvs(loc=mu, scale=sigma, size=simulations) * np.exp(-lambda_ * i) * (lambda_ * i) ** n / factorial(n)
            r_t[:, i - 1] += term

    s_t[:, 1:] = w0 * np.exp(r_t)

    fund = np.zeros((simulations, T + 1))
    fund[:, 0] = w0

    for i in range(T):
        fund[:, i] *= (1 - delta)
        fund[:, i + 1] = fund[:, i] * (s_t[:, i + 1] / s_t[:, i]) - g

    fund = np.where(fund < 0, 0, fund)
    expected_value = np.mean(fund, axis=0)

    times = np.arange(1, T + 1)
    db = expected_value[1:T + 1] * qx[:T] * tpx0[:T]
    guaranteed_annuity = tpx0[1:T + 1] * g

    PV1 = present_value(db, times, r)
    PV2 = present_value(guaranteed_annuity, times, r)
    lb = tpx0[T] * expected_value[T]
    PV = PV1 + PV2 + lb

    return PV

# Input parameters
deltas = np.arange(0.010, 0.021, 0.001)
Hs = [0.6, 0.7, 0.8]
prob_file = "/mnt/data/US_55to65_actual.csv"

X_T10 = np.zeros((len(Hs), len(deltas)))
X_T15 = np.zeros((len(Hs), len(deltas)))

for i, H in enumerate(Hs):
    for j, delta in enumerate(deltas):
        X_T10[i, j] = GMWB(delta, H, T=10, prob_file=prob_file)
        X_T15[i, j] = GMWB(delta, H, T=15, prob_file=prob_file)

# Save to CSV
pd.DataFrame(X_T10).to_csv("/mnt/data/GMWB_EPV_actualprob_T10.csv", index=False)
pd.DataFrame(X_T15).to_csv("/mnt/data/GMWB_EPV_actualprob_T15.csv", index=False)

