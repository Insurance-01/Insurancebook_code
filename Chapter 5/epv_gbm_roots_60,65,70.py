# -*- coding: utf-8 -*-
"""EPV_GBM_ROOTS_60,65,70.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gO8-BTrgqG33d5dyN8Ko_uccBawzS_5Q
"""

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from numpy.random import normal
import math

def GBM(x0, r, sigma, T, N):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    W = np.cumsum(np.sqrt(dt) * normal(size=N))
    W = np.insert(W, 0, 0)  # include W_0 = 0
    S = x0 * np.exp((r - 0.5 * sigma ** 2) * t + sigma * W)
    return S

def present_value(cashflows, time_ids, interest_rate):
    return np.sum(np.array(cashflows) / (1 + interest_rate) ** np.array(time_ids))

def value(d, grate, age, rate, mortality_path, equity_path):
    N = 10000
    s0 = 100

    # Load data
    prob = pd.read_csv(mortality_path)
    stock_data = pd.read_csv(equity_path)
    stock_prices = stock_data.iloc[:, 1].dropna().values
    returns = np.diff(np.log(stock_prices))
    v = np.var(returns)

    qx = prob.iloc[:110, 1].values
    px = prob.iloc[:110, 2].values

    x = age
    l = 110 - x
    tpx = np.cumprod(px[(x + 1):])
    tdeferred_qx = np.zeros(l)
    tdeferred_qx[0] = qx[x + 1]
    min_len = min(len(tpx[:-1]), len(qx[(x + 2):]))
    tdeferred_qx[1:1+min_len] = tpx[:-1][:min_len] * qx[(x + 2):(x + 2 + min_len)]


    irate = rate / 100
    cint = np.log(1 + irate)
    n = 50 * 252

    stock = np.zeros((n + 1, N))
    for i in range(N):
        stock[:, i] = GBM(x0=100, r=cint / 252, sigma=np.sqrt(v), T=50, N=n)

    stock = stock[::252, :]  # yearly values
    g = grate * s0 / 100
    fund = np.zeros((l, N))
    fund[0, :] = stock[0, :]

    for i in range(1, l):
        fund[i - 1, :] *= (1 - d / 10000)
        fund[i, :] = fund[i - 1, :] * (stock[i, :] / stock[i - 1, :]) - g

    fund = np.where(fund < 0, 0, fund)
    expected_value = np.mean(fund, axis=1)[:l] * tdeferred_qx[:l]
    guaranteed_annuity = tpx[:l] * g
    times = np.arange(1, l + 1)

    PV1 = present_value(expected_value, times, irate)
    PV2 = present_value(guaranteed_annuity, times, irate)
    return PV1 + PV2

def find_roots_for_age(age, grate_range, mortality_path, equity_path, filename):
    results = []
    for g in grate_range:
        val0 = value(0, g, age, 3, mortality_path, equity_path)
        val1 = value(1000, g, age, 3, mortality_path, equity_path)

        if val0 > 100 and val1 < 100:
            sol = root_scalar(lambda d: value(d, g, age, 3, mortality_path, equity_path) - 100,
                              bracket=[0, 1000], method='brentq')
            if sol.converged:
                results.append((g, sol.root))

    return np.array(results)

# Change these to your CSV paths in Colab
mortality_path = '/content/us_combined_mortality.csv'  # or japan_combined_mortality.csv
nikkei_path = '/content/Nikkei_1970_2019.csv'
snp_path = '/content/S&P_composite_1970_2019.csv'
msci_path = '/content/MSCIworld.csv'

ages = [60, 65, 70]
grate_ranges = {
    60: np.arange(5, 6.6, 0.1),
    65: np.arange(5.5, 7.1, 0.1),
    70: np.arange(6, 7.6, 0.1)
}
equity_files = {
    'nikkei': nikkei_path,
    'snp': snp_path,
    'msci': msci_path
}

for label, equity_path in equity_files.items():
    all_roots = []
    for age in ages:
        roots = find_roots_for_age(age, grate_ranges[age], mortality_path, equity_path, '')
        if roots.size:
            age_col = np.full((roots.shape[0], 1), age)
            all_roots.append(np.hstack((age_col, roots)))

    all_roots = np.vstack(all_roots)
    df = pd.DataFrame(all_roots, columns=['Age', 'Grate', 'Root'])
    df.to_csv(f'/content/roots_cors_age_60_65_70_{label}_gbm.csv', index=False)