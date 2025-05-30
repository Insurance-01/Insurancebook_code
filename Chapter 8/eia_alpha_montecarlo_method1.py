# -*- coding: utf-8 -*-
"""EIA_alpha_montecarlo_method1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1947dscYfWyoDJCRNZnjfm2H8oC16DGKj
"""

# Import required libraries
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, lognorm
from scipy.optimize import root_scalar

def simulate_hawkes(lambda0, theta, delta, T):
    # Simulate Hawkes process jump times up to time T
    # Ogata's thinning algorithm (simplified for exponential kernel)
    jump_times = []
    t = 0
    lambda_t = lambda0
    while t < T:
        u = np.random.uniform()
        w = -np.log(u) / lambda_t
        t += w
        if t >= T:
            break
        d = np.random.uniform()
        if d <= lambda_t / (lambda0 + theta):
            jump_times.append(t)
            lambda_t += theta
        lambda_t = lambda0 + (lambda_t - lambda0) * np.exp(-delta * w)
    return np.array(jump_times)

def CPP_t(time, alpha, N, a, b, sigma1, sigma2, rho, m, sigma, dt, lambda0, theta, delta, gamma, g):
    t = time
    integral_M2 = t + 1/(2*a) - 2/a - np.exp(-2*a*t)/(2*a) + 2*np.exp(-a*t)/a
    E = np.zeros(N)
    for j in range(N):
        times = np.arange(0, t + dt, dt)
        X = simulate_hawkes(lambda0, theta, delta, t)
        TJ = X
        n = len(TJ)
        k = 0
        l = np.zeros(len(times))
        l[0] = lambda0
        r = np.zeros(len(times))
        s = np.zeros(len(times))
        r[0] = 0.05
        s[0] = 1
        for i in range(len(times) - 1):
            cov = np.array([[1, rho], [rho, 1]])
            ex = np.random.multivariate_normal([0, 0], cov)
            r[i+1] = r[i] + a * (b - r[i]) * dt + sigma2 * np.sqrt(dt) * ex[0]
            l[i+1] = delta * (lambda0 - l[i]) * dt + l[i]
            s[i+1] = s[i] + s[i] * (r[i] * dt + sigma1 * np.sqrt(dt) * ex[1] - l[i] * dt * m)
            # Handle jumps
            while k < n and TJ[k] >= times[i] and TJ[k] <= times[i+1]:
                l[i+1] += theta
                z = lognorm(s=sigma, scale=np.exp(m)).rvs() - 1
                s[i+1] += s[i] * z
                k += 1
        fund = s[::int(1/dt)]
        fund = np.log(fund)
        E_T = max(min(np.exp(alpha * fund[-1]), np.exp(gamma * t)), np.exp(g * t))
        E[j] = E_T
    X_T = np.exp(-b * t - (0.05 - b) * (1 - np.exp(-a * t)) / a + sigma2**2 * integral_M2 / (2 * a**2))
    mean_E = np.mean(E)
    return mean_E, X_T

def eia_annual(age, maturity, sigma_r, sigma_s, rho1, theta1, delta1, N=10000):
    # Parameters
    gamma = 0.2
    g = 0.03
    lambda0 = 0.5
    theta = theta1
    delta = delta1
    sigma1 = sigma_s
    a = 0.85837
    b = 0.089102
    sigma2 = sigma_r
    rho = rho1
    m = 0.05
    sigma = 0.03
    dt = 1/252
    # Load mortality table (assume file is in current directory)
    prob = pd.read_csv('/content/china_mortality_2010_13.csv')
    qx = prob.iloc[:, 1].values
    px = prob.iloc[:, 2].values
    x = int(age)
    tpx = np.cumprod(px[(x+1):110])
    def Ppp(alpha):
        T1 = maturity
        t = np.arange(1, T1+1)
        val = CPP_t(1, alpha, N, a, b, sigma1, sigma2, rho, m, sigma, dt, lambda0, theta, delta, gamma, g)
        P_pp = val[0] * val[1] * qx[x+1]
        if T1 > 1:
            for i in range(1, len(t)):
                val = CPP_t(t[i], alpha, N, a, b, sigma1, sigma2, rho, m, sigma, dt, lambda0, theta, delta, gamma, g)
                P_pp += val[0] * val[1] * qx[x+t[i]] * tpx[t[i-1]-1]
        value = P_pp + CPP_t(T1, alpha, N, a, b, sigma1, sigma2, rho, m, sigma, dt, lambda0, theta, delta, gamma, g)[0] *CPP_t(T1, alpha, N, a, b, sigma1, sigma2, rho, m, sigma, dt, lambda0, theta, delta, gamma, g)[1] * tpx[T1-1] - 1
        return value
    sol = root_scalar(Ppp, bracket=[0, 2], method='bisect')
    return sol.root

results = []
results.append(eia_annual(50, 7, 0.05, 0.2, 0.1, 0, 1))
results.append(eia_annual(50, 7, 0.05, 0.3, 0.1, 0, 1))
results.append(eia_annual(50, 5, 0.05, 0.2, 0.1, 0, 1))
results.append(eia_annual(50, 5, 0.05, 0.3, 0.1, 0, 1))
results.append(eia_annual(40, 7, 0.05, 0.2, 0.1, 0, 1))
results.append(eia_annual(40, 7, 0.05, 0.3, 0.1, 0, 1))
results.append(eia_annual(40, 5, 0))