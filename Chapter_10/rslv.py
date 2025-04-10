import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# === Parameters ===
S0 = 100.0  # initial stock price
T = 1.0     # maturity
N = 252     # time steps
dt = T / N
M = 10000   # number of MC paths
regimes = [0, 1]
Q = np.array([[-0.1, 0.1], [0.05, -0.05]])  # generator matrix
r_regime = [0.02, 0.04]  # interest rates for regimes
sigma_regime = [0.2, 0.3]  # initial guesses for regime volatilities

# === Simulate Regime Process ===
def simulate_regimes(T, dt, Q, M):
    states = np.zeros((M, int(T / dt) + 1), dtype=int)
    for m in range(M):
        state = 0
        for t in range(1, int(T / dt) + 1):
            if np.random.rand() < -Q[state, state] * dt:
                state = 1 - state
            states[m, t] = state
    return states

# === Local volatility function (can be splines, NN, etc.) ===
def local_volatility(t, S, regime, params):
    # Example: power-law local volatility
    a, b = params[regime]
    return a * S**b

# === MC Simulation for given params ===
def simulate_paths(S0, T, N, M, Q, r_regime, lv_params):
    dt = T / N
    S = np.zeros((M, N + 1))
    S[:, 0] = S0
    regimes = simulate_regimes(T, dt, Q, M)

    for i in range(N):
        t = i * dt
        Z = np.random.randn(M)
        for m in range(M):
            reg = regimes[m, i]
            r = r_regime[reg]
            sigma = local_volatility(t, S[m, i], reg, lv_params)
            S[m, i + 1] = S[m, i] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[m])
    return S, regimes

# === Pricing function (e.g., European Call) ===
def price_option_mc(S_paths, K):
    payoff = np.maximum(S_paths[:, -1] - K, 0)
    return np.mean(payoff)

# === Objective function to minimize ===
def calibration_objective(flat_params, market_prices, K_values):
    # unpack parameters
    lv_params = [flat_params[:2], flat_params[2:]]  # [a0, b0], [a1, b1]
    S_paths, _ = simulate_paths(S0, T, N, M, Q, r_regime, lv_params)
    model_prices = np.array([price_option_mc(S_paths, K) for K in K_values])
    return np.sum((model_prices - market_prices) ** 2)

# === Market Data (example) ===
K_values = np.array([90, 100, 110])
market_prices = np.array([12.0, 8.0, 5.0])  # sample market prices

# === Initial guess for [a0, b0, a1, b1] ===
init_guess = [0.2, 0.0, 0.3, 0.0]

# === Calibrate using optimization ===
res = minimize(calibration_objective, init_guess, args=(market_prices, K_values), method='Nelder-Mead')
opt_params = res.x
lv_params_opt = [opt_params[:2], opt_params[2:]]

print("Optimized local volatility parameters:")
print("Regime 0: a = {:.4f}, b = {:.4f}".format(*lv_params_opt[0]))
print("Regime 1: a = {:.4f}, b = {:.4f}".format(*lv_params_opt[1]))
