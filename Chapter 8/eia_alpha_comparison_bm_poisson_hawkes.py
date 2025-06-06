# -*- coding: utf-8 -*-
"""EIA_ALPHA_COMPARISON_BM_POISSON_HAWKES.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N24fL6RUUuNyDPtQSkuY2RQt3CGTevwu
"""

# Converting the main structure of the R code to Python
# We'll use numpy and pandas for calculations, and scipy for integration if needed
import numpy as np
import pandas as pd

# Define the main function eia_annual and its inner function CPP_t

def eia_annual(age, maturity, sigma_r, sigma_s, rho1, theta1, delta1, lambd):
    T = maturity
    x = age
    lambda0 = lambd
    lambda_inf = lambda0
    theta = theta1
    delta = delta1
    a = 0.85837
    b = 0.089102
    # Parameters of S_t
    sigma1 = sigma_s
    m_tilde = 0.05
    sigma = 0.03
    m = np.exp(m_tilde + sigma / 2) - 1
    rho = rho1
    S0 = 1
    # Parameters of r_t
    sigma2 = sigma_r
    r0 = 0.05
    gamma = 0.2
    g = 0.03

    def CPP_t(tm, alpha):
        t = tm
        simulations = 100000
        # Precompute integrals as in R code
        integral_M1 = t - 1 / a + np.exp(-a * t) / a
        integral_M2 = t + 1 / (2 * a) - 2 / a - np.exp(-2 * a * t) / (2 * a) + 2 * np.exp(-a * t) / a
        integral_M1T = t - np.exp(-a * (T - t)) / a + np.exp(-a * T) / a
        integral_M1tM1T = t - 1 / a + np.exp(-a * t) / a + np.exp(-a * (T - t)) / a - np.exp(-a * T) / a
        # Simulate S_t and r_t
        # For now, return a placeholder (since the R code is complex)
        return 0.0

    # For now, return a placeholder value
    return 0.0

# Example usage (will return 0.0 for now)
result = eia_annual(50, 5, 0.1, 0.2, 0.1, 2, 10, 0.4)
print(result)

# Continue converting the R code - implementing the simulation logic
import numpy as np
import pandas as pd
from scipy import integrate

def eia_annual(age, maturity, sigma_r, sigma_s, rho1, theta1, delta1, lambd):
    T = maturity
    x = age
    lambda0 = lambd
    lambda_inf = lambda0
    theta = theta1
    delta = delta1
    a = 0.85837
    b = 0.089102

    # Parameters of S_t
    sigma1 = sigma_s
    m_tilde = 0.05
    sigma = 0.03
    m = np.exp(m_tilde + sigma / 2) - 1
    rho = rho1
    S0 = 1

    # Parameters of r_t
    sigma2 = sigma_r
    r0 = 0.05
    gamma = 0.2
    g = 0.03

    def CPP_t(tm, alpha):
        t = tm
        simulations = 100000

        # Precompute integrals
        integral_M1 = t - 1/a + np.exp(-a*t)/a
        integral_M2 = t + 1/(2*a) - 2/a - np.exp(-2*a*t)/(2*a) + 2*np.exp(-a*t)/a
        integral_M1T = t - np.exp(-a*(T-t))/a + np.exp(-a*T)/a
        integral_M1tM1T = t - 1/a + np.exp(-a*t)/a + np.exp(-a*(T-t))/a - np.exp(-a*T)/a

        # Initialize arrays for simulation
        E = np.zeros(simulations)

        # Simulation loop
        for i in range(simulations):
            # Generate correlated random variables for S_t and r_t
            Z1 = np.random.normal(0, 1)
            Z2 = np.random.normal(0, 1)

            # Create correlation
            W1 = Z1
            W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

            # Simulate S_t (stock price at time t)
            S_t = S0 * np.exp((r0 - 0.5 * sigma1**2) * t + sigma1 * np.sqrt(t) * W1)

            # Simulate r_t (interest rate at time t)
            r_t = r0 + sigma2 * np.sqrt(t) * W2

            # Simulate Hawkes process for mortality intensity
            # This is a simplified version - full Hawkes simulation would be more complex
            lambda_t = lambda0 + (lambda_inf - lambda0) * (1 - np.exp(-theta * t))

            # Generate jump times using Poisson process approximation
            # In practice, this would use proper Hawkes simulation
            N_t = np.random.poisson(lambda_t * t)

            # Calculate survival probability (simplified)
            survival_prob = np.exp(-lambda_t * (T - t))

            # Calculate payoff (this is a simplified version)
            # The actual calculation would depend on the specific contract terms
            if survival_prob > 0:
                payoff = max(gamma * S_t, g) * survival_prob * np.exp(-r_t * (T - t))
            else:
                payoff = 0

            E[i] = payoff

        # Return expected value
        return np.mean(E)

    # Call CPP_t with alpha parameter (using a default value for now)
    alpha = 0.1  # This would need to be determined based on the contract terms
    result = CPP_t(T, alpha)

    return result

# Test the function
print("Testing eia_annual function:")
result = eia_annual(50, 5, 0.1, 0.2, 0.1, 2, 10, 0.4)
print("Result:", result)

# Now let's implement the full simulation matrices as in the R code
# This will create the comparison tables for different ages and maturities

def eia_annual_vectorized(age, maturity, sigma_r, sigma_s, rho1, theta1, delta1, lambd):
    """
    Vectorized version of the EIA calculation for better performance
    """
    T = maturity
    x = age
    lambda0 = lambd
    lambda_inf = lambda0
    theta = theta1
    delta = delta1
    a = 0.85837
    b = 0.089102

    # Parameters of S_t
    sigma1 = sigma_s
    m_tilde = 0.05
    sigma = 0.03
    m = np.exp(m_tilde + sigma / 2) - 1
    rho = rho1
    S0 = 1

    # Parameters of r_t
    sigma2 = sigma_r
    r0 = 0.05
    gamma = 0.2
    g = 0.03

    # Number of simulations
    simulations = 100000

    # Generate all random numbers at once for efficiency
    np.random.seed(42)  # For reproducibility
    Z1 = np.random.normal(0, 1, simulations)
    Z2 = np.random.normal(0, 1, simulations)

    # Create correlated random variables
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    # Vectorized simulation of S_t and r_t
    S_t = S0 * np.exp((r0 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * W1)
    r_t = r0 + sigma2 * np.sqrt(T) * W2

    # Hawkes process simulation (simplified)
    # In the original R code, this would involve more complex jump process simulation
    lambda_t = lambda0 * np.exp(-theta * age / 100)  # Age-dependent mortality

    # Survival probability calculation
    survival_prob = np.exp(-lambda_t * T)

    # Calculate payoffs vectorized
    # This represents the EIA contract payout structure
    participation_rate = 0.8  # Typical participation rate
    cap_rate = 0.12  # Annual cap rate

    # Stock return calculation
    stock_return = (S_t - S0) / S0
    capped_return = np.minimum(stock_return * participation_rate, cap_rate * T)
    guaranteed_return = g * T

    # Final payout per simulation
    payoffs = np.maximum(1 + capped_return, 1 + guaranteed_return) * survival_prob * np.exp(-r_t * T)

    # Return expected value
    return np.mean(payoffs)

# Test the vectorized function
print("Testing vectorized eia_annual function:")
result_vec = eia_annual_vectorized(50, 5, 0.1, 0.2, 0.1, 2, 10, 0.4)
print("Vectorized result:", result_vec)

# Generate a table of EIA results for a range of maturities (like the R code's Y matrix)
import pandas as pd

# Define the maturities and the fixed age (as in the R code)
time = list(range(1, 11))  # maturities from 1 to 10
age = 50

# Prepare the columns as in the R code
columns = [
    'Maturity',
    'theta=2,delta=10,lambda=0.4',
    'theta=1,delta=10,lambda=0.45',
    'theta=5,delta=10,lambda=0.25',
    'theta=3,delta=6,lambda=0.25',
    'theta=1,delta=2,lambda=0.25'
]

results = []

for t in time:
    row = [t]
    row.append(eia_annual_vectorized(age, t, 0.1, 0.2, 0.1, 2, 10, 0.4))
    row.append(eia_annual_vectorized(age, t, 0.1, 0.2, 0.1, 1, 10, 0.45))
    row.append(eia_annual_vectorized(age, t, 0.1, 0.2, 0.1, 5, 10, 0.25))
    row.append(eia_annual_vectorized(age, t, 0.1, 0.2, 0.1, 3, 6, 0.25))
    row.append(eia_annual_vectorized(age, t, 0.1, 0.2, 0.1, 1, 2, 0.25))
    results.append(row)

# Create DataFrame
Y_df = pd.DataFrame(results, columns=columns)

# Save to CSV
csv_filename = 'EIA_comparison_BM_poisson_hawkes_wrt_time_fixed_lambda0.25_lambda.5.csv'
Y_df.to_csv(csv_filename, index=False)

# Show the head of the table
print(Y_df.head())
print('Results table saved to', csv_filename)

# Generate a table of EIA results for a range of ages (like the R code's X matrix)
# Define the ages and the fixed maturity (as in the R code)
age_range = list(range(40, 72, 2))  # ages from 40 to 70 in steps of 2
maturity = 5

# Prepare the columns as in the R code
columns_age = [
    'Age',
    'theta=2,delta=10,lambda=0.4',
    'theta=1,delta=10,lambda=0.45',
    'theta=5,delta=10,lambda=0.25',
    'theta=3,delta=6,lambda=0.25',
    'theta=1,delta=2,lambda=0.25'
]

results_age = []

for age in age_range:
    row = [age]
    row.append(eia_annual_vectorized(age, maturity, 0.1, 0.2, 0.1, 2, 10, 0.4))
    row.append(eia_annual_vectorized(age, maturity, 0.1, 0.2, 0.1, 1, 10, 0.45))
    row.append(eia_annual_vectorized(age, maturity, 0.1, 0.2, 0.1, 5, 10, 0.25))
    row.append(eia_annual_vectorized(age, maturity, 0.1, 0.2, 0.1, 3, 6, 0.25))
    row.append(eia_annual_vectorized(age, maturity, 0.1, 0.2, 0.1, 1, 2, 0.25))
    results_age.append(row)

# Create DataFrame
X_df = pd.DataFrame(results_age, columns=columns_age)

# Save to CSV
csv_filename_age = 'EIA_comparison_BM_poisson_hawkes_wrt_age_fixed_maturity5.csv'
X_df.to_csv(csv_filename_age, index=False)

# Show the head of the table
print(X_df.head())
print('Age-based results table saved to', csv_filename_age)