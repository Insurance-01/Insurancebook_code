import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

# Load interest rates
rate_df = pd.read_csv("C:/Users/admin/Desktop/usa interest rate yearly.csv")
r = rate_df.iloc[:50, 2].values.reshape(-1, 1)

# Value function part 1 (value1)
def value1(d, grate):
    mortality = pd.read_csv("C:/Users/admin/Desktop/shakti//Users/shaktisingh/Desktop/Paper 4 literature, codes and data/new mortality rates of usa.csv")
    s0 = 100
    qx = mortality.iloc[:599, 1].values
    px = mortality.iloc[:599, 2].values
    x = age = 1  # use x=1 for age 65

    l = 599 - x
    tpx = np.cumprod(px[x:])
    tdeferredqx = np.zeros(l)
    tdeferredqx[0] = qx[x]
    tdeferredqx[1:] = tpx[:-1] * qx[x+1:]

    A = np.array([
        [0.9001, 0.0241, 0.0192, 0.0059, 0.0056, 0.0047],
        [0.3069, 0.3153, 0.1570, 0.0455, 0.0428, 0.0219],
        [0.1625, 0.1334, 0.3596, 0.1050, 0.0605, 0.0294],
        [0.0719, 0.0469, 0.2822, 0.2310, 0.1319, 0.0474],
        [0.0702, 0.0577, 0.0875, 0.0905, 0.3298, 0.0586],
        [0.0754, 0.0097, 0.0163, 0.0230, 0.0012, 0.7222]
    ])

    grate = grate * (s0 / 100)

    def G(j):
        if j <= 1:
            return 2 + grate
        elif j <= 2:
            return 2 + grate
        elif j <= 3:
            return 3 + grate
        elif j <= 4:
            return 3 + grate
        elif j <= 5:
            return 4 + grate
        else:
            return 4 + grate

    LB = 0
    for k in range(1, 24):
        for j in range(6):
            LB += G(j) * A[0, j] * s0 * np.exp(-r[2*k, 0] * k * 24)
    return LB

# Value function part 2 (value2)
def value2(d, grate, u):
    prob = pd.read_csv("C:/Users/admin/Desktop/shakti/new mortality rates of usa.csv")
    s0 = 100
    qx = prob.iloc[1:23, 1].values
    px = prob.iloc[1:23, 2].values
    x = age = 1  # age 65

    l = 23 - x
    tpx = np.cumprod(px[x:])
    tdeferredqx = np.zeros(l)
    tdeferredqx[0] = qx[x]
    tdeferredqx[1:] = tpx[:-1] * qx[x+1:]

    A = np.array([
        [0.9001, 0.0241, 0.0192, 0.0059, 0.0056, 0.0047],
        [0.3069, 0.3153, 0.1570, 0.0455, 0.0428, 0.0219],
        [0.1625, 0.1334, 0.3596, 0.1050, 0.0605, 0.0294],
        [0.0719, 0.0469, 0.2822, 0.2310, 0.1319, 0.0474],
        [0.0702, 0.0577, 0.0875, 0.0905, 0.3298, 0.0586],
        [0.0754, 0.0097, 0.0163, 0.0230, 0.0012, 0.7222]
    ])

    grate = grate * (s0 / 100)

    DB = 0
    for j in range(1, 24):
        DB += s0 * np.exp((u - d / 10000) * 24 * j) * tdeferredqx[j - 1]
    return DB

# Total value function
def value(d, grate, u):
    return value1(d, grate) + value2(d, grate, u)
#Unit root finding 
grate_values = np.arange(1, 1.4, 0.001)
u = 0.0008
root65 = []

for g in grate_values:
    v0 = value(0, g, u)
    v1 = value(1000, g, u)
    if v0 > 100 and v1 < 100:
        result = root_scalar(lambda x: value(x, g, u) - 100, bracket=[0, 1000])
        if result.converged:
            root65.append((g, result.root))

root65 = np.array(root65)
print(root65)
