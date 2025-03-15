import numpy as np
import statsmodels.api as sm
from scipy.stats import t
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('m', type=int, help='An integer to be read')
parser.add_argument('q', type=int, help='An integer to be read')
parser.add_argument('percent', type=float, help='An integer to be read')
args = parser.parse_args()


def equation(a, alpha, k=1):
    return  1 - (1 - 2 * (1 + a * norm.pdf(a) - norm.cdf(a))) ** k  - (alpha)

def solve_for_a(alpha):
    a_low = -10
    a_high = 10
    a = brentq(equation, a_low, a_high, args=(alpha,))
    return a

alpha = args.percent
a = solve_for_a(alpha)
print('a=',a)


countfail = 0
serx = 0
sery = 1
m = args.m
q = args.q
n_samples = q * m + 5
selected_lag_order = 1

np.random.seed(47)
ar_param = np.random.uniform(0.8, 0.95)
ma_param = 0.2
N = 2
hit = 0



def generate_garch11(n, omega, alpha, beta):
    epsilon = np.zeros(n)
    sigma2 = np.zeros(n)

    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * epsilon[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma = np.sqrt(sigma2[t])
        z = np.random.normal()
        epsilon[t] = sigma * z

    return epsilon


omega = 0.1
alpha = 0.1
beta = 0.8



bondtable = np.full(q*m, np.nan, dtype=np.float32)
def bondary_func(m, n, a=2.795):
    part1 = np.sqrt(m)
    part2 = (n - m) / m
    part3 = (n / (n - m)) * (a**2 + np.log(n / (n - m)))
    part4 = np.sqrt(part3)
    result = part1 * part2 * part4
    return result

for n in range(m+1, q*m):
    bondtable[n] = bondary_func(m, n, a)

## update way
for _ in tqdm(range(3000)):

    f_t = np.random.laplace(0, 1, n_samples)
    alphas = np.random.uniform(low=0.1, high=1.0, size=N)
    u_it = np.random.laplace(0, 1, (n_samples, N))
    correlated_errors = f_t[:, np.newaxis] * alphas[np.newaxis, :] + u_it
    series_list = []

    for i in range(N):
        arma_model = sm.tsa.ArmaProcess(ar=np.r_[1, -ar_param], ma=np.r_[1, ma_param])
        series = arma_model.generate_sample(nsample=n_samples, scale=1,
                                            distrvs=lambda size: correlated_errors[:size[0], i])
        series_list.append(series)
    df = (np.array(series_list, dtype=np.float32).T)
    model_VAR = VAR(df)

    results = model_VAR.fit(selected_lag_order)
    residuals = results.resid

    y = residuals[:m, sery]
    x = residuals[:m, serx]
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    beta_m = model.params[1]
    Dm = model.cov_params()[1][1] * m

    for n in range(m + 1, q * m):
        y = residuals[:n, sery]
        x = residuals[:n, serx]
        x_with_const = sm.add_constant(x)
        model_n = sm.OLS(y, x_with_const).fit()
        beta_n = model_n.params[1]
        Zn = n * (Dm) ** (-1 / 2) * (beta_n - beta_m)
        if (np.abs(Zn) > bondtable[n]):
            countfail += 1
            break

print('size test, countfail={}, m={}, q={}, a={}'.format(countfail, m, q, a))
with open('Laplace.txt', 'a+') as f:
    f.writelines('size test, countfail={}, m={}, q={}, a={}\n'.format(countfail, m, q, a))


countfail = 0
serx = 0
sery = 1
m = args.m
q = args.q
n_samples = q * m + 5
selected_lag_order = 1

np.random.seed(47)
ar_param = np.random.uniform(0.8, 0.95)
ma_param = 0.2
N = 2
hit = 0


bondtable = np.full(q*m, np.nan, dtype=np.float32)
def bondary_func(m, n, a=2.795):
    part1 = np.sqrt(m)
    part2 = (n - m) / m
    part3 = (n / (n - m)) * (a**2 + np.log(n / (n - m)))
    part4 = np.sqrt(part3)
    result = part1 * part2 * part4
    return result

for n in range(m+1, q*m):
    bondtable[n] = bondary_func(m, n, a)
## update way
for _ in tqdm(range(3000)):

    f_t1 = np.random.laplace(0, 1, int(1.1 * m))
    alphas1 = np.array([0.2711117, 0.92084525])
    u_it1 = np.random.laplace(0, 1, (int(1.1 * m), N))
    correlated_errors1 = f_t1[:, np.newaxis] * alphas1[np.newaxis, :] + u_it1

    f_t2 = np.random.laplace(0, 1, n_samples - int(1.1 * m))
    alphas2 = np.array([1, 0])
    u_it2 = np.random.laplace(0, 1, (n_samples - int(1.1 * m), N))
    correlated_errors2 = f_t2[:, np.newaxis] * alphas2[np.newaxis, :] + u_it2
    correlated_errors = np.concatenate((correlated_errors1, correlated_errors2), axis=0)
    series_list = []

    for i in range(N):
        arma_model = sm.tsa.ArmaProcess(ar=np.r_[1, -ar_param], ma=np.r_[1, ma_param])
        series = arma_model.generate_sample(nsample=n_samples, scale=1,
                                            distrvs=lambda size: correlated_errors[:size[0], i])
        series_list.append(series)
    df = (np.array(series_list, dtype=np.float32).T)
    model_VAR = VAR(df)

    results = model_VAR.fit(selected_lag_order)
    residuals = results.resid

    y = residuals[:m, sery]
    x = residuals[:m, serx]
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    beta_m = model.params[1]
    Dm = model.cov_params()[1][1] * m

    for n in range(m + 1, q * m):
        y = residuals[:n, sery]
        x = residuals[:n, serx]
        x_with_const = sm.add_constant(x)
        model_n = sm.OLS(y, x_with_const).fit()
        beta_n = model_n.params[1]
        Zn = n * (Dm) ** (-1 / 2) * (beta_n - beta_m)
        if (np.abs(Zn) > bondtable[n]):
            countfail += 1
            hit += n
            break

print('power test, countfail={}, m={}, q={}, a={}, hit/countfail={}'.format(countfail, m, q, a, hit/countfail))
with open('Laplace.txt', 'a+') as f:
    f.writelines('power test, countfail={}, m={}, q={}, a={}, hit/countfail={}\n'.format(countfail, m, q, a, hit/countfail))



from tqdm import tqdm
countfail = 0
serx = 0
sery = 1
m = args.m
q = args.q
n_samples = q * m + 5
selected_lag_order = 1

np.random.seed(47)
ar_param = np.random.uniform(0.8, 0.95)
ma_param = 0.2
N = 2
hit = 0

bondtable = np.full(q*m, np.nan, dtype=np.float32)
def bondary_func(m, n, a=2.795):
    part1 = np.sqrt(m)
    part2 = (n - m) / m
    part3 = (n / (n - m)) * (a**2 + np.log(n / (n - m)))
    part4 = np.sqrt(part3)
    result = part1 * part2 * part4
    return result

for n in range(m+1, q*m):
    bondtable[n] = bondary_func(m, n, a)

## update way
for _ in tqdm(range(3000)):

    f_t = np.random.laplace(0, 1, n_samples)
    alphas = np.random.uniform(low=0.1, high=1.0, size=N)
    u_it = np.random.laplace(0, 1, (n_samples, N))
    correlated_errors = f_t[:, np.newaxis] * alphas[np.newaxis, :] + u_it
    series_list = []

    for i in range(N):
        arma_model = sm.tsa.ArmaProcess(ar=np.r_[1, -ar_param], ma=np.r_[1, ma_param])
        series = arma_model.generate_sample(nsample=n_samples, scale=1,
                                            distrvs=lambda size: correlated_errors[:size[0], i])
        series_list.append(series)
    df = (np.array(series_list, dtype=np.float32).T)
    #     model_VAR = VAR(df)

    #     results = model_VAR.fit(selected_lag_order)
    #     residuals = results.resid

    y = df[:m, sery]
    x = df[:m, serx]
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    beta_m = model.params[1]
    Dm = model.cov_params()[1][1] * m

    for n in range(m + 1, q * m):
        y = df[:n, sery]
        x = df[:n, serx]
        x_with_const = sm.add_constant(x)
        model_n = sm.OLS(y, x_with_const).fit()
        beta_n = model_n.params[1]
        Zn = n * (Dm) ** (-1 / 2) * (beta_n - beta_m)
        if (np.abs(Zn) > bondtable[n]):
            countfail += 1
            break
print('without test, countfail={}, m={}, q={}, a={}'.format(countfail, m, q, a))
with open('Laplace.txt', 'a+') as f:
    f.writelines('without test, countfail={}, m={}, q={}, a={}\n'.format(countfail, m, q, a))

print('-----------------------')