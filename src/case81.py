

import numpy as np
import statsmodels.api as sm
from scipy.stats import t
import pandas as pd
from scipy.stats import t
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import norm
from scipy.optimize import brentq
import argparse
from tqdm import tqdm
import nolds
from concurrent.futures import ProcessPoolExecutor

hurst = 0.7



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('m', type=int, help='An integer to be read')
parser.add_argument('q', type=int, help='An integer to be read')
parser.add_argument('percent', type=float, help='An integer to be read')
args = parser.parse_args()




def equation(a, alpha, k=1):
    return 1 - (1 - 2 * (1 + a * norm.pdf(a) - norm.cdf(a))) ** k - (alpha)


def solve_for_a(alpha):
    a_low = -10
    a_high = 10
    a = brentq(equation, a_low, a_high, args=(alpha,))
    return a

alpha = args.percent
a = solve_for_a(alpha)
print('a=',a)


np.random.seed(47)
ar_param = 0.05  ### depending on your case
ma_param = 0.2  ### depending on your case
N = 2




countfail = 0
serx = 0
sery = 1
m = args.m
q = args.q
n_samples = q * m + 5
selected_lag_order = 1 ### depending on your test



bondtable = np.full(q * m, np.nan, dtype=np.float32)
def bondary_func(m, n, a=2.795):
    part1 = np.sqrt(m)
    part2 = (n - m) / m
    part3 = (n / (n - m)) * (a ** 2 + np.log(n / (n - m)))
    part4 = np.sqrt(part3)
    result = part1 * part2 * part4
    return result


for n in range(m + 1, q * m):
    bondtable[n] = bondary_func(m, n, a)

print('size test, m={}, q={}, a={}'.format(m, q, a))


countfail = 0
serx = 0
sery = 1
m = args.m
q = args.q
n_samples = q * m + 5
selected_lag_order = 1  ### depending on your test

np.random.seed(47)
ar_param = 0.05   ### depending on your case
ma_param = 0.2   ### depending on your case
N = 2
hit = 0




def bondary_func(m, n, a=2.795):
    part1 = np.sqrt(m)
    part2 = (n - m) / m
    part3 = (n / (n - m)) * (a ** 2 + np.log(n / (n - m)))
    part4 = np.sqrt(part3)
    result = part1 * part2 * part4
    return result


bondtable = np.full(q * m, np.nan, dtype=np.float32)
for n in range(m + 1, q * m):
    bondtable[n] = bondary_func(m, n, a)



import gc


def process_iteration1(_):
    countfail = 0
    hit = 0
    f_t1 = np.random.normal(size=int(1.1*m))  ### depending on your case
    alphas1 = np.array([0.2711117, 0.92084525])
    u_it1 = np.random.normal(size=(int(1.1*m), N))  ### depending on your case
    correlated_errors1 = f_t1[:, np.newaxis] * alphas1[np.newaxis, :] + u_it1

    f_t2 = np.random.normal(size=n_samples - int(1.1*m))  ### depending on your case
    alphas2 = np.array([1, 0])
    u_it2 = np.random.normal(size=(n_samples - int(1.1*m), N))  ### depending on your case
    correlated_errors2 = f_t2[:, np.newaxis] * alphas2[np.newaxis, :] + u_it2
    correlated_errors = np.concatenate((correlated_errors1, correlated_errors2), axis=0)
    series_list = []

    for i in range(N):
        # arma_model = sm.tsa.ArmaProcess(ar=np.r_[1, -ar_param], ma=np.r_[1, ma_param])
        series = nolds.fgn(n_samples, hurst) + correlated_errors[:, i]
        series_list.append(series)
    df = np.array(series_list, dtype=np.float32).T
    model = VAR(df)
    results = model.fit(selected_lag_order)
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
    return countfail*10000+hit/10000

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=5) as executor:
        results1 = list(tqdm(executor.map(process_iteration1, range(500)), total=500))
    countfail = sum(results1)
    print('power test, countfail={}, m={}, q={}, a={}, hit/countfail={}'.format(countfail, m, q, a, 0/countfail))
    with open('case7.txt', 'a+') as f:    ### depending on your case
        f.writelines('power test, countfail={}, m={}, q={}, a={}, hit/countfail={}\n'.format(countfail, m, q, a, 0/countfail))

