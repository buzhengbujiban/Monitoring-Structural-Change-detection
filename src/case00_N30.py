
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
from joblib import Parallel, delayed
import multiprocessing

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
ar_param = 0.5  ### depending on your case
ma_param = 0.6  ### depending on your case
N = 30  ### change to 30




countfail = 0

m = args.m
q = args.q
n_samples = q * m + 5
selected_lag_order = 3  ### depending on your test



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

## update way
residuals_list = np.full((500, n_samples-3, N), np.nan, dtype=np.float32)
for kk in tqdm(range(500)):


    f_t = np.random.normal(size=n_samples)    ### depending on your case
    alphas = np.random.uniform(low=0.1, high=1.0, size=N)
    u_it = np.random.normal(size=(n_samples, N))   ### depending on your case
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
    residuals_list[kk,:,:] = results.resid

sec_dic = {}
cnt = 0
kk_cnt = 0
process_dict = {}
for kk in range(500):
    for serx in range(N - 1):
        for sery in range(serx + 1, N):
            if kk == 0:
                sec_dic[(serx, sery)] = cnt
                cnt += 1
            process_dict[kk_cnt] = (kk, serx, sery)
            kk_cnt += 1


countfail = np.zeros((500, N*(N-1)//2), dtype=bool)
def process_countfail(kk, serx, sery):
    residuals = residuals_list[kk]
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
            return True
    return False

aggCountin = Parallel(n_jobs=multiprocessing.cpu_count() // 2, verbose=10)(delayed(process_countfail)(kk, serx, sery) for kk, serx, sery in tqdm(process_dict.values()))
for i, (kk, serx, sery) in enumerate(process_dict.values()):
    countfail[kk, sec_dic[(serx, sery)]] = aggCountin[i]

for serx in range(N - 1):
    for sery in range(serx + 1, N):
        print('size test, countfail={}, m={}, q={}, a={}, serx={}, sery={}\n'.format(
            sum(countfail[:, sec_dic[(serx, sery)]]), m, q, a, serx, sery))

        with open('case0_N30.txt', 'a+') as f:
            f.writelines('size test, countfail={}, m={}, q={}, a={}, serx={}, sery={}\n'.format(
            sum(countfail[:, sec_dic[(serx, sery)]]), m, q, a, serx, sery))



countfail = 0
serx = 0
sery = 1
m = args.m
q = args.q
n_samples = q * m + 5
selected_lag_order = 3  ### depending on your test

np.random.seed(47)
ar_param = 0.5   ### depending on your case
ma_param = 0.6   ### depending on your case
N = 30   ### change to 30
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



residuals_list_power = np.full((500, n_samples-3, N), np.nan, dtype=np.float32)
for kk in tqdm(range(500)):

    f_t1 = np.random.normal(size=int(1.1*m))  ### depending on your case
    alphas1 = np.array([0.00712, 0.3029, 0.66976, -0.51787, -0.92242,
                        -0.46019, -0.75454, 0.17434, -0.37762, 0.05673,
                        -0.70219, -0.57898, -0.42476, -0.49312, -0.12365,
                        0.35712, -0.19012, 0.36968, -0.10558, 0.3017,
                        -0.32022, 0.09903, 0.83528, 0.84263, 0.28673,
                        0.12099, 0.28259, 0.5747, -0.70148, 0.44789])[:N]
    u_it1 = np.random.normal(size=(int(1.1*m), N))  ### depending on your case
    correlated_errors1 = f_t1[:, np.newaxis] * alphas1[np.newaxis, :] + u_it1

    f_t2 = np.random.normal(size=n_samples - int(1.1*m))  ### depending on your case
    alphas2 = np.array([-0.17939, -0.00697, -0.05205, -0.10621,  0.88445,
                         0.67575, -0.76072, -0.18383,  0.29213,  0.89941,
                         0.81066, -0.05525, 0.47708,  0.82752,  0.09103,
                        0.35712, -0.19012, 0.36968, -0.10558, 0.3017,
                        -0.32022, 0.09903, 0.83528, 0.84263, 0.28673,
                        0.12099, 0.28259, 0.5747, -0.70148, 0.44789])[:N]
    u_it2 = np.random.normal(size=(n_samples - int(1.1*m), N))  ### depending on your case
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
    residuals_list_power[kk, :, :] = results.resid


# sec_dic = {}
# cnt = 0
# kk_cnt = 0
# process_dict = {}
# for kk in range(500):
#     for serx in range(N - 1):
#         for sery in range(serx + 1, N):
#             if kk == 0:
#                 sec_dic[(serx, sery)] = cnt
#                 cnt += 1
#             process_dict[kk_cnt] = (kk, serx, sery)
#             kk_cnt += 1



countfail_power = np.zeros((500, N*(N-1)//2), dtype=bool)
hit = np.full((500, N*(N-1)//2), np.nan, dtype=np.float32)
def process_countfail_power(kk, serx, sery):
    residuals = residuals_list_power[kk]
    y = residuals[:m, sery]
    x = residuals[:m, serx]
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    beta_m = model.params[1]
    Dm = model.cov_params()[1][1] * m


    for n in range(m+1, q*m):
        y = residuals[:n, sery]
        x = residuals[:n, serx]
        x_with_const = sm.add_constant(x)
        model_n = sm.OLS(y, x_with_const).fit()
        beta_n = model_n.params[1]
        Zn = n * (Dm)**(-1/2) * (beta_n - beta_m)
        if(np.abs(Zn)>bondtable[n]):
            return True,n
    return False, 0

# Parallel(n_jobs=multiprocessing.cpu_count() // 2, verbose=10)(delayed(process_countfail_power)(kk, serx, sery) for kk, serx, sery in tqdm(process_dict.values()))

aggCountpower = Parallel(n_jobs=multiprocessing.cpu_count() // 2, verbose=10)(delayed(process_countfail_power)(kk, serx, sery) for kk, serx, sery in tqdm(process_dict.values()))
for i, (kk, serx, sery) in enumerate(process_dict.values()):
    countfail_power[kk, sec_dic[(serx, sery)]] = aggCountpower[i][0]
    hit[kk, sec_dic[(serx, sery)]] = aggCountpower[i][1]



for serx in range(N-1):
    for sery in range(serx+1, N):
        print('power test, countfail={}, m={}, q={}, a={}, serx={}, sery={}, hit={}\n'.format(
            sum(countfail_power[:, sec_dic[(serx, sery)]]), m, q, a, serx, sery, np.nansum(hit[:, sec_dic[(serx, sery)]])))

        with open('case0_N30.txt', 'a+') as f:
            f.writelines('power test, countfail={}, m={}, q={}, a={}, serx={}, sery={}, hit={}\n'.format(
            sum(countfail_power[:, sec_dic[(serx, sery)]]), m, q, a, serx, sery, np.nansum(hit[:, sec_dic[(serx, sery)]])))


