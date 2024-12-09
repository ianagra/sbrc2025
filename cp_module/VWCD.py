"""
Script: Detecção de pontos de mudança no conjunto de dados NDT utilizando o método Voting Windows Changepoint Detection.
Descrição: Aplica o método VWCD a todas as séries temporais do conjunto de dados NDT para detecção de pontos de mudança.

Requisitos (dentro da função)
----------
- Dataframe './datasets/ts_ndt.pkl' com as informações das séries temporais.
- Arquivos .txt das séries temporais em './datasets/ts_ndt/'.
- Configuração dos hiperparâmetros.

Resultado
----------
- Dataframe com os resultados da detecção de pontos de mudança

Autores
----------
- Cleiton Moya de Almeida (2024): autor do método VWCD e da versão original do script.
- Ian José Agra Gomes (2025): autor das modificações e adaptações no script.
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro, betabinom
from statsmodels.tsa.stattools import adfuller
from rpy2.robjects.packages import importr
import time

# Importações de pacotes R
changepoint_np = importr('changepoint.np')
changepoint = importr('changepoint')

# Configuração de verbosidade
verbose = False


# Funções auxiliares para testes estatísticos
def normality_test(y, alpha):
    """Teste de normalidade Shapiro-Wilk."""
    _, pvalue = shapiro(y)
    return pvalue > alpha


def stationarity_test(y, alpha):
    """Teste de estacionaridade Augmented Dickey-Fuller."""
    adf = adfuller(y)
    pvalue = adf[1]
    return pvalue < alpha


# Funções auxiliares de log-verossimilhança
def logpdf(x, loc, scale):
    """Calcula a log-pdf para distribuição normal."""
    c = 1 / np.sqrt(2 * np.pi)
    y = np.log(c) - np.log(scale) - (1 / 2) * ((x - loc) / scale) ** 2
    return y


def loglik(x, loc, scale):
    """Calcula a log-verossimilhança para distribuição normal."""
    n = len(x)
    c = 1 / np.sqrt(2 * np.pi)
    y = n * np.log(c / scale) - (1 / (2 * scale**2)) * ((x - loc) ** 2).sum()
    return y


# Algoritmo Voting Windows Changepoint Detection
def vwcd(X, w, w0, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Implementação do algoritmo Voting Windows Changepoint Detection.
    """
    def pos_fun(ll, prior, tau):
        c = np.nanmax(ll)
        lse = c + np.log(np.nansum(prior * np.exp(ll - c)))
        p = ll[tau] + np.log(prior[tau]) - lse
        return np.exp(p)

    def votes_pos(vote_list, prior_v):
        vote_list = np.array(vote_list)
        prod1 = vote_list.prod() * prior_v
        prod2 = (1 - vote_list).prod() * (1 - prior_v)
        p = prod1 / (prod1 + prod2)
        return p

    def logistic_prior(x, w, y0, yw):
        a = np.log((1 - y0) / y0)
        b = np.log((1 - yw) / yw)
        k = (a - b) / w
        x0 = a / k
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y

    N = len(X)
    vote_n_thr = np.floor(w * vote_n_thr)
    i_ = np.arange(0, w - 3)
    prior_w = betabinom(n=w - 4, a=ab, b=ab).pmf(i_)
    x_votes = np.arange(1, w + 1)
    prior_v = logistic_prior(x_votes, w, y0, yw)

    votes = {i: [] for i in range(N)}
    votes_agg = {}
    lcp = 0
    CP = []
    M0 = []
    S0 = []

    startTime = time.time()
    for n in range(N):
        if n >= w - 1:
            if n == lcp + w0:
                m_w0 = X[n - w0 + 1 : n + 1].mean()
                s_w0 = X[n - w0 + 1 : n + 1].std(ddof=1)
                M0.append(m_w0)
                S0.append(s_w0)

            Xw = X[n - w + 1 : n + 1]
            LLR_h = []
            for nu in range(1, w - 3 + 1):
                x1 = Xw[: nu + 1]
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                if np.round(s1, 3) == 0:
                    s1 = 0.001
                logL1 = loglik(x1, loc=m1, scale=s1)

                x2 = Xw[nu + 1 :]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                if np.round(s2, 3) == 0:
                    s2 = 0.001
                logL2 = loglik(x2, loc=m2, scale=s2)

                llr = logL1 + logL2
                LLR_h.append(llr)

            LLR_h = np.array(LLR_h)
            pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w - 3)]
            pos = [np.nan] + pos + [np.nan] * 2
            pos = np.array(pos)

            p_vote_h = np.nanmax(pos)
            nu_map_h = np.nanargmax(pos)

            if p_vote_h >= p_thr:
                j = n - w + 1 + nu_map_h
                votes[j].append(p_vote_h)

            votes_list = votes[n - w + 1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes - 1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                votes_agg[n - w + 1] = agg_vote

                if agg_vote > vote_p_thr:
                    if verbose:
                        print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n - w + 1
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime - startTime
    return CP, M0, S0, elapsedTime


# Função principal para processar séries temporais
def detect_changepoints(df, series_type, wv, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    results = []
    N = len(df)

    for n in range(N):
        client = df.iloc[n]['client']
        site = df.iloc[n]['site']
        prefixo = f"{client}_{site}_"

        for s_type in series_type:
            file = prefixo + s_type + ".txt"
            y = np.loadtxt(f'datasets/ts_ndt/{file}', usecols=1, delimiter=',')
            y = y[~np.isnan(y)]

            kargs = {
                'X': y, 'w': wv, 'w0': wv, 'ab': ab,
                'p_thr': p_thr, 'vote_p_thr': vote_p_thr,
                'vote_n_thr': vote_n_thr, 'y0': y0, 'yw': yw, 'aggreg': aggreg
            }

            CP, M0, S0, elapsed_time = vwcd(**kargs)

            res = {
                'client': client, 'site': site, 'serie': s_type,
                'method': vwcd.__name__, 'CP': CP, 'num_cp': len(CP),
                'M0': M0, 'S0': S0, 'elapsed_time': elapsed_time
            }
            results.append(res)

    df_results = pd.DataFrame(results)
    return df_results