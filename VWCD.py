"""
Autor do método e do código original: Cleiton Moya de Almeida (2024).

Autor das modificações: Ian José Agra Gomes (2025).
"""
import numpy as np
from scipy.stats import shapiro, betabinom, multivariate_normal, betabinom, chi2
from statsmodels.tsa.stattools import adfuller
import time
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def normality_test(y, alpha):
    """
    Realiza o teste de normalidade Shapiro-Wilk.

    Parâmetros:
        y (array-like): Dados da amostra.
        alpha (float): Nível de significância para o teste.

    Retorna:
        bool: True se a hipótese nula (normalidade) não for rejeitada; False caso contrário.
    """
    _, pvalue = shapiro(y)
    return pvalue > alpha


def stationarity_test(y, alpha):
    """
    Realiza o teste de estacionaridade Augmented Dickey-Fuller.

    Parâmetros:
        y (array-like): Série temporal para análise.
        alpha (float): Nível de significância para o teste.

    Retorna:
        bool: True se a hipótese nula (não estacionariedade) for rejeitada; False caso contrário.
    """
    adf = adfuller(y)
    pvalue = adf[1]
    return pvalue < alpha


def logpdf(x, loc, scale):
    """
    Calcula o logaritmo da densidade de probabilidade (log-pdf) para uma distribuição normal.

    Parâmetros:
        x (array-like): Dados da amostra.
        loc (float): Média da distribuição.
        scale (float): Desvio padrão da distribuição.

    Retorna:
        array-like: Valores da log-pdf calculados para os dados.
    """
    c = 1 / np.sqrt(2 * np.pi)
    y = np.log(c) - np.log(scale) - (1 / 2) * ((x - loc) / scale) ** 2
    return y


def loglik(x, loc, scale):
    """
    Calcula a log-verossimilhança para uma distribuição normal.

    Parâmetros:
        x (array-like): Dados da amostra.
        loc (float): Média da distribuição.
        scale (float): Desvio padrão da distribuição.

    Retorna:
        float: Valor da log-verossimilhança calculada.
    """
    n = len(x)
    c = 1 / np.sqrt(2 * np.pi)
    y = n * np.log(c / scale) - (1 / (2 * scale**2)) * ((x - loc) ** 2).sum()
    return y


def vwcd(X, w, w0, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg, verbose=False):
    """
    Detecta pontos de mudança em uma série temporal usando o algoritmo Voting Windows Changepoint Detection.

    Parâmetros:
        X (array-like): Série temporal.
        w (int): Tamanho da janela deslizante de votação.
        w0 (int): Janela inicial para estimar parâmetros iniciais.
        ab (float): Hiperparâmetros alfa e beta da distribuição beta-binomial
        p_thr (float): Limiar de probabilidade para o voto de uma janela ser registrado.
        vote_p_thr (float): Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
        vote_n_thr (float): Fração mínima da janela que precisa votar.
        y0 (float): Probabilidade a priori da função logística (início da janela).
        yw (float): Probabilidade a priori da função logística (início da janela).
        aggreg (str): Função de agregação para os votos ('posterior' ou 'mean').
        verbose (bool): Se True, exibe informações sobre os pontos de mudança detectados.

    Retorna:
        tuple: 
            - CP (list): Lista de índices dos pontos de mudança detectados.
            - M0 (list): Lista de médias estimadas nas janelas.
            - S0 (list): Lista de desvios padrão estimados nas janelas.
            - elapsedTime (float): Tempo total de execução do algoritmo.
            - vote_counts (array): Array com o número de votos para cada ponto.
            - vote_probs (array): Array com as probabilidades máximas dos votos individuais.
            - agg_probs (array): Array com as probabilidades após agregação dos votos.
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
    lcp = 0
    CP = []
    M0 = []
    S0 = []
    
    vote_counts = np.zeros(N)      # Array para armazenar o número de votos
    vote_probs = np.zeros(N)       # Array para armazenar probabilidades individuais dos votos
    agg_probs = np.zeros(N)        # Array para armazenar probabilidades agregadas

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
                vote_counts[j] += 1
                vote_probs[j] = max(vote_probs[j], p_vote_h)

            votes_list = votes[n - w + 1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes - 1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                agg_probs[n - w + 1] = agg_vote  # Armazenar probabilidade agregada

                if agg_vote > vote_p_thr:
                    if verbose:
                        print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n - w + 1
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime - startTime
    return CP, M0, S0, elapsedTime, vote_counts, vote_probs, agg_probs