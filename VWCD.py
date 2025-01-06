"""
Autor do método e do código original: Cleiton Moya de Almeida (2024).

Autor das modificações e da implementação multivariada do método: Ian José Agra Gomes (2025).
"""
import numpy as np
from scipy.stats import shapiro, betabinom, multivariate_normal
from statsmodels.tsa.stattools import adfuller
import time
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon

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
        ab (float): Hiperparâmetros alfa e beta da distribuição beta-binomial, distribuição a priori da janela
        p_thr (float): Limiar de probabilidade para o voto de uma janela ser registrado.
        vote_p_thr (float): Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
        vote_n_thr (float): Fração mínima da janela que precisa votar para definir um ponto de mudança após a agregação.
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


def loglik_mv(X, mean, cov, regularization_factor=1e-4):
    """
    Calcula a log-verossimilhança para uma distribuição normal multivariada.

    Parâmetros:
        X (array-like): Dados da amostra (matriz com observações nas linhas e variáveis nas colunas).
        mean (array-like): Vetor de médias da distribuição.
        cov (array-like): Matriz de covariância da distribuição.
        regularization_factor (float): Fator de regularização.

    Retorna:
        float: Valor da log-verossimilhança calculada.
    """
    cov = validate_covariance(cov, X.shape[1], regularization_factor)
    try:
        mvn = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        return mvn.logpdf(X).sum()
    except np.linalg.LinAlgError:
        return -np.inf


def validate_covariance(cov, d, regularization_factor=1e-4):
    """
    Valida a matriz de covariância e aplica regularização se necessário.

    Parâmetros:
        cov (array-like): Matriz de covariância.
        d (int): Dimensão esperada da matriz.
        regularization_factor (float): Fator de regularização.

    Retorna:
        array-like: Matriz de covariância válida.
    """
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)) or np.linalg.det(cov) <= 1e-10:
        return np.eye(d) * regularization_factor
    return cov + regularization_factor * np.eye(d)



def vwcd_mv(X, w, w0, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg, verbose=False):
    """
    Detecta pontos de mudança em uma série temporal multivariada usando VWCD.

    Parâmetros:
        X (array-like): Série temporal multivariada (matriz com observações nas linhas e variáveis nas colunas).
        w (int): Tamanho da janela deslizante de votação.
        w0 (int): Janela inicial para estimar parâmetros iniciais.
        ab (float): Hiperparâmetros alfa e beta da distribuição beta-binomial, distribuição a priori da janela.
        p_thr (float): Limiar de probabilidade para o voto de uma janela ser registrado.
        vote_p_thr (float): Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
        vote_n_thr (float): Fração mínima da janela que precisa votar para definir um ponto de mudança após a agregação.
        y0 (float): Probabilidade a priori da função logística (início da janela).
        yw (float): Probabilidade a priori da função logística (início da janela).
        aggreg (str): Função de agregação para os votos ('posterior' ou 'mean').
        verbose (bool): Se True, exibe informações sobre os pontos de mudança detectados.

    Retorna:
        tuple: 
            - CP (list): Lista de índices dos pontos de mudança detectados.
            - Params (list): Lista de parâmetros (médias e covariâncias) estimados nas janelas.
            - elapsedTime (float): Tempo total de execução do algoritmo.
    """
    # Normalização robusta
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    N, d = X.shape
    vote_n_thr = np.floor(w * vote_n_thr)
    CP = []
    Params = []

    startTime = time.time()
    for n in range(N):
        if n >= w - 1:
            Xw = X[n - w + 1 : n + 1]
            LLR_h = []
            for nu in range(1, w - 1):
                x1 = Xw[:nu + 1]
                x2 = Xw[nu + 1:]

                if len(x1) < d or len(x2) < d:
                    LLR_h.append(-np.inf)
                    continue

                mean1 = x1.mean(axis=0)
                cov1 = np.cov(x1, rowvar=False)
                cov1 = validate_covariance(cov1, d)

                mean2 = x2.mean(axis=0)
                cov2 = np.cov(x2, rowvar=False)
                cov2 = validate_covariance(cov2, d)

                # Verificação adicional para covariância
                if np.linalg.det(cov1) < 1e-10 or np.linalg.det(cov2) < 1e-10:
                    LLR_h.append(-np.inf)
                    continue

                logL1 = loglik_mv(x1, mean=mean1, cov=cov1)
                logL2 = loglik_mv(x2, mean=mean2, cov=cov2)
                llr = logL1 + logL2
                LLR_h.append(llr)

            LLR_h = np.array(LLR_h)
            if np.all(np.isnan(LLR_h)):
                continue

            pos = np.nanargmax(LLR_h)

            if LLR_h[pos] >= p_thr:
                CP.append(n - w + 1 + pos)
                Params.append(((mean1, cov1), (mean2, cov2)))

                if verbose:
                    print(f"Changepoint at n={n-w+1+pos}, LLR={LLR_h[pos]:.4f}")


    endTime = time.time()
    elapsedTime = endTime - startTime

    return CP, Params, elapsedTime


def estimate_density(data):
    """
    Estima a densidade de probabilidade usando KDE.

    Parâmetros:
        data (array-like): Dados para estimar a densidade.

    Retorna:
        KernelDensity: Objeto KDE ajustado aos dados.
    """
    kde = KernelDensity(kernel='gaussian').fit(data)
    return kde


def compute_js_divergence(density1, density2, points):
    """
    Calcula a divergência de Jensen-Shannon entre duas densidades.

    Parâmetros:
        density1 (KernelDensity): Estimativa da densidade 1.
        density2 (KernelDensity): Estimativa da densidade 2.
        points (array-like): Conjunto de pontos para avaliar as densidades.

    Retorna:
        float: Divergência de Jensen-Shannon.
    """
    log_density1 = density1.score_samples(points)
    log_density2 = density2.score_samples(points)
    p1 = np.exp(log_density1)
    p2 = np.exp(log_density2)
    return jensenshannon(p1, p2)


def vwcd_np(X, w, vote_js_thr, vote_n_thr, verbose=False):
    """
    Detecta pontos de mudança em uma série temporal multivariada usando uma abordagem não-paramétrica.

    Parâmetros:
        X (array-like): Série temporal multivariada (n_samples x n_features).
        w (int): Tamanho da janela deslizante.
        vote_js_thr (float): Limiar para divergência JS ser considerada significativa.
        vote_n_thr (float): Fração mínima de votos necessários para definir um ponto de mudança.
        verbose (bool): Se True, exibe informações sobre os pontos de mudança detectados.

    Retorna:
        tuple: 
            - CP (list): Lista de índices dos pontos de mudança detectados.
            - elapsedTime (float): Tempo total de execução do algoritmo.
    """
    import time
    startTime = time.time()

    # Normalização robusta
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    N, d = X.shape
    CP = []  # Lista de pontos de mudança detectados

    vote_n_thr = np.floor(w * vote_n_thr)
    votes = {i: [] for i in range(N)}  # Mapa de votos

    for n in range(w, N):
        # Dados da janela deslizante
        Xw = X[n - w:n]

        # Iterar sobre todas as possíveis divisões da janela
        LLR_h = []
        for nu in range(1, w):
            x1 = Xw[:nu]
            x2 = Xw[nu:]

            # Estimar densidades KDE
            kde1 = estimate_density(x1)
            kde2 = estimate_density(x2)

            # Conjunto de pontos para comparar as distribuições
            points = np.vstack([x1, x2])

            # Calcular JS
            js_div = compute_js_divergence(kde1, kde2, points)
            LLR_h.append(js_div)

        # Escolher a maior divergência JS
        LLR_h = np.array(LLR_h)
        max_js = np.nanmax(LLR_h)
        nu_map_h = np.nanargmax(LLR_h)

        # Votação se JS ultrapassa o limiar
        if max_js >= vote_js_thr:
            j = n - w + nu_map_h
            votes[j].append(max_js)

        # Agregação de votos
        votes_list = votes[n - w]
        if len(votes_list) >= vote_n_thr:
            avg_vote = np.mean(votes_list)
            if avg_vote > vote_js_thr:
                CP.append(n - w)
                if verbose:
                    print(f"Changepoint at n={n-w}, avg JS={avg_vote:.4f}")

    endTime = time.time()
    elapsedTime = endTime - startTime

    return CP, elapsedTime