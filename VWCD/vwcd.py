"""
Script: Detecção de pontos de mudança no conjunto de dados NDT
Descrição: Aplica o método VWCD a todas as séries temporais do conjunto de dados NDT para detecção de pontos de mudança.

Requisitos (dentro da função)
----------
- Dataframe '../dataset/series.pkl' com as informações das séries temporais.
- Arquivos .txt das séries temporais em '../dataset/ndt/'.
- Configuração dos hiperparâmetros.

Resultado
----------
- Dataframe 'results_ndt/results_ndt_vwcd.pkl' com os resultados da detecção de pontos de mudança

Autores
----------
- Cleiton Moya de Almeida (2024): autor do método VWCD e da versão original do script.
- Ian José Agra Gomes (2025): autor das modificações no script.
"""

import numpy as np
import pandas as pd
import changepoint_module as cm

# Ler o dataframe com informações das séries temporais
df = pd.read_pickle('../datasets/series.pkl')
N = len(df)

series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']

# list if the methods
methods = [cm.vwcd]

# Configuração dos hiperparâmetros
wv = 20             # Tamanho da janela deslizante
ab = 1              # Alfa e Beta da alfa-binomial (distribuição a priori dentro da janela)
p_thr = 0.8         # Limiar de probabilidade para uma janela definir um ponto de mudança
vote_p_thr = 0.9    # Limiar de probabilidade para definir um ponto de mudança após a agregação
vote_n_thr = 0.5    # Número de votos mínimo para a definição de um ponto de mudança
y0 = 0.5            # Logistic prior hyperparameter
yw = 0.9            # Logistic prior hyperparameter
aggreg = 'mean'     # Função de agregação

# Execução do método
m = cm.vwcd  
print(f"\nProcessando método {m.__name__}")

results = [] 
for n in range(N):
    
    client = df.iloc[n]['client']
    site = df.iloc[n]['site']
        
    # Prefixo do arquivo
    prefixo = client + "_" + site + "_"
    
    print(f"Processando par cliente-site {n+1}/{N}")
        
    for s_type in series_type:
        
        # Carregar a série temporal
        file = prefixo + s_type + ".txt"
        y = np.loadtxt(f'../datasets/ts_ndt/{file}', usecols=1, delimiter=',')
        
        # Remover valores ausentes
        y = y[~np.isnan(y)]
        
        # Definir os kargs
        kargs = {'X':y, 'w':wv, 'w0':wv, 'ab':ab, 
                 'p_thr':p_thr, 'vote_p_thr':vote_p_thr, 
                 'vote_n_thr':vote_n_thr, 'y0':y0, 'yw':yw, 'aggreg':aggreg}
        
        # Chamar o método
        num_anom_u = num_anom_l = M0 = S0 = None
        out = m(**kargs)
        CP, M0, S0, elapsed_time = out
        
        # Armazenar os resultados
        res = {'client': client, 'site': site, 'serie': s_type,
                'method': m.__name__, 'CP': CP, 'num_cp': len(CP), 
                'num_anom_u': num_anom_u, 'num_anom_l': num_anom_l,
                'M0': M0, 'S0': S0, 'elapsed_time': elapsed_time} 

        results.append(res)

    # Dataframe com os resultados
    df_results = pd.DataFrame(results)
    df_results.to_pickle('changepoints_ndt.pkl')