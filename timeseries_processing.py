import numpy as np
import pandas as pd
import os
from VWCD import vwcd, vwcd_mv, vwcd_np
import matplotlib.pyplot as plt

def export_time_series(data):
    """
    Exporta séries temporais das variáveis de download e upload para cada par cliente-site.

    Para cada cliente e site:
    - Cria um DataFrame contendo os dados de download (DownRTTMean, Download) e de upload (UpRTTMean, Upload) para cada par cliente-site.
    - Cada DataFrame é salvos em arquivo .parquet.
    - Gera metadados consolidados contendo informações resumidas sobre as séries temporais.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para carregar e nomear os arquivos correspondentes).

    Saída:
    ------
    pd.DataFrame
        Um DataFrame contendo os metadados das séries temporais, incluindo:
        - client (str): Identificação do cliente.
        - site (str): Identificação do site.
        - inicio (datetime): Timestamp da primeira medição de download.
        - fim (datetime): Timestamp da última medição de download.
        - num_med (int): Número de medições.
        - mean_time (float): Intervalo médio entre medições, em horas.
        - file_prefix (str): Prefixo usado nos nomes dos arquivos gerados.

    Diretórios criados:
    -------------------
    datasets/ts_<data>/

    Arquivos gerados:
    -----------------
    - Séries temporais para cada par cliente-site: <client>_<site>.parquet
    - Metadados: ts_metadata_<data>.parquet
    """
    # Importação dos dados
    df = pd.read_parquet(f'datasets/dados_{data}.parquet')

    # Filtros
    clients = df['ClientMac'].unique()
    sites = df['Site'].unique()

    # Diretórios de saída
    output_dir = f'datasets/ts_{data}/'
    os.makedirs(output_dir, exist_ok=True)

    med = []
    for c in clients:
        for s in sites:
            # Filtros por cliente e site para download e upload
            df_pair = df[(df.ClientMac == c) & (df.Site == s)]
                    
            if len(df_pair) >= 100:
                # Criar DataFrame para download
                df_ts = pd.DataFrame({
                    'timestamp': df_pair['DataHora'].values,
                    'rtt_download': df_pair['DownRTTMean'].values,
                    'throughput_download': df_pair['Download'].values,
                    'rtt_upload': df_pair['UpRTTMean'].values,
                    'throughput_upload': df_pair['Upload'].values
                })

                # Ordenar por timestamp
                df_ts.sort_values(by='timestamp', inplace=True)

                # Salvar em arquivo
                output_file = f"{output_dir}/{c}_{s}.parquet"
                df_ts.to_parquet(output_file, index=False)

                # Coletar metadados
                inicio = df_pair['DataHora'].iloc[0]
                fim = df_pair['DataHora'].iloc[-1]
                num_med = len(df_pair)
                mean_time = np.round(df_pair['DataHora'].diff().mean().seconds / 3600, 1)
                file_prefix = f"{c}_{s}"
                
                quant = {
                    "client": c, 
                    "site": s, 
                    "inicio": inicio,
                    "fim": fim,
                    "num_med": num_med,
                    "mean_time": mean_time,
                    "file_prefix": file_prefix
                }
                med.append(quant)

    # Conjunto de metadados
    df_series = pd.DataFrame(med)
    df_series.to_parquet(f'datasets/ts_metadata_{data}.parquet', index=False)

    return df_series


def detect_changepoints(data, wv, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Detecta pontos de mudança (changepoints) em todas as colunas numéricas de séries temporais.

    Para cada arquivo Parquet gerado pela função `export_time_series`, a função:
    - Detecta changepoints em todas as colunas numéricas (exceto 'timestamp').
    - Adiciona colunas binárias indicando os changepoints para cada variável.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    wv : int
        Tamanho da janela deslizante de votação.
    ab : float
        Hiperparâmetros alfa e beta da distribuição beta-binomial.
    p_thr : float
        Limiar de probabilidade para o voto de uma janela ser registrado.
    vote_p_thr : float
        Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
    vote_n_thr : float
        Fração mínima da janela que precisa votar para definir um ponto de mudança.
    y0 : float
        Probabilidade a priori da função logística (início da janela).
    yw : float
        Probabilidade a priori da função logística (tamanho da janela).
    aggreg : str
        Função de agregação para os votos ('posterior' ou 'mean').

    Retorna:
    -------
    None
        Cria novos arquivos Parquet com colunas binárias indicando os changepoints detectados para cada variável.
    """
    # Diretório de entrada e saída
    input_dir = f'datasets/ts_{data}/'
    output_dir = f'datasets/ts_{data}_cp/'
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada arquivo Parquet de séries temporais
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            # Carregar a série temporal
            df = pd.read_parquet(os.path.join(input_dir, file))

            # Iterar pelas colunas numéricas, exceto 'timestamp'
            for column in df.select_dtypes(include=[np.number]).columns:
                y = df[column].dropna().values

                # Parâmetros do algoritmo VWCD
                kargs = {
                    'X': y, 'w': wv, 'w0': wv, 'ab': ab,
                    'p_thr': p_thr, 'vote_p_thr': vote_p_thr,
                    'vote_n_thr': vote_n_thr, 'y0': y0, 'yw': yw, 'aggreg': aggreg
                }

                # Executar a detecção de changepoints com VWCD
                CP, _, _, _ = vwcd(**kargs)

                # Criar uma coluna binária indicando changepoints
                changepoints = np.zeros(len(y), dtype=int)
                changepoints[CP] = 1

                # Adicionar a coluna de changepoints ao DataFrame
                changepoint_column = f'{column}_cp'
                df[changepoint_column] = 0
                df.loc[df[column].dropna().index, changepoint_column] = changepoints

            # Salvar o DataFrame com as colunas de changepoints, médias e desvios padrão locais
            output_file = os.path.join(output_dir, file)
            df.to_parquet(output_file, index=False)

    print(f"Changepoints detectados e salvos em: {output_dir}")


def detect_changepoints_mv(data, wv, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Detecta pontos de mudança (changepoints) em todas as colunas numéricas de séries temporais.

    Para cada arquivo Parquet gerado pela função `export_time_series`, a função:
    - Detecta changepoints em todas as colunas numéricas (exceto 'timestamp').
    - Adiciona colunas binárias indicando os changepoints para cada variável.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    wv : int
        Tamanho da janela deslizante de votação.
    ab : float
        Hiperparâmetros alfa e beta da distribuição beta-binomial.
    p_thr : float
        Limiar de probabilidade para o voto de uma janela ser registrado.
    vote_p_thr : float
        Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
    vote_n_thr : float
        Fração mínima da janela que precisa votar para definir um ponto de mudança.
    y0 : float
        Probabilidade a priori da função logística (início da janela).
    yw : float
        Probabilidade a priori da função logística (tamanho da janela).
    aggreg : str
        Função de agregação para os votos ('posterior' ou 'mean').

    Retorna:
    -------
    None
        Cria novos arquivos Parquet com colunas binárias indicando os changepoints detectados para cada variável.
    """
    # Diretório de entrada e saída
    input_dir = f'datasets/ts_{data}/'
    output_dir = f'datasets/ts_{data}_cp/'
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada arquivo Parquet de séries temporais
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            # Carregar a série temporal
            df = pd.read_parquet(os.path.join(input_dir, file))
            columns = ['throughput_download', 'rtt_download', 'throughput_upload', 'rtt_upload']

            X = df[columns].values

            # Parâmetros do algoritmo VWCD
            kargs = {
                'X': X, 'w': wv, 'w0': wv, 'ab': ab,
                'p_thr': p_thr, 'vote_p_thr': vote_p_thr,
                'vote_n_thr': vote_n_thr, 'y0': y0, 'yw': yw, 'aggreg': aggreg
            }

            # Executar a detecção de changepoints com VWCD
            CP, _, _ = vwcd_mv(**kargs)

            # Criar uma coluna binária indicando changepoints
            changepoints = np.zeros(len(X), dtype=int)
            changepoints[CP] = 1

            # Adicionar a coluna de changepoints ao DataFrame
            df['cp'] = 0
            df.loc[:, 'cp'] = changepoints

            # Salvar o DataFrame com as colunas de changepoints, médias e desvios padrão locais
            output_file = os.path.join(output_dir, file)
            df.to_parquet(output_file, index=False)

    print(f"Changepoints detectados e salvos em: {output_dir}")


def detect_changepoints_np(data, wv, vote_js_thr, vote_n_thr):
    """
    Detecta pontos de mudança (changepoints) em todas as colunas numéricas de séries temporais.

    Para cada arquivo Parquet gerado pela função `export_time_series`, a função:
    - Detecta changepoints em todas as colunas numéricas (exceto 'timestamp').
    - Adiciona colunas binárias indicando os changepoints para cada variável.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    wv : int
        Tamanho da janela deslizante de votação.
    ab : float
        Hiperparâmetros alfa e beta da distribuição beta-binomial.
    p_thr : float
        Limiar de probabilidade para o voto de uma janela ser registrado.
    vote_p_thr : float
        Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
    vote_n_thr : float
        Fração mínima da janela que precisa votar para definir um ponto de mudança.
    y0 : float
        Probabilidade a priori da função logística (início da janela).
    yw : float
        Probabilidade a priori da função logística (tamanho da janela).
    aggreg : str
        Função de agregação para os votos ('posterior' ou 'mean').

    Retorna:
    -------
    None
        Cria novos arquivos Parquet com colunas binárias indicando os changepoints detectados para cada variável.
    """
    # Diretório de entrada e saída
    input_dir = f'datasets/ts_{data}/'
    output_dir = f'datasets/ts_{data}_cp/'
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada arquivo Parquet de séries temporais
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            # Carregar a série temporal
            df = pd.read_parquet(os.path.join(input_dir, file))
            columns = ['throughput_download', 'rtt_download', 'throughput_upload', 'rtt_upload']

            X = df[columns].values

            # Parâmetros do algoritmo VWCD
            kargs = {
                'X': X,
                'w': wv,
                'vote_js_thr': vote_js_thr,
                'vote_n_thr': vote_n_thr,
            }

            # Executar a detecção de changepoints com VWCD
            CP, _ = vwcd_np(**kargs)

            # Criar uma coluna binária indicando changepoints
            changepoints = np.zeros(len(X), dtype=int)
            changepoints[CP] = 1

            # Adicionar a coluna de changepoints ao DataFrame
            df['cp'] = 0
            df.loc[:, 'cp'] = changepoints

            # Salvar o DataFrame com as colunas de changepoints, médias e desvios padrão locais
            output_file = os.path.join(output_dir, file)
            df.to_parquet(output_file, index=False)

    print(f"Changepoints detectados e salvos em: {output_dir}")


def plot_changepoints(data, client, site, variable, ylim=None):
    """
    Plota os valores de uma variável ao longo do tempo com changepoints destacados como linhas verticais.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    client : str
        Identificador do cliente.
    site : str
        Identificador do site.
    variable : str
        Nome da variável a ser plotada.
    ylim : tuple or None, optional
        Limites do eixo Y no formato (y_min, y_max). Se None, os limites serão automáticos.

    Retorna:
    -------
    None
        Exibe um gráfico com os valores da variável ao longo do tempo e os changepoints destacados.
    """
    # Diretório onde os arquivos com changepoints estão armazenados
    input_dir = f'datasets/ts_{data}_cp/'
    file_name = f"{client}_{site}.parquet"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Arquivo para o cliente {client} e site {site} não encontrado.")
        return

    # Carregar o arquivo Parquet
    df = pd.read_parquet(file_path)

    # Verificar se a variável e a coluna de changepoints existem
    changepoint_column = f"{variable}_cp"
    if variable not in df.columns or changepoint_column not in df.columns:
        print(f"A variável '{variable}' ou os changepoints não estão disponíveis no arquivo.")
        return

    # Obter os timestamps dos changepoints
    changepoints = df['timestamp'][df[changepoint_column] == 1]

    # Plotar os dados
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df[variable], label='Valores', color='blue', alpha=0.7)

    # Adicionar linhas verticais para os changepoints
    for cp in changepoints:
        plt.axvline(x=cp, color='red', linestyle='--', label='Ponto de mudança' if cp == changepoints.iloc[0] else '')

    # Configurações do gráfico
    plt.xlabel('Tempo')
    plt.ylabel(variable)
    plt.title(f"{variable} - {client} - {site}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Aplicar limites do eixo Y, se especificado
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()


def create_survival_dataset(data, feature, max_gap_days=3):
    """
    Cria um dataset de sobrevivência baseado em séries temporais e pontos de mudança de uma variável específica.

    Parâmetros:
    ----------
    cp_dir : str
        Diretório contendo os arquivos Parquet com colunas de changepoints e dados das séries temporais.
    feature : str
        Nome da variável para a qual os changepoints serão considerados.
    max_gap_days : int
        Intervalo máximo de dias permitido entre medições consecutivas antes de considerar um intervalo censurado.

    Retorna:
    -------
    pd.DataFrame
        Dataset de sobrevivência com as seguintes colunas:
        - 'client', 'site': Identificação do cliente e site.
        - 'time': Duração do intervalo em dias.
        - 'timestamp_start', 'timestamp_end': Timestamps de início e fim do intervalo.
        - Variáveis originais: Valores no início do intervalo.
        - 'event': 1 se o intervalo termina em um changepoint, 0 se for censurado.
    """
    survival_data = []
    cp_dir = f'datasets/ts_{data}_cp/'

    for file in os.listdir(cp_dir):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(cp_dir, file))
            client, site = file.split('.')[0].split('_', 1)

            # Ordenar pelo timestamp
            df.sort_values(by='timestamp', inplace=True)

            # Identificar changepoints
            changepoint_column = f'{feature}_cp'
            if changepoint_column not in df.columns:
                print(f"Changepoint column '{changepoint_column}' not found in {file}. Skipping.")
                continue
            changepoint_indices = df.index[df[changepoint_column] == 1].tolist()

            # Calcular gaps de tempo
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / (60 * 60 * 24)
            large_gaps = df.index[df['time_diff'] > max_gap_days].tolist()
            split_indices = [0]

            # Adicionar os índices antes e depois dos gaps como split indices
            for gap in large_gaps:
                if gap - 1 >= 0:
                    split_indices.append(gap - 1)  # Antes do gap
                split_indices.append(gap)  # Depois do gap
            split_indices.append(len(df) - 1)

            split_indices = sorted(set(split_indices))  # Garantir que os índices são únicos e ordenados

            # Iterar entre as subdivisões
            for i, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
                sub_df = df.iloc[start:end + 1]  # Incluir o índice final
                if len(sub_df) < 2:
                    continue

                # Identificar changepoints nesta subsequência
                sub_changepoints = [cp for cp in changepoint_indices if start <= cp <= end]
                all_points = [start] + sub_changepoints + [end]

                # Iterar sobre os intervalos entre todos os pontos
                for j in range(len(all_points) - 1):
                    start_idx = all_points[j]
                    end_idx = all_points[j + 1]
                    if start_idx >= len(df) or end_idx >= len(df) or start_idx >= end_idx:
                        continue

                    start_time = df['timestamp'].iloc[start_idx]
                    end_time = df['timestamp'].iloc[end_idx]
                    duration = (end_time - start_time).total_seconds() / (60 * 60 * 24)

                    initial_values = df.iloc[start_idx].to_dict()
                    
                    # Buscar o primeiro valor válido para cada variável
                    throughput_download = initial_values.get('throughput_download', np.nan)
                    if pd.isna(throughput_download):
                        throughput_download = df['throughput_download'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['throughput_download'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    throughput_upload = initial_values.get('throughput_upload', np.nan)
                    if pd.isna(throughput_upload):
                        throughput_u = df['throughput_upload'].iloc[startthroughput_u_idx:end_idx+1].dropna().values[0] if not df['throughput_upload'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    rtt_download = initial_values.get('rtt_download', np.nan)
                    if pd.isna(rtt_download):
                        rtt_download = df['rtt_download'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['rtt_download'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    rtt_upload = initial_values.get('rtt_upload', np.nan)
                    if pd.isna(rtt_upload):
                        rtt_upload = df['rtt_upload'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['rtt_upload'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    event = 1 if start_idx in changepoint_indices and end_idx in changepoint_indices else 0

                    survival_data.append({
                        'client': client,
                        'site': site,
                        'timestamp_start': start_time,
                        'timestamp_end': end_time,
                        'time': duration,
                        'throughput_download': throughput_download,
                        'rtt_download': rtt_download,
                        'throughput_upload': throughput_upload,
                        'rtt_upload': rtt_upload,
                        'event': event
                    })

    # Converter para DataFrame
    survival_df = pd.DataFrame(survival_data)
    survival_df = pd.get_dummies(survival_df, columns=['client', 'site'])

    for col in survival_df.columns:
        if col.startswith('client_') or col.startswith('site_'):
            survival_df[col] = survival_df[col].astype(int)

    survival_df.to_parquet(f'datasets/survival_{data}.parquet', index=False)
    return survival_df