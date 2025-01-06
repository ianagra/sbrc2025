import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_pairs(data, client_site_list, title, survival=True, local_mean=True, changepoints=True):
    """
    Plota os valores de todas as variáveis ao longo do tempo em gráficos separados, sincronizados no eixo X,
    para os pares (cliente, site) fornecidos. Cada par ocupa uma coluna e cada variável ocupa uma linha
    da mesma coluna.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    client_site_list : list of tuples
        Lista contendo pares (cliente, site).
    survival : bool, default=True
        Se True, plota os valores de 'survival_probability' em um gráfico separado.
    local_mean : bool, default=True
        Se True, plota as médias locais e limites de Z-score.
    changepoints : bool, default=True
        Se True, plota as linhas verticais nos changepoints.

    Retorna:
    -------
    None
        Exibe os gráficos selecionados com os elementos indicados.
    """
    # Variáveis para serem plotadas
    variables = ['throughput_download', 'throughput_upload', 'rtt_download', 'rtt_upload']
    if survival:
        variables.append('survival_probability')

    # Escalas fixas para cada tipo de variável
    ylims = {
        'throughput_download': (0, 950),
        'throughput_upload': (0, 950),
        'rtt_download': (0, 250),
        'rtt_upload': (0, 250),
        'survival_probability': (0, 1)
    }

    num_variables = len(variables)
    num_pairs = len(client_site_list)
    num_cols = 3
    num_rows = (num_pairs // num_cols + (1 if num_pairs % num_cols != 0 else 0)) * num_variables

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 3 * num_rows), sharex=True)
    axes = axes.reshape(num_rows, num_cols)

    for pair_idx, (client, site) in enumerate(client_site_list):
        col = pair_idx % num_cols
        start_row = (pair_idx // num_cols) * num_variables

        input_dir = f'datasets/ts_{data}_results/'
        file_name = f"{client}_{site}.parquet"
        file_path = os.path.join(input_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Arquivo para o cliente {client} e site {site} não encontrado.")
            continue

        # Carregar o arquivo Parquet
        df = pd.read_parquet(file_path)
        df = df.sort_values('timestamp')

        # Mapear cores para clusters, se a coluna existir
        cluster_colors = {0: 'red', 1: 'blue'}
        if 'cluster' in df.columns:
            df['color'] = df['cluster'].map(cluster_colors).fillna('gray')
        else:
            df['color'] = 'gray'  # Cor padrão se não houver clusters

        for var_idx, variable in enumerate(variables):
            ax = axes[start_row + var_idx, col]
            if variable != 'survival_probability':
                changepoint_column = f"{variable}_cp"

                for i in range(len(df) - 1):
                    ax.plot(df['timestamp'].iloc[i:i+2], df[variable].iloc[i:i+2],
                            color=df['color'].iloc[i], linewidth=1)

                if local_mean and f'{variable}_local_mean' in df.columns:
                    ax.plot(df['timestamp'], df[f'{variable}_local_mean'], linestyle='--', color='black', alpha=0.7)
                    ax.fill_between(df['timestamp'],
                                    df[f'{variable}_local_mean'] - 2 * df[f'{variable}_local_std'],
                                    df[f'{variable}_local_mean'] + 2 * df[f'{variable}_local_std'],
                                    color='black', alpha=0.1)

                if changepoints and changepoint_column in df.columns:
                    changepoints_timestamps = df['timestamp'][df[changepoint_column] == 1]
                    for cp in changepoints_timestamps:
                        ax.axvline(x=cp, color='green', linestyle=':', alpha=0.8)
            else:
                ax.plot(df['timestamp'], df[variable], color='purple', linewidth=1)

            ax.set_ylabel(variable)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(ylims[variable])

        # Adicionar título para a coluna
        axes[start_row, col].set_title(f'{client} - {site}')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_random_pairs(data, client_site_list, title, survival=True, local_mean=True, changepoints=True):
    """
    Plota os valores de todas as variáveis ao longo do tempo em gráficos separados, sincronizados no eixo X,
    para 3 pares (cliente, site) escolhidos aleatoriamente de uma única lista fornecida.

    Variáveis: throughput_download, throughput_upload, rtt_download, rtt_upload

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    client_site_list : list of tuples
        Lista contendo pares (cliente, site).
    survival : bool, default=True
        Se True, plota os valores de 'survival_probability' em um gráfico separado.
    local_mean : bool, default=True
        Se True, plota as médias locais e limites de Z-score.
    changepoints : bool, default=True
        Se True, plota as linhas verticais nos changepoints.

    Retorna:
    -------
    None
        Exibe os gráficos selecionados com os elementos indicados.
    """
    # Selecionar 3 pares aleatórios da lista
    # selected_pairs = random.sample(client_site_list, 3)

    # Variáveis para serem plotadas
    variables = ['throughput_download', 'throughput_upload', 'rtt_download', 'rtt_upload']

    # Escalas fixas para cada tipo de variável
    ylims = {
        'throughput_download': (0, 950),
        'throughput_upload': (0, 950),
        'rtt_download': (0, 250),
        'rtt_upload': (0, 250)
    }

    # Configuração da figura com subplots organizados por par
    num_variables = len(variables) + (1 if survival else 0)
    num_pairs = len(client_site_list)
    num_cols = 3 if num_pairs > 3 else num_pairs
    num_rows_pairs = num_pairs // 3 + (1 if num_pairs % 3 != 0 else 0)
    num_rows = num_rows_pairs * num_variables
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 3 * num_rows), sharex=True)

    for pair_idx, (client, site) in enumerate(client_site_list):
        input_dir = f'datasets/ts_{data}_results/'
        file_name = f"{client}_{site}.parquet"
        file_path = os.path.join(input_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Arquivo para o cliente {client} e site {site} não encontrado.")
            continue

        # Carregar o arquivo Parquet
        df = pd.read_parquet(file_path)

        # Ordenar o dataframe por timestamp
        df = df.sort_values('timestamp')

        # Configurações de cores e estilos
        linestyles = {'local_mean': '--', 'changepoints': ':'}
        cluster_colors = {0: 'red', 1: 'blue'}
        df['color'] = df['cluster'].map(cluster_colors).fillna('gray')

        for var_idx, variable in enumerate(variables):
            ax = axes[var_idx, pair_idx]
            changepoint_column = f"{variable}_cp"

            # Plot 1: Gráfico principal com clusters e variáveis locais
            for i in range(len(df) - 1):
                ax.plot(df['timestamp'].iloc[i:i+2], df[variable].iloc[i:i+2],
                        color=df['color'].iloc[i], linewidth=1)

            for cluster, color in cluster_colors.items():
                ax.plot([], [], color=color, label=f'Cluster {cluster}')

            # Plotar média local e limites, se necessário
            if local_mean and f'{variable}_local_mean' in df.columns:
                ax.plot(df['timestamp'], df[f'{variable}_local_mean'], linestyle=linestyles['local_mean'], color='black', alpha=0.7, label=f'Média local')
                ax.fill_between(df['timestamp'],
                                df[f'{variable}_local_mean'] - 2 * df[f'{variable}_local_std'],
                                df[f'{variable}_local_mean'] + 2 * df[f'{variable}_local_std'],
                                color='black', alpha=0.1)

            # Plotar changepoints
            if changepoints and changepoint_column in df.columns:
                changepoints_timestamps = df['timestamp'][df[changepoint_column] == 1]
                for cp in changepoints_timestamps:
                    ax.axvline(x=cp, color='green', linestyle=linestyles['changepoints'], alpha=0.8)

            ax.set_ylabel(variable)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(ylims[variable])

        # Plot survival
        if survival:
            ax = axes[-1, pair_idx]
            ax.plot(df['timestamp'], df['survival_probability'], color='purple', label='Função de sobrevivência', linewidth=1)
            ax.set_xlabel('Tempo')
            ax.set_ylabel('Probabilidade')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.legend()

        # Adicionar título para cada coluna (par cliente-site)
        axes[0, pair_idx].set_title(f'{client} - {site}')

    # Configurações finais para a figura
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_one_feature(data, client, site, variable, survival=True, local_mean=True, changepoints=True, ylim=None):
    """
    Plota os valores de uma variável ao longo do tempo com opções para incluir changepoints, médias locais e 'survival'.

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
    survival : bool, default=True
        Se True, plota os valores de 'survival_probability' em um gráfico separado.
    local_mean : bool, default=True
        Se True, plota as médias locais e limites de Z-score.
    changepoints : bool, default=True
        Se True, plota as linhas verticais nos changepoints.
    ylim : tuple or None, optional
        Limites do eixo Y no formato (y_min, y_max). Se None, os limites serão automáticos.

    Retorna:
    -------
    None
        Exibe os gráficos selecionados com os elementos indicados.
    """
    input_dir = f'datasets/ts_{data}_results/'
    file_name = f"{client}_{site}.parquet"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Arquivo para o cliente {client} e site {site} não encontrado.")
        return

    # Carregar o arquivo Parquet
    df = pd.read_parquet(file_path)

    # Verificar colunas necessárias
    changepoint_column = f"{variable}_cp"
    if variable not in df.columns or 'cluster' not in df.columns:
        print(f"A variável '{variable}' ou os clusters não estão disponíveis no arquivo.")
        return
    if survival and 'survival_probability' not in df.columns:
        print("A coluna 'survival_probability' não está disponível no arquivo.")
        return
    if changepoints and changepoint_column not in df.columns:
        print(f"Os changepoints '{changepoint_column}' não estão disponíveis no arquivo.")
        return

    # Configurações de cores para clusters
    cluster_colors = {0: 'red', 1: 'blue'}
    df['color'] = df['cluster'].map(cluster_colors).fillna('gray')

    # Calcular média local e limites, se necessário
    if local_mean:
        df['upper_limit'] = df[f'{variable}_local_mean'] + 2 * df[f'{variable}_local_std']
        df['lower_limit'] = df[f'{variable}_local_mean'] - 2 * df[f'{variable}_local_std']

    # Obter changepoints, se necessário
    if changepoints:
        changepoints_timestamps = df['timestamp'][df[changepoint_column] == 1]

    # Criar a figura com dois subplots apenas se survival for True
    if survival:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax = [ax]  # Transforma em lista para unificar a manipulação

    # Plot 1: Gráfico principal com clusters e variáveis locais
    for i in range(len(df) - 1):
        ax[0].plot(df['timestamp'].iloc[i:i+2], df[variable].iloc[i:i+2],
                   color=df['color'].iloc[i], linewidth=1)

    for cluster, color in cluster_colors.items():
        ax[0].plot([], [], color=color, label=f'Cluster {cluster}')

    if local_mean:
        ax[0].plot(df['timestamp'], df[f'{variable}_local_mean'], color='gray', label='Média Local', linewidth=1)
        ax[0].plot(df['timestamp'], df['upper_limit'], linestyle='--', color='gray', label='Z-score +2', alpha=0.7, linewidth=1)
        ax[0].plot(df['timestamp'], df['lower_limit'], linestyle='--', color='gray', label='Z-score -2', alpha=0.7, linewidth=1)

    if changepoints:
        for cp in changepoints_timestamps:
            ax[0].axvline(x=cp, color='green', linestyle='--', linewidth=1, label='Changepoint' if cp == changepoints_timestamps.iloc[0] else '')

    ax[0].set_ylabel(variable)
    ax[0].set_title(f"{variable} - {client} - {site}")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    if ylim:
        ax[0].set_ylim(ylim)

    # Plot 2: Gráfico 'survival', se solicitado
    if survival:
        ax[1].plot(df['timestamp'], df['survival_probability'], color='purple', label='Survival', linewidth=1)
        ax[1].set_xlabel('Tempo')
        ax[1].set_ylabel('Survival')
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

    # Configurações finais
    plt.tight_layout()
    plt.show()


def plot_one_pair(data, client, site, survival=True, local_mean=True, changepoints=True, ylim=None):
    """
    Plota os valores de todas as variáveis ao longo do tempo em gráficos separados, sincronizados no eixo X,
    com opções para incluir changepoints, médias locais, 'survival' e clusters. 
    Cores dos clusters são ajustadas dinamicamente ao longo do tempo.

    Variáveis: throughput_download, throughput_upload, rtt_download, rtt_upload
    
    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    client : str
        Identificador do cliente.
    site : str
        Identificador do site.
    survival : bool, default=True
        Se True, plota os valores de 'survival_probability' em um gráfico separado.
    local_mean : bool, default=True
        Se True, plota as médias locais e limites de Z-score.
    changepoints : bool, default=True
        Se True, plota as linhas verticais nos changepoints.
    ylim : tuple or None, optional
        Limites do eixo Y no formato (y_min, y_max). Se None, os limites serão automáticos.

    Retorna:
    -------
    None
        Exibe os gráficos selecionados com os elementos indicados.
    """
    # Variáveis para serem plotadas
    variables = ['throughput_download', 'throughput_upload', 'rtt_download', 'rtt_upload']

    input_dir = f'datasets/ts_{data}_results/'
    file_name = f"{client}_{site}.parquet"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Arquivo para o cliente {client} e site {site} não encontrado.")
        return

    # Carregar o arquivo Parquet
    df = pd.read_parquet(file_path)

    # Ordenar o dataframe por timestamp
    df = df.sort_values('timestamp')

    # Configurações de cores e estilos
    linestyles = {'local_mean': '--', 'changepoints': ':'}
    
    # Configurações de cores para clusters
    cluster_colors = {0: 'red', 1: 'blue'}
    df['color'] = df['cluster'].map(cluster_colors).fillna('gray')

    # Configuração da figura com subplots para cada variável
    num_plots = len(variables) + (1 if survival else 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    # Loop para plotar cada variável em um subplot separado
    for idx, variable in enumerate(variables):
        ax = axes[idx]
        changepoint_column = f"{variable}_cp"

        # Plot 1: Gráfico principal com clusters e variáveis locais
        for i in range(len(df) - 1):
            ax.plot(df['timestamp'].iloc[i:i+2], df[variable].iloc[i:i+2],
                    color=df['color'].iloc[i], linewidth=1)

        for cluster, color in cluster_colors.items():
            ax.plot([], [], color=color, label=f'Cluster {cluster}')
        
        # Plotar média local e limites, se necessário
        if local_mean and f'{variable}_local_mean' in df.columns:
            ax.plot(df['timestamp'], df[f'{variable}_local_mean'], linestyle=linestyles['local_mean'], color='black', alpha=0.7, label=f'Média local')
            ax.fill_between(df['timestamp'],
                            df[f'{variable}_local_mean'] - 2 * df[f'{variable}_local_std'],
                            df[f'{variable}_local_mean'] + 2 * df[f'{variable}_local_std'],
                            color='black', alpha=0.1)
        
        # Plotar changepoints
        if changepoints and changepoint_column in df.columns:
            changepoints_timestamps = df['timestamp'][df[changepoint_column] == 1]
            for cp in changepoints_timestamps:
                ax.axvline(x=cp, color='green', linestyle=linestyles['changepoints'], alpha=0.8)

        ax.set_ylabel(variable)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if ylim:
            ax.set_ylim(ylim)

    # Plot survival
    if survival:
        ax = axes[-1]
        ax.plot(df['timestamp'], df['survival_probability'], color='purple', label='Survival', linewidth=1)
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Survival')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Configurações finais
    plt.tight_layout()
    plt.show()