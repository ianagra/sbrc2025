from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import os
from pathlib import Path

def decode_one_hot(df, prefix):
    """
    Decodifica um conjunto de colunas one-hot encoded em uma única coluna com as labels originais.

    Parâmetros:
        df (pd.DataFrame): DataFrame com as colunas one-hot encoded.
        prefix (str): Prefixo das colunas one-hot encoded.

    Retorna:
        pd.DataFrame: DataFrame com as colunas decodificadas.
    """
    cols = [col for col in df.columns if col.startswith(prefix)]
    decoded_column = df[cols].idxmax(axis=1).str.replace(prefix, '', regex=False)
    df = df.drop(columns=cols).assign(**{prefix.replace('_',''): decoded_column})
    return df


def assign_labels(survival_df, data, timestamp_col='timestamp', survival_functions=None, multivariate=False):
    """
    Associa os rótulos de clusters gerados pelo SurvMixClust às séries temporais originais.

    Parâmetros:
    ----------
    survival_df : pd.DataFrame
        Dataset de sobrevivência com os rótulos dos clusters adicionados.
    series_dir : str
        Diretório contendo as séries temporais originais em formato Parquet.
    output_dir : str
        Diretório onde as séries temporais atualizadas serão salvas.
    timestamp_col : str, default='timestamp'
        Nome da coluna de timestamp nas séries temporais.
    survival_functions : pd.DataFrame, default=None
        Funções de sobrevivência para cada intervalo de tempo.
    multivariate : bool, default=False
        Se True, considera os changepoints detectados com a abordagem multivariada do VWCD.

    Retorna:
    -------
    None
        Salva as séries temporais atualizadas com os rótulos dos clusters no diretório `output_dir`.
    """
    if multivariate:
        series_dir = f'datasets/ts_{data}_cp_mv/'
        output_dir = f'datasets/ts_{data}_results_mv/'
    else:
        series_dir = f'datasets/ts_{data}_cp/'
        output_dir = f'datasets/ts_{data}_results/'

    os.makedirs(output_dir, exist_ok=True)

    # Criar colunas para as médias e desvios padrão locais e a taxa de outliers
    for feature in ['rtt_download', 'rtt_upload', 'throughput_download', 'throughput_upload']:
        #survival_df[f'{feature}_local_mean'] = -1
        #survival_df[f'{feature}_local_std'] = -1
        survival_df[f'{feature}_outliers_rate'] = -1

    for file in os.listdir(series_dir):
        if file.endswith(".parquet"):
            series_df = pd.read_parquet(os.path.join(series_dir, file))
            client, site = file.split('.')[0].split('_', 1)

            # Analisar os intervalos do dataset de sobrevivência correspondentes ao par cliente-site
            if f'client_{client}' in survival_df.columns and f'site_{site}' in survival_df.columns:
                intervals = survival_df[(survival_df[f'client_{client}'] == 1) & 
                                        (survival_df[f'site_{site}'] == 1)]
            else:
                print(f"Colunas 'client_{client}' ou 'site_{site}' não encontradas. Pulando {file}.")
                continue

            series_df[timestamp_col] = pd.to_datetime(series_df[timestamp_col])
            series_df['cluster'] = -1
            series_df['cluster_probability'] = -1
            series_df['survival_probability'] = -1

            for idx, row in intervals.iterrows():
                start_time = pd.to_datetime(row['timestamp_start'])
                end_time = pd.to_datetime(row['timestamp_end'])
                cluster_label = row['cluster']
                cluster_prob = row.get(f'probability_cluster_{cluster_label}', -1)

                mask = (series_df[timestamp_col] >= start_time) & (series_df[timestamp_col] <= end_time)
                series_df.loc[mask, 'cluster'] = cluster_label
                series_df.loc[mask, 'cluster_probability'] = cluster_prob

                for _, row_series in series_df[mask].iterrows():
                    # Interpolando a função de sobrevivência
                    time = row_series[timestamp_col] - start_time
                    time = time.total_seconds() / (60 * 60 * 24)  # Converter para dias
                    
                    # Interpolação linear para encontrar a probabilidade de sobrevivência para o tempo 'time'.
                    surv_prob = np.interp(
                        time, 
                        survival_functions.index.to_numpy().flatten(),
                        survival_functions[idx].values.flatten()
                    )

                    # Atribuir a probabilidade de sobrevivência à coluna 'survival' da entrada correspondente
                    series_df.loc[series_df[timestamp_col] == row_series[timestamp_col], 'survival_probability'] = surv_prob
                    
                # Média e desvio padrão local das métricas
                for feature in ['rtt_download', 'rtt_upload', 'throughput_download', 'throughput_upload']:
                    local_mean = series_df.loc[mask, feature].mean()
                    local_std = series_df.loc[mask, feature].std()

                    if pd.notnull(local_std) and local_std != 0:
                        z_scores = (series_df.loc[mask, feature] - local_mean) / local_std
                        outliers_rate = (z_scores > 3).sum() / len(z_scores)
                    else:
                        outliers_rate = 0

                    #series_df.loc[mask, f'{feature}_local_mean'] = local_mean
                    #series_df.loc[mask, f'{feature}_local_std'] = local_std
                    series_df.loc[mask, f'{feature}_outliers_rate'] = outliers_rate

                    # Atribuir a média e desvio padrão locais e a taxa de outliers à entrada correspondente no surival_df
                    #survival_df.loc[idx, f'{feature}_local_mean'] = local_mean
                    #survival_df.loc[idx, f'{feature}_local_std'] = local_std
                    survival_df.loc[idx, f'{feature}_outliers_rate'] = outliers_rate

            series_df.to_parquet(os.path.join(output_dir, file), index=False)

    print(f"Clusters associados e séries temporais salvas em: {output_dir}")

    return survival_df