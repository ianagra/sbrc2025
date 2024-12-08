import pandas as pd
import numpy as np

def create_survival_dataset(df_cp, file_path):
    survival_data = []
    
    # Filtrar apenas as séries de throughput de download
    df_cp_download = df_cp[df_cp['serie'] == 'd_throughput']
    
    for _, row in df_cp_download.iterrows():
        client = row['client']
        site = row['site']
        serie = row['serie']
        changepoints = row['CP']
        m0_values = row['M0']
        s0_values = row['S0']
        
        # Montar o nome do arquivo de throughput de download
        file_name = f"{client}_{site}_d_throughput.txt"
        file = f"{file_path}/{file_name}"
        
        # Carregar os dados da série temporal de throughput de download
        try:
            time_series_download = pd.read_csv(file, names=['datetime', 'value'], parse_dates=['datetime'])
        except FileNotFoundError:
            print(f"File not found: {file}")
            continue
        
        # Verificar intervalos de tempo superiores a 3 dias
        time_series_download['time_diff'] = time_series_download['datetime'].diff().dt.total_seconds() / (60 * 60 * 24)
        long_gaps = time_series_download.index[time_series_download['time_diff'] > 3].tolist()
        
        # Dividir a série em subsequências baseadas nos long_gaps
        split_indices = [0] + long_gaps + [len(time_series_download)]
        
        # Função auxiliar para carregar outras séries
        def load_series(client, site, tipo, medida):
            try:
                file_name = f"{client}_{site}_{tipo}_{medida}.txt"
                file = f"{file_path}/{file_name}"
                return pd.read_csv(file, names=['datetime', 'value'], parse_dates=['datetime'])
            except FileNotFoundError:
                print(f"File not found: {file}")
                return pd.DataFrame(columns=['datetime', 'value'])
        
        # Carregar outras séries
        series_upload = load_series(client, site, 'u', 'throughput')
        series_rtt_download = load_series(client, site, 'd', 'rttmean')
        series_rtt_upload = load_series(client, site, 'u', 'rttmean')
        
        # Calcular survival data para cada subsequência
        for start, end in zip(split_indices[:-1], split_indices[1:]):
            sub_series = time_series_download.iloc[start:end]
            
            # Adicionar índices do início e do final ao changepoints
            changepoints = [0] + [cp for cp in changepoints if start <= cp < end] + [len(sub_series) - 1]
            
            for i in range(len(changepoints) - 1):
                start_idx = changepoints[i]
                end_idx = changepoints[i + 1]
                
                if start_idx >= end_idx:  # Subsequência vazia ou intervalo inválido
                    continue
                
                # Tempo inicial e final
                start_time = sub_series['datetime'].iloc[start_idx]
                end_time = sub_series['datetime'].iloc[end_idx]
                duration = (end_time - start_time).total_seconds() / (60 * 60 * 24)  # Duration in days
                
                # Primeiro valor após o changepoint
                first_value_download = sub_series['value'].iloc[start_idx]
                
                # Encontrar o primeiro valor das outras medidas no intervalo
                def get_first_value(series, start_time):
                    if not series.empty:
                        subset = series[series['datetime'] >= start_time]
                        if not subset.empty:
                            return subset['value'].iloc[0]
                    return np.nan
                
                first_value_upload = get_first_value(series_upload, start_time)
                first_value_rtt_download = get_first_value(series_rtt_download, start_time)
                first_value_rtt_upload = get_first_value(series_rtt_upload, start_time)
                
                # Checar censura
                if i == len(changepoints) - 2 or (end - start) < len(sub_series):  # Último intervalo ou subsequência dividida
                    event = 0
                else:
                    event = 1
                
                # Adicionar linha ao dataset
                survival_data.append({
                    'client': client,
                    'site': site,
                    'time': duration,
                    'throughput_download': first_value_download,
                    'throughput_upload': first_value_upload,
                    'rtt_download': first_value_rtt_download,
                    'rtt_upload': first_value_rtt_upload,
                    'event': event
                })
    
    # Converter para DataFrame
    survival_df = pd.DataFrame(survival_data)

    # One-hot encoding das colunas 'client' e 'site'
    survival_df = pd.get_dummies(survival_df, columns=['client', 'site'])

    # Converter colunas de dummies para inteiros
    for col in survival_df.columns:
        if col.startswith('client_') or col.startswith('site_'):
            survival_df[col] = survival_df[col].astype(int)
    
    return survival_df

df_cp = pd.read_pickle('../VWCD/changepoints_ndt.pkl')
timeseries_path = "../datasets/ts_ndt"  # Caminho dos arquivos .txt
survival_df = create_survival_dataset(df_cp, timeseries_path)
survival_df.to_pickle('survival_dataset.pkl')