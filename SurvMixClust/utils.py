import time
from datetime import timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from lifelines import KaplanMeierFitter


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

def load_obj(name):
    """Retrieve a pickle object from the datasets folder.
    It is used to load the data in order to run the experiments.

    Parameters
    ----------
    name : str
        File name.

    Returns
    -------
    object
        Stored object.
    """
    import pickle
    with open('./datasets/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def y_array(E, T):
    """Utility function to get a single label array from the event and time arrays.

    Parameters
    ----------
    E : np.array
        Array of booleans indicating if the event happened or not.
    T : np.array
        Array of floats indicating the time of the event.

    Returns
    -------
    y : np.array
        Array of tuples (time, event) containing the time and event status of each sample.
        You can access each array with y['time'] and y['event'].
    """
    y = [(bool(x), y) for x,y in zip(E,T)]
    y = np.array(y ,  dtype=[('event', '?'), ('time', '<f8')])
    return y

def X_Xd_from_set(subset, features_used):
    """Utility function to get X and Xd from a subset of the data.
    The other functions in this module use this function.
    """
    
    subset = (subset[[c for c in subset.columns if c in features_used or c in ["time", "event"]]]).reset_index(drop=True)
    subset_d = subset[[n for n in subset.columns if n not in ["time", "event"]]].copy()
    
    return subset, subset_d

def approximate_derivative(t, kmf, global_fixed_bw):
    """Approximate derivate of the survival function in kmf using the survPresmooth package.

    Parameters
    ----------
    t : np.array
        Time values to calculate the derivative.
    kmf : KaplanMeierFitter.
        A fitted KaplanMeierFitter object from lifelines package.
    global_fixed_bw : np.array
        Bandwidth returned by the function select_bandwidth.

    Returns
    -------
    dict
        A dict with the estimate and the bandwidth used.
    """

    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as robjects
    survPresmooth = rpackages.importr('survPresmooth')
    r_presmooth = survPresmooth.presmooth

    x_est = robjects.FloatVector(t.copy())

    t = robjects.FloatVector(np.array(kmf.durations))
    delta = robjects.FloatVector(np.array(kmf.event_observed))

    X_ = pd.DataFrame({'time': t, 'event':delta})
    t = robjects.FloatVector(X_['time'].values)
    delta = robjects.FloatVector(X_['event'].values)


    fixed_bw = robjects.FloatVector(np.array(global_fixed_bw))
    r_ps = r_presmooth(t, delta, estimand = 'f', bw_selec = 'fixed', fixed_bw = fixed_bw, x_est = x_est) 


    return { 'estimate' : np.array(r_ps.rx2['estimate']), 'bandwidth': list(r_ps.rx2['bandwidth'])}

def select_bandwidth(X, frac=1.):
    """Generate a bandwidth using the survPresmooth package for data X.

    Parameters
    ----------
    X : np.array or pd.DataFrame
        Array of shape (n_samples, n_features+2).
        It includes two columns with the labels 'time' and 'event'.
        The model works best with already normalized data.
    frac : float, optional
        The fraction of data to use to calculate the bandwidth, by default 1.

    Returns
    -------
    list
        A list with the bandwidth given by the survPresmooth package for X.
    """
    
    if len(X) > 5000:
        print(f"\n{4*'#'} Dataset is unecessarily big for bandwidth calculation, subsampling with replacement to 1000. {4*'#'}")
        X = X.sample(n=1000, replace=False).copy()
    
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as robjects
    survPresmooth = rpackages.importr('survPresmooth')
    r_presmooth = survPresmooth.presmooth

    print(f"\n{8*'#'} Selecting bandwidth {8*'#'}")

    try: 

        print(f"\n{4*'#'} plug-in estimate with 100% of data {4*'#'}")

        X_ = X.copy()

        t = robjects.FloatVector(X_['time'].values)
        delta = robjects.FloatVector(X_['event'].values)

        x_est_python = np.linspace(X_['time'].min(), (X_['time'].max()), num=500)
        x_est = robjects.FloatVector(x_est_python.copy())

        # Only meaningful for density and hazard function estimation. Internally computed when NULL, the default
        r_ps = r_presmooth(t, delta, estimand = 'f', bw_selec = 'plug-in', x_est = x_est)

        print(f"{4*'#'} success {4*'#'}\n")

        return list(r_ps.rx2['bandwidth'])

    except Exception as error:

        print(f"{4*'#'} exception of type {type(error).__name__} ocurred with 100% of data {4*'#'}\n")
        
        max_number_tries = 50
         
        for n_tries in range(max_number_tries):

            try:

                print(f"{4*'#'} plug-in estimate with sub-sampled {int(frac*100)}% of data (n_tries: {n_tries+1}) {4*'#'}")

                if frac == 1.:
                    print(f"{4*'#'} sampling 100% with replacement (n_tries: {n_tries+1}) {4*'#'}")
                    X_ = X.sample(n=int(len(X)), replace=True).copy()
                if frac < 1. :
                    print(f"{4*'#'} sampling {int(frac*100)}% without replacement (n_tries: {n_tries+1}) {4*'#'}")
                    X_ = X.sample(n=int(len(X)*frac), replace=False).copy()                       
                
                t = robjects.FloatVector(X_['time'].values)
                delta = robjects.FloatVector(X_['event'].values)

                x_est_python = np.linspace(X_['time'].min(), (X_['time'].max()), num=500)
                x_est = robjects.FloatVector(x_est_python.copy())

                # Only meaningful for density and hazard function estimation. Internally computed when NULL, the default
                r_ps = r_presmooth(t, delta, estimand = 'f', bw_selec = 'plug-in', x_est = x_est)

                print(f"{4*'#'} success for {list(r_ps.rx2['bandwidth'])} {4*'#'}\n")

                return list(r_ps.rx2['bandwidth'])

            except Exception as sub_error:

                print(f"{4*'#'} exception of type {type(sub_error).__name__} ocurred {4*'#'}\n")
                n_tries += 1
                
def cindex_km_from_model(X_target, Xd_target, logit, kmfs, global_fixed_bw):
    """_summary_

    Parameters
    ----------
    X_target : np.array or pd.DataFrame
        Array of shape (n_samples, n_features+2).
        It includes two columns with the labels 'time' and 'event'.
        The model works best with already normalized data.
    Xd_target : np.array or pd.DataFrame
        Array of shape (n_samples, n_features).
        It does not include the columns with the labels 'time' and 'event'. 
        The rest of the array is the same as X.
        The model works best with already normalized data.
    logit : LogisticRegression sklearn object
        Logistic Regression model trained for the features in Xd_target.
    kmfs : dict of kmfs objects.
        Information of the kmfs (non-parametric curves) for each cluster. 
        Check model.info['kmfs'] for details
    global_fixed_bw : np.array
        Bandwidth returned by the function select_bandwidth.

    Returns
    -------
    dict
        A dict with the c-index, integrated brier score and integrated negative binomial log-likelihood.
    """
    X = X_target.copy()
    Xd = Xd_target.copy()

    logit_proba = logit.predict_proba(Xd)
    ordered_labels = list(np.sort(logit.classes_))

    list_rns = list()
    for l in ordered_labels:

        def survival_function(t):
            return (kmfs[l]['kmf']).survival_function_at_times(t).values            

        X[f'label_{l}_logit'] = logit_proba[:,l]

        t = X['time'].values

        a_d = approximate_derivative(t, kmf = kmfs[l]['kmf'], global_fixed_bw=global_fixed_bw)
        X[f'label_{l}_negative_derivative_survival'] = a_d['estimate']

        X[f'label_{l}_survival'] = survival_function(X['time'].values)

        X[f'label_{l}_rn'] = (X[f'label_{l}_negative_derivative_survival']**(X[f'event']))*\
                             (X[f'label_{l}_survival']**(1-X[f'event']))*\
                             (X[f'label_{l}_logit'])

        list_rns.append(f'label_{l}_rn')

    label_rn_list = [f'label_{l}_rn' for l in ordered_labels]
    ss = X[label_rn_list].sum(axis=1)

    surv_df_list = [kmfs[l]['kmf'].survival_function_ for l in ordered_labels]

    ee = surv_df_list
    indexes = ee[0].index
    for e in ee[1:]:
        indexes = indexes.union(e.index)

    for i,e in enumerate(ee):        
        ee[i] = e.reindex(indexes,method = 'ffill').bfill()        

    idx_all_zero = X[label_rn_list].sum(axis=1)==0

    X.loc[idx_all_zero,label_rn_list] = [1/len(ordered_labels)]*len(ordered_labels)    
    ss.loc[idx_all_zero] = 1

    import functools
    kmfs_list = [functools.reduce(lambda a, b: a+b, [ee[l]*(X[label_rn_list[l]].loc[idx]/ss.loc[idx]) for l in ordered_labels]) for idx in X.index]

    surv_df = pd.concat(kmfs_list, axis=1, ignore_index=True).\
                            bfill().ffill()
    
    # With surv_df we can calculate all needed metrics.
    from pycox.evaluation import EvalSurv
    durations_test, events_test = (X['time']).values.astype('float32'), (X['event']).values.astype('float32') 
    ev = EvalSurv(surv_df, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td()
    time_grid = np.linspace(X['time'].min(), X['time'].max(), 200)
    integrated_brier_score = ev.integrated_brier_score(time_grid)
    integrated_nbll = ev.integrated_nbll(time_grid)
 
    return {'cindex': c_index, 'integrated_brier_score': integrated_brier_score, 'integrated_nbll': integrated_nbll}  

def surv_df_from_model(Xd_target, logit, kmfs):
    """It utilizes the model in logit and kmfs to predict the survival function for each sample in X.

    Parameters
    ----------
    Xd_target : np.array or pd.DataFrame
        Array of shape (n_samples, n_features) containing the samples.
        The model works best with already normalized data.
    logit : LogisticRegression sklearn object
        Logistic Regression model trained for the features in Xd_target.
    kmfs : dict of kmfs objects.
        Information of the kmfs (non-parametric curves) for each cluster. 
        Check model.info['kmfs'] for details. 

    Returns
    -------
    pd.DataFrame 
        DataFrame having the predicted survival function for each sample in X. 
        The indexes represent time and columns indicate each sample in X.
    """
    
    X = Xd_target.copy()
    Xd = Xd_target.copy()

    logit_proba = logit.predict_proba(Xd)
    ordered_labels = list(np.sort(logit.classes_))

    list_rns = list()
    for l in ordered_labels:     

        X[f'label_{l}_logit'] = logit_proba[:,l]
        X[f'label_{l}_rn'] = X[f'label_{l}_logit']

        list_rns.append(f'label_{l}_rn')

    label_rn_list = [f'label_{l}_rn' for l in ordered_labels]
    ss = X[label_rn_list].sum(axis=1)

    surv_df_list = [kmfs[l]['kmf'].survival_function_ for l in ordered_labels]

    ee = surv_df_list
    indexes = ee[0].index
    for e in ee[1:]:
        indexes = indexes.union(e.index)

    for i,e in enumerate(ee):        
        ee[i] = e.reindex(indexes,method = 'ffill').bfill()        


    idx_all_zero = X[label_rn_list].sum(axis=1)==0

    X.loc[idx_all_zero,label_rn_list] = [1/len(ordered_labels)]*len(ordered_labels)    
    ss.loc[idx_all_zero] = 1

    import functools
    kmfs_list = [functools.reduce(lambda a, b: a+b, [ee[l]*(X[label_rn_list[l]].loc[idx]/ss.loc[idx]) for l in ordered_labels]) for idx in X.index]

    surv_df = pd.concat(kmfs_list, axis=1, ignore_index=True).\
                            bfill().ffill()

    return surv_df

def labels_from_model(Xd_target, logit):
    """Inference of labels using logit.

    Parameters
    ----------
    Xd_target : np.array or pd.DataFrame
        Array of shape (n_samples, n_features) containing the samples.
        The model works best with already normalized data.
    logit : LogisticRegression
        Logistic Regression model trained for the features in Xd_target.

    Returns
    -------
    np.array
        Clusterized labels for each sample in Xd_target using the model in logit.
    """
    
    X = Xd_target.copy()
    Xd = Xd_target.copy()

    logit_proba = logit.predict_proba(Xd)
    ordered_labels = list(np.sort(logit.classes_))

    list_rns = list()
    for l in ordered_labels:
        
        X[f'label_{l}_logit'] = logit_proba[:,l]
        X[f'label_{l}_rn'] = X[f'label_{l}_logit']
        
        list_rns.append(f'label_{l}_rn')
     
    labels = np.argmax(X[list_rns].values, axis=1)
        
    return labels

def info_flat_from_info_list_K(info_list_K):
    """Transforms info_list_K into a flat pd.Dataframe containing only the needed information.

    Parameters
    ----------
    info_list_K : dict of list of dicts
        Dict where the keys are the number of clusters and the values are lists of dicts.
        Each dict inside the list of dicts is the output of the function cluster_EM_mod.

    Returns
    -------
    pd.Dataframe
        Each run of the EM algorithm is a row in the pd.Dataframe.
    """
    
    columns = ['n_clusters', 'idx', 'cindex_val', 'cindex',
               'integrated_nbll_val', 'integrated_nbll',
               'integrated_brier_score_val', 'integrated_brier_score'
              ]

    info_flat = pd.DataFrame(columns=columns)
    for n_clusters in info_list_K.keys():

        for idx, info in enumerate(info_list_K[n_clusters]):

            info['ite'] = info['cindex_by_ite'][-1]['ite']

            info_list_K[n_clusters][idx] = info        

            if info['cindex_val_by_ite'][-1]['cindex'] != -1:

                info_flat = info_flat.append(pd.DataFrame(data=[[n_clusters, idx, info['cindex_val_by_ite'][-1]['cindex'], info['cindex_last'],\
                                              info['cindex_val_by_ite'][-1]['integrated_nbll'], info['cindex_by_ite'][-1]['integrated_nbll'],
                                              info['cindex_val_by_ite'][-1]['integrated_brier_score'], info['cindex_by_ite'][-1]['integrated_brier_score'],
                                              ]],
                                              columns=columns))

    return info_flat

def best_model_keys_from_info_list_K(info_list_K):
    """Get the keys of the best model in info_list_K.

    Parameters
    ----------
    info_list_K : dict of list of dicts
        Dict where the keys are the number of clusters and the values are lists of dicts.
        Each dict inside the list of dicts is the output of the function cluster_EM_mod.

    Returns
    -------
    tuple of int
        (n_cluster, idx) of the best model in info_list_K.
    """
    
    info_flat = info_flat_from_info_list_K(info_list_K)
    
    best_model_row = info_flat.sort_values(by=['cindex_val']).iloc[-1,:]

    sub_experiment_key = best_model_row['n_clusters']
    idx = best_model_row['idx']
    
    return sub_experiment_key, idx

def plot_kms_clusters(info, time_max, show_n_samples=True):
    """Plot the clusterized results for the model in info.

    Parameters
    ----------
    info : dict
        The dict returned by the function cluster_EM_mod.
    time_max : float
        The maximum time to show in the plot.
    show_n_samples : bool, optional
        If True it shows the number of samples inside each cluster, by default True.
        If False it does not show the number of samples inside each cluster.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The plotly figure containing the clusterized results. 
        The confidence intervals come from the Kaplan-Meier estimator.
    """

    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    pio.templates.default = "plotly_white"
    
    title = "The Kaplan Meiers of each cluster found for the training set."

    g = go.Figure()

    kmfs = info['kmfs']
    kmfs_p_dict = deepcopy(kmfs)

    for k,v in kmfs.items():

        kmf = v['kmf']    
        kmfs_single = {'confidence' : kmf.confidence_interval_,
                       'survival' : kmf.survival_function_,
                       'sample_size': len(kmf.event_observed)} 

        kmfs_p_dict[k] = kmfs_single

    kmfs_p = kmfs_p_dict

    k = list(kmfs_p.keys())

    trace = dict()
    annotations = list()
    colors = px.colors.qualitative.Set1
    name2color = {individual_sf:colors[i%len(colors)] for i,individual_sf in enumerate(k)}

    for i,cluster_name in enumerate(k):

        survival = kmfs_p_dict[cluster_name]['survival']
        confidences = kmfs_p_dict[cluster_name]['confidence']   

        x = survival.index.values.flatten()
        x_rev = x[::-1]

        y3 = survival.values.flatten()
        y3_upper = confidences.iloc[:,1].values.flatten()
        y3_lower = confidences.iloc[:,0].values.flatten()
        y3_lower = y3_lower[::-1]

        trace[(i,'confidence')] = go.Scatter(
            x=list(x)+list(x_rev),
            y=list(y3_upper)+list(y3_lower),
            fill='toself',
            fillcolor='rgba' + str(name2color[cluster_name])[3:-1]+ ', 0.3)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            legendgroup=str(cluster_name),
            name='CI 95%: '+ str(cluster_name),
            hoveron= 'points'
        )

        x_a = ((x.max()-x.min())/(len(k)+1))*(i+1)
        y_a = y3[abs(x-x_a).argmin()]
        
        if show_n_samples:

            annotations.append(dict(x=x_a,
                    y=y_a,
                    ax= 20,
                    ay = -20,
                    text=str(kmfs_p_dict[cluster_name]['sample_size']) + ' samples',
                    xref="x",
                    yref="y",showarrow=True,
                    font_size = 16,
                    font_color = 'black',
                    arrowhead=7))

        trace[(i,'survival')] = go.Scatter(
            x=x, y=y3,
            line_color=name2color[cluster_name],
            name=str(cluster_name),
            showlegend=True,
            legendgroup=str(cluster_name)
        )

    for i,key in enumerate(trace.keys()):
        o = trace[key]
        g.add_trace(o)
     

    if show_n_samples:
        g.update_layout(annotations = annotations, overwrite = True)

    g.update_yaxes(range=[0,1.0])
    g.update_xaxes(range=[0, time_max])

    g.update_layout(
                    title=dict(
                       text=title,
                        x = 0.5,
                        font_size = 15,
                        xanchor = 'center',
                        yanchor = 'middle'
                    ), xaxis_title = 'Time units',
                        yaxis_title = 'Probability',
                    legend = dict(title = 'Clusters'),
                    height = 500, width=950
                   )  
    return g

def cluster_EM(X, Xd, X_val, Xd_val,
                   n_clusters, n_iterations, global_fixed_bw):
    """

    Parameters
    ----------
    X : np.array or pd.DataFrame
        Array of shape (n_samples, n_features+2) for the training data.
        It includes two columns with the labels 'time' and 'event'.
        The model works best with already normalized data.
    Xd : np.array or pd.DataFrame
        Array of shape (n_samples, n_features) for the training data.
        It does not include the columns with the labels 'time' and 'event'. 
        The rest of the array is the same as X.
        The model works best with already normalized data.
    X_val : np.array or pd.DataFrame
        Array of shape (n_samples, n_features+2) for the validation data.
        It includes two columns with the labels 'time' and 'event'.
        The model works best with already normalized data.
    Xd_val : np.array or pd.DataFrame
        Array of shape (n_samples, n_features) for the validation data.
        It does not include the columns with the labels 'time' and 'event'. 
        The rest of the array is the same as X.
        The model works best with already normalized data.
    n_clusters : int
        Number of clusters for the clusterizations algorithm.
    n_iterations : int
        The maximum number of iterations for each full application of the EM algorithm, by default 60.
        If the c-index metric stops improving, the EM algorithm stops before reaching the maximum number of iterations.
    global_fixed_bw : np.array
        Bandwidth returned by the function select_bandwidth.

    Returns
    -------
    list of a single dict
        A list with a single dict having information about the EM run. 
    """

    info_list = list() 

    # Random cluster initialization
    new_random_init_n_clusters = n_clusters

    new_labels = np.random.randint(0,new_random_init_n_clusters,len(Xd))
    ordered_labels = np.sort(np.unique(new_labels))
    updated_clusters = {(label): list(np.argwhere(new_labels==label).flatten()) for label in ordered_labels}
    X['labels'] = new_labels

    new_medoids = {k:[] for k in range(new_random_init_n_clusters)}
    for l in ordered_labels:
        new_medoids[l] = Xd.loc[updated_clusters[l],:].mean().values


    # Generating the initial kmfs for the randomly initiated clusterization.
    clusters = updated_clusters.copy()
    idxs = list(clusters.keys())
    kmfs = dict()
    for i in idxs:
        time_km, event_km = X.iloc[clusters[i],:]['time'].astype('float32'), X.iloc[clusters[i],:]['event'].astype('float32')
        kmf = KaplanMeierFitter()  
        kmf.fit(time_km, event_km)

        kmfs[i] = {'kmf': kmf}


    # Initializing some variables.
    iterations = n_iterations
    cindex_by_ite = list()
    cindex_val_by_ite = list()

    try:

        # Main EM algorithm
        val_tests = 1
        time_algorithm = time.time()
        for ite in np.arange(iterations):

            time_max = time.time()

            print(f'######## iteration: {ite}')
            
            X['labels'] = new_labels  
            
            # Formally, this logistic regression still corresponds to the Maximization step.
            from sklearn.linear_model import LogisticRegression
            logit = LogisticRegression(max_iter=5000)
            logit.fit(Xd,new_labels)
            logit_proba = logit.predict_proba(Xd)

            ###################
            ### Expectation ###
            ###################
            
            if (ite%3) == 0:

                metrics = cindex_km_from_model(X, Xd, logit, kmfs, global_fixed_bw)

                snap_cindex = metrics['cindex']

                cindex_by_ite.append({'ite':ite, 'cindex':snap_cindex,
                                      'integrated_brier_score': metrics['integrated_brier_score'],
                                      'integrated_nbll': metrics['integrated_nbll']})


                print(f'\t# cindex (with labels) with train_set at iteration {ite}: {snap_cindex}')

                if val_tests:

                    metrics = cindex_km_from_model(X_val, Xd_val, logit, kmfs, global_fixed_bw)

                    snap_cindex = metrics['cindex']

                    cindex_val_by_ite.append({'ite':ite, 'cindex':snap_cindex,
                                              'integrated_brier_score': metrics['integrated_brier_score'],
                                              'integrated_nbll': metrics['integrated_nbll']})

                    print(f'\t# cindex (with labels) with val_set at iteration {ite}: {snap_cindex}')


            if (len(cindex_by_ite)>2):   
                if (abs(cindex_by_ite[-2]['cindex'] - cindex_by_ite[-1]['cindex'])<1e-6):
                    print(f"\t\tcindex_by_ite[-2]['cindex']: {cindex_by_ite[-2]['cindex']}, cindex_by_ite[-1]['cindex']: {cindex_by_ite[-1]['cindex']}")
                    print(f"\t\tcindex_val_by_ite[-2]['cindex']: {cindex_val_by_ite[-2]['cindex']}, cindex_val_by_ite[-1]['cindex']: {cindex_val_by_ite[-1]['cindex']}")
                    print(f'######## Converged: {ite} ########')
                    print(f'######## Time for all iterations:{timedelta(seconds=time.time() - time_algorithm)} ########\n\n')
                    break


            list_rns = list()
            for l in ordered_labels:

                def survival_function(t):
                    return (kmfs[l]['kmf']).survival_function_at_times(t).values

                X[f'label_{l}_logit'] = logit_proba[:,l]

                t = X['time'].values

                a_d = approximate_derivative(t, kmf = kmfs[l]['kmf'], global_fixed_bw=global_fixed_bw)
                X[f'label_{l}_negative_derivative_survival'] = a_d['estimate']
                X[f'label_{l}_survival'] = survival_function(X['time'].values)


                X[f'label_{l}_rn'] = (X[f'label_{l}_negative_derivative_survival']**(X[f'event']))*\
                                     (X[f'label_{l}_survival']**(1-X[f'event']))*\
                                     (X[f'label_{l}_logit'])


                list_rns.append(f'label_{l}_rn')

            X['labels'] = np.argmax(X[list_rns].values, axis=1)   


            ####################
            ### Maximization ###
            ####################

            new_labels = X['labels'].values.copy()
            updated_clusters = {(label): list(np.argwhere(new_labels==label).flatten()) for label in ordered_labels}

            for l in ordered_labels:
                new_medoids[l] = Xd.loc[updated_clusters[l],:].mean().values

            clusters = updated_clusters.copy()
            idxs = list(clusters.keys())
            kmfs = dict()

            empty_kmf = False
            for i in idxs:
                time_km, event_km = X.iloc[clusters[i],:]['time'].astype('float32'), X.iloc[clusters[i],:]['event'].astype('float32')
                if len(time_km) == 0:
                    empty_kmf = True
            if empty_kmf:
                print('EMPTY_kmf, stopped training.')
                break

            for i in idxs:
                time_km, event_km = X.iloc[clusters[i],:]['time'].astype('float32'), X.iloc[clusters[i],:]['event'].astype('float32')
                kmf = KaplanMeierFitter()  
                kmf.fit(time_km, event_km)

                kmfs[i] = {'kmf': kmf}

            print("\tTime elapsed for the entire iteration: ", timedelta(seconds=time.time() - time_max))

        if empty_kmf:
            print('Error for a single EM run. Skipped. Reason: empty_kmf.\n\n')
        else:

            dict_to_store = {'kmfs': deepcopy(kmfs), 'logit': logit, 'new_labels' : new_labels, 'global_fixed_bw' : global_fixed_bw,
                              'cindex_by_ite': deepcopy(cindex_by_ite),'cindex_last': cindex_by_ite[-1]['cindex'],
                              'cindex_val_by_ite': deepcopy(cindex_val_by_ite),'cindex_val_last': cindex_val_by_ite[-1]['cindex']
                              }

            info_list.append(dict_to_store)

    except Exception as error:

        # Show the error type and exit this EM run.
        print("An exception occurred:", type(error).__name__, "–", error)
        print('Error for a single EM run. Skipped.\n\n')


    return info_list