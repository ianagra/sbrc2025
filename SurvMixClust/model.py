from joblib import Parallel, delayed

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import pandas as pd

from SurvMixClust.utils import X_Xd_from_set, surv_df_from_model,\
    labels_from_model, best_model_keys_from_info_list_K, plot_kms_clusters, cluster_EM,\
    select_bandwidth


class SurvMixClust(BaseEstimator):
    """ClustSurvMix is a clustering algorithm for survival analysis.
    """

    def __init__(self, n_clusters, n_fits=4, max_EM_interations=60, n_jobs=None):
        """
        Parameters
        ----------
        n_clusters : int or list of ints
            If int, it is the number of clusters for the model to use for its training and inference.
            If list of ints, it uses each number of cluster for the model fit, then selecting the best fit, more details at model.fit function.
        n_fits : int, optional
            Number of completetly different EM fits done for each number of cluster in n_clusters, by default 5.
        max_EM_interations : int, optional
            The maximum number of iterations for each full application of the EM algorithm, by default 60.
            If the c-index metric stops improving, the EM algorithm stops before reaching the maximum number of iterations.
        n_jobs : int, optional
            If None, it fits the model sequentially in a normal manner, by default None.
            IF -1, it fits using all the available cores with joblib.Parallel.
            If a positive integer, it fits using the specified number of cores with joblib.Parallel.
        """

        if isinstance(n_clusters, int):
            self.n_clusters = [n_clusters]
        else:
            self.n_clusters = n_clusters
        self.n_fits = n_fits
        self.max_EM_interations = max_EM_interations
        self.n_jobs = n_jobs


        self.global_fixed_bw = (0, 0)
        self.time_max = 0

    def fit(self, X, y):
        """Fits the model to the training set (X, y). 
        First it divides X into a training and validation set.
        Internally it generate n_fits for each n_cluster in n_clusters.
        It then selects the best model based on the performance on the validation set.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Array of shape (n_samples, n_features) containing the training samples.
            The model works best with already normalized data.
        y : np.array
            Array of tuples (time, event) containing the time and event status of each sample.
            You can generate this array using the function datasets.y_array(E,T).
            Explicitly:
                def y_array(E, T):
                    y = [(bool(x), y) for x,y in zip(E,T)]
                    y = np.array(y ,  dtype=[('event', '?'), ('time', '<f8')])
                    return y
        """

        # Check that X and y have correct shape.
        X, y = check_X_y(X, y)

        split_ = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                                        random_state=np.random.randint(100))

        X_input, E_input = X.astype('float32'), y['event']

        train_set_index = np.array([])
        val_set_index = np.array([])
        for train_index, val_index in split_.split(X_input, E_input):

            train_set_index = train_index
            val_set_index = val_index

        X = pd.DataFrame(data=X)
        train_set = X.loc[train_set_index]
        train_set['time'] = y['time'][train_set_index]
        train_set['event'] = y['event'][train_set_index]

        val_set = X.loc[val_set_index]
        val_set['time'] = y['time'][val_set_index]
        val_set['event'] = y['event'][val_set_index]

        self.time_max = y['time'].max()

        features_used = train_set.columns[:]
        X, Xd = X_Xd_from_set(train_set, features_used)
        X_val, Xd_val = X_Xd_from_set(val_set, features_used)

        self.global_fixed_bw = select_bandwidth(X)
#         self.global_fixed_bw = [17.52130828203924, 12.733703749619613] # support
        # self.global_fixed_bw = [19.610724293026085,
        #                         65.89859497742401]  # metabric

        n_clusters = self.n_clusters
        number_parallel_tries = self.n_fits
        n_iterations = self.max_EM_interations
        n_jobs = self.n_jobs
        global_fixed_bw = self.global_fixed_bw

        info_list_K = dict()

        for n_cluster in n_clusters:

            if n_jobs is None:
                print(f"Non-parallel processing for {n_cluster} number of clusters.")
                list_list_dicts = list()

                for _ in range(number_parallel_tries):

                    info_list = cluster_EM(X, Xd, X_val, Xd_val,
                                               n_cluster,
                                               n_iterations, global_fixed_bw)
                    list_list_dicts.append(info_list)

                import functools
                info_list = functools.reduce(lambda a, b: a+b, list_list_dicts)

            else:
                print(f"Parallel processing for {n_cluster} number of clusters, n_jobs: {self.n_jobs}.")
                list_list_dicts = Parallel(n_jobs=n_jobs)(delayed(cluster_EM)(X, Xd, X_val, Xd_val,
                                                                                  n_cluster,
                                                                                  n_iterations, global_fixed_bw)
                                                          for ite_gen in np.arange(number_parallel_tries))

                import functools
                info_list = functools.reduce(lambda a, b: a+b, list_list_dicts)

            from copy import deepcopy
            info_list_K[(n_cluster)] = deepcopy(info_list)

        sub_experiment_key, idx = best_model_keys_from_info_list_K(info_list_K)

        self.info_list_K = info_list_K

        self.info = info_list_K[sub_experiment_key][idx]
        self.kmfs = info_list_K[sub_experiment_key][idx]['kmfs']
        self.logit = info_list_K[sub_experiment_key][idx]['logit']
        self.global_fixed_bw = info_list_K[sub_experiment_key][idx]['global_fixed_bw']

        self.X_ = 'X'
        self.y_ = 'y'

        # Return the model
        return self

    def predict_surv_df(self, X, n_cluster=None, n_fit=None):
        """It utilizes the best performing model obtained in model.fit to predict the survival function for each sample in X.
        You can choose a specific model from the list of models generated in model.fit by specifying n_cluster and n_fit.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Array of shape (n_samples, n_features) containing the samples to generate the survival functions.
            The model works best with already normalized data.
        n_cluster : int, optional
            The cluster number of the specific result to be used, by default None.
            If None, It defaults to using the clusterization of the best performing model obtained in model.fit.
        n_fit : int, optional
            The n_fit of the specific result to be used, by default None.
            If None, It defaults to using the clusterization of the best performing model obtained in model.fit.

        Returns
        -------
        pd.DataFrame 
            DataFrame having the predicted survival function for each sample in X. 
            The indexes represent time and columns indicate each sample in X.
        """

        # Check if fit had been called.
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X).astype('float32')

        Xd_target = pd.DataFrame(data=X)
    
        if (n_cluster is None) and (n_fit is None):
            surv_df = surv_df_from_model(
            Xd_target.copy(), self.logit, self.kmfs)
            
            return surv_df
        
        else:
            surv_df = surv_df_from_model(
            Xd_target.copy(), self.info_list_K[n_cluster][n_fit]['logit'], self.info_list_K[n_cluster][n_fit]['kmfs'])

            return surv_df

    def predict(self, X, n_cluster=None, n_fit=None):
        """Predicts the clusterized labels for each sample in X using the model fitted in model.fit.
        You can choose a specific model from the list of models generated in model.fit by specifying n_cluster and n_fit.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Array of shape (n_samples, n_features) containing the samples.
            The model works best with already normalized data.
        n_cluster : int, optional
            The cluster number of the specific result to be used, by default None.
            If None, It defaults to using the clusterization of the best performing model obtained in model.fit.
        n_fit : int, optional
            The n_fit of the specific result to be used, by default None.
            If None, It defaults to using the clusterization of the best performing model obtained in model.fit.

        Returns
        -------
        np.array
            Clusterized labels for each sample in X using the model fitted in model.fit.
        """

        # Check if fit had been called.
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation.
        X = check_array(X).astype('float32')

        Xd_target = pd.DataFrame(data=X)

        if (n_cluster is None) and (n_fit is None):
            labels = labels_from_model(
            Xd_target.copy(), self.logit)
            
            return labels
        
        else:
            labels = labels_from_model(
            Xd_target.copy(), self.info_list_K[n_cluster][n_fit]['logit'])

            return labels

    def training_set_labels(self):
        """Returns the clusterized labels for each sample in the training set for the best performing model obtained in model.fit.
        These labels were generated by the last iteration of the EM algorithm for the best performing model.

        Returns
        -------
        np.array
            Clusterized labels for each sample in the training set for the best performing model obtained in model.fit.
            These labels were generated by the last iteration of the EM algorithm for the best performing model.
        """
        
        # Check if fit had been called.
        check_is_fitted(self, ['X_', 'y_'])           
        
        return self.info['new_labels']
    
    def score(self, X, y, metric='cindex', n_cluster=None, n_fit=None):
        """Calculates the score of the model using the X and y provided.
        It accepts c-index and integrated brier score as metrics. 
        You can choose a specific model from the list of models generated in model.fit by specifying n_cluster and n_fit.

        Parameters
        ----------
        X : np.array or pd.DataFrame
            Array of shape (n_samples, n_features) containing the samples.
            The model works best with already normalized data.
        y : np.array
            Array of tuples (time, event) containing the time and event status of each sample.
            You can generate this array using the function datasets.y_array(E,T).
            Explicitly:
                def y_array(E, T):
                    y = [(bool(x), y) for x,y in zip(E,T)]
                    y = np.array(y ,  dtype=[('event', '?'), ('time', '<f8')])
                    return y
        n_cluster : int, optional
            The cluster number of the specific result to be used, by default None.
            If None, It defaults to using the clusterization of the best performing model obtained in model.fit.
        n_fit : int, optional
            The n_fit of the specific result to be used, by default None.
            If None, It defaults to using the clusterization of the best performing model obtained in model.fit.
        metric : str, optional
            If cindex, calculate the c-index score from survival analysis, by default 'cindex'.
            If ibs, calculate the integrated brier score from survival analysis.

        Returns
        -------
        float
            the score of the model using the X and y provided.
        """

        assert (metric in ['cindex', 'c-index',
                'integrated brier score', 'ibs'])
        

        if (n_cluster is None) and (n_fit is None):
            surv_df = self.predict_surv_df(X)
            
        else:
            surv_df = self.predict_surv_df(X, n_cluster=n_cluster, n_fit=n_fit)
        
        
        from pycox.evaluation import EvalSurv
        durations_test, events_test = \
            (y['time']).astype('float32'), (y['event']).astype('float32')
        ev = EvalSurv(surv_df, durations_test, events_test, censor_surv='km')

        if metric in ['cindex', 'c-index']:
            cindex = ev.concordance_td()
            metric_score = cindex
        else:
            time_grid = np.linspace(
                durations_test.min(), durations_test.max(), 100)
            ibs = ev.integrated_brier_score(time_grid)
            metric_score = ibs

        return metric_score

    def plot_clusters(self, n_cluster=None, n_fit=None, show_n_samples=True):
        """Plot the clusterized results for the best performing model obtained in model.fit, if n_cluster and n_fit are None.
        If n_cluster and n_fit are not None, it plots the clusterized results for the n_fit iteration of the n_cluster model.

        Parameters
        ----------
        n_cluster : int, optional
            The cluster number of the specific result to be shown, by default None.
            If None, It defaults to showing the clusterization of the best performing model obtained in model.fit.
        n_fit : int, optional
            The n_fit of the specific result to be shown, by default None.
            If None, It defaults to showing the clusterization of the best performing model obtained in model.fit.
        show_n_samples : bool, optional
            If True it shows the number of samples inside each cluster, by default True.
            If False it does not show the number of samples inside each cluster.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            The plotly figure containing the clusterized results. 
            The confidence intervals come from the Kaplan-Meier estimator.
        """

        if (n_cluster is None) and (n_fit is None):
            fig = plot_kms_clusters(
                self.info, self.time_max, show_n_samples=show_n_samples)
        else:
            fig = plot_kms_clusters(
                self.info_list_K[n_cluster][n_fit], self.time_max, show_n_samples=show_n_samples)

        return fig
