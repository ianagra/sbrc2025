"""
Online changepoint detection module

    - Basic implementation: suffix 'ba'
    - Proposed implementation: suffix 'ps'
    - Functions:
        Shewhart: shewhart_ba, shewhart_ps
        Exponential Weighted Moving Average: ewma_ba, ewma_ps
        Two-sided CUSUM: cusum_2s_ba, cusum_2s_ps
        Window-Limited CUSUM: cusum_wl_ba, cusum_wl_ps
        Voting Windows Changepoint Detection: vwcd
        Bayesian Online Changepoint Detection: bocd_ba, bocd_ps
        Robust Random Cut Forest: rrcf_ps
        Non-Parametric Pelt: pelt_np

@author: Cleiton Moya de Almeida
"""

import numpy as np
from scipy.stats import shapiro, betabinom
from statsmodels.tsa.stattools import adfuller
from rpy2.robjects.packages import importr
changepoint_np = importr('changepoint.np')
changepoint = importr('changepoint')
import time

verbose = False

# Shapiro-Wilk normality test
# H0: normal distribution
def normality_test(y, alpha):
    _, pvalue = shapiro(y)
    return pvalue > alpha


# Augmented Dickey-Fuller test for unitary root (non-stationarity)
# H0: the process has a unit root (non-stationary)
def stationarity_test(y, alpha):
    adf = adfuller(y)
    pvalue = adf[1]
    return pvalue < alpha


# Compute the log-pdf for the normal distribution
# Obs.: the scipy built-in function logpdf does not use numpy and so is inneficient
def logpdf(x,loc,scale):
    c = 1/np.sqrt(2*np.pi)
    y = np.log(c) - np.log(scale) - (1/2)*((x-loc)/scale)**2
    return y


# Compute the log-likelihood value for the normal distribution
# Obs.: the scipy built-in function logpdf does not use numpy and so is inneficient
def loglik(x,loc,scale):
    n = len(x)
    c = 1/np.sqrt(2*np.pi)
    y = n*np.log(c/scale) -(1/(2*scale**2))*((x-loc)**2).sum()
    return y

# Voting Windows Changepoint Detection
def vwcd(X, w, w0, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Voting Windows Changepoint Detection
   
    Parameters:
    ----------
    X (numpy array): the input time-series
    w (int): sliding window size
    w0 (int): pre-chage estimating window size
    h (float): statistic threshold (in terms of sigma0)
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    M0 (list): estimated mean of the segments
    S0 (list): estimated standar deviation of the segments
    elapsedTime (float): running-time  (microseconds)
    """
    
    # Auxiliary functions
    # Compute the window posterior probability given the log-likelihood and prior
    # using the log-sum-exp trick
    def pos_fun(ll, prior, tau):
        c = np.nanmax(ll)
        lse = c + np.log(np.nansum(prior*np.exp(ll - c)))
        p = ll[tau] + np.log(prior[tau]) - lse
        return np.exp(p)

    # Aggregate a list of votes - compute the posterior probability
    def votes_pos(vote_list, prior_v):
        vote_list = np.array(vote_list)
        prod1 = vote_list.prod()*prior_v
        prod2 = (1-vote_list).prod()*(1-prior_v)
        p = prod1/(prod1+prod2)
        return p

    # Prior probabily for votes aggregation
    def logistic_prior(x, w, y0, yw):
        a = np.log((1-y0)/y0)
        b = np.log((1-yw)/yw)
        k = (a-b)/w
        x0 = a/k
        y = 1./(1+np.exp(-k*(x-x0)))
        return y
    
    # Auxiliary variables
    N = len(X)
    vote_n_thr = np.floor(w*vote_n_thr)

    # Prior probatilty for a changepoint in a window - Beta-B
    i_ = np.arange(0,w-3)
    prior_w = betabinom(n=w-4,a=ab,b=ab).pmf(i_)

    # prior for vot aggregation
    x_votes = np.arange(1,w+1)
    prior_v = logistic_prior(x_votes, w, y0, yw) 

    votes = {i:[] for i in range(N)} # dictionary of votes 
    votes_agg = {}  # aggregated voteylims

    lcp = 0 # last changepoint
    CP = [] # changepoint list
    M0 = [] # list of post-change mean
    S0 = [] # list of post-change standard deviation

    startTime = time.time()
    for n in range(N):
        if n>=w-1:
            
            # estimate the paramaters (w0 window)
            if n == lcp+w0:
                # estimate the post-change mean and variace
                m_w0 = X[n-w0+1:n+1].mean()
                s_w0 = X[n-w0+1:n+1].std(ddof=1)
                M0.append(m_w0)
                S0.append(s_w0)
            
            # current window
            Xw = X[n-w+1:n+1]
            
            LLR_h = []
            for nu in range(1,w-3+1):
            #for nu in range(w):
                # MLE and log-likelihood for H1
                x1 = Xw[:nu+1] #Xw até nu
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                if np.round(s1,3) == 0:
                    s1 = 0.001
                logL1 = loglik(x1, loc=m1, scale=s1)
                
                # MLE and log-likelihood  for H2
                x2 = Xw[nu+1:]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                if np.round(s2,3) == 0:
                    s2 = 0.001
                logL2 = loglik(x2, loc=m2, scale=s2)

                # log-likelihood ratio
                llr = logL1+logL2
                LLR_h.append(llr)

            
            # Compute the posterior probability
            LLR_h = np.array(LLR_h)
            pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w-3)]
            pos = [np.nan] + pos + [np.nan]*2
            pos = np.array(pos)
            
            # Compute the MAP (vote)
            p_vote_h = np.nanmax(pos)
            nu_map_h = np.nanargmax(pos)
            
            # Store the vote if it meets the hypothesis test threshold
            if p_vote_h >= p_thr:
                j = n-w+1+nu_map_h # Adjusted index 
                votes[j].append(p_vote_h)
            
            # Aggregate the votes for X[n-w+1]
            votes_list = votes[n-w+1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes-1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                votes_agg[n-w+1] = agg_vote
                
                # Decide for a changepoit
                if agg_vote > vote_p_thr:
                    if verbose: print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n-w+1 # last changepoint
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime-startTime
    return CP, M0, S0, elapsedTime