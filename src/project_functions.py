import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset

import os
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

from itertools import combinations
import networkx as nx


def get_upper_threshold(close):
    difference = close.diff()
    difference[0] = 0
    difference = difference.abs()
    bins = pd.cut(difference, bins=10)
    bins = bins.value_counts().to_frame().reset_index()
    bins["index"] = bins["index"].apply(lambda x: x.right)
    bins = bins.to_numpy()
    percentile_count = len(difference) * 0.85
    count = 0
    for i in range(10):
        count += bins[i, 1]
        if count > percentile_count:
            return bins[i, 0]


def get_entropy(labels, base=None):
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    base = math.e if base is None else base
    return -(vc * np.log(vc)/np.log(base)).sum()


def get_threshold(close, stepsize=0.00001):
    difference = close.diff()
    difference = difference.drop(0)
    difference = difference.tolist()

    threshold = 0
    thres_upper_bound = get_upper_threshold(close)
    temp_thres = 0
    best_entropy = -float('inf')

    while temp_thres < thres_upper_bound:
        labels = []
        for diff in difference:
            if diff > temp_thres:
                labels.append(2)
            elif -diff > temp_thres:
                labels.append(1)
            else:
                labels.append(0)
        entropy = get_entropy(labels)
        if entropy > best_entropy:
            best_entropy = entropy
            threshold = temp_thres
        temp_thres = temp_thres + stepsize
    return np.round(threshold,5)




def get_prediction_hybrid(row, original=False):
    out = None

    if original:
        if (row['pred_technical'] == 1):
            out = 1
        elif (row['pred_fundamental'] == 1):
            out = 1
        elif row['pred_technical'] == row['pred_fundamental']:
            out = row['pred_technical']
        else:
            if row['score_technical'] >= row['score_fundamental']:
                out = row['pred_technical']
            else:
                out = row['pred_fundamental']
    else:
        if row['pred_technical'] == row['pred_fundamental']:
            out = row['pred_technical']
        else:
            out = 1

    return out


def get_prediction_hybrid_max(row):
    out = None
    if row['score_technical'] >= row['score_fundamental']:
        out = row['pred_technical']
    else:
        out = row['pred_fundamental']

    return out


def get_prediction_hybrid_greedy(row):
    out = None
    if row['pred_technical'] == row['pred_fundamental']:
        out = row['pred_technical']
    elif row['pred_technical'] == 1:
        out = row['pred_fundamental']
    elif row['pred_fundamental'] == 1:
        out = row['pred_technical']
    else:
        if row['score_technical'] >= row['score_fundamental']:
            out = row['pred_technical']
        else:
            out = row['pred_fundamental']

    return out




def get_prediction_hybrid_regression(row, technical_mse, fundamental_mse):
    out = None

    if (row['pred_technical'] == 1):
        out = 1
    elif (row['pred_fundamental'] == 1):
        out = 1
    elif row['pred_technical'] == row['pred_fundamental']:
        out = row['pred_technical']
    else:
        if technical_mse <= fundamental_mse:
            out = row['pred_technical']
        else:
            out = row['pred_fundamental']

    return out



def get_profit_accuracy(test_df, col_pred, col_target_gains):

    true_dec = np.sum((test_df[col_pred] == 0) * (test_df[col_target_gains] < 0) * 1)
    true_inc = np.sum((test_df[col_pred] == 2) * (test_df[col_target_gains] > 0) * 1)
    false_dec = np.sum((test_df[col_pred] == 0) * (test_df[col_target_gains] > 0) * 1)
    false_inc = np.sum((test_df[col_pred] == 2) * (test_df[col_target_gains] < 0) * 1)

    profit_accuracy = (true_dec + true_inc) / (true_dec + true_inc + false_dec + false_inc)
    pred_count = true_dec + true_inc
    total_count = len(test_df)

    short_accuracy = true_dec / (true_dec + false_dec)
    long_accuracy = true_inc / (true_inc + false_inc)

    return (profit_accuracy, pred_count, total_count, short_accuracy, long_accuracy)




def get_regression_pred_decision(diff, col_target_gains_thres):
    if diff > col_target_gains_thres:
        return 2
    if -diff > col_target_gains_thres:
        return 0
    else:
        return 1




def VMD(tsv, K, alpha=2000, tau=0, DC=False, init=1, tol=1e-7):
    """
    u,u_hat,omega = VMD(tsv, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.

    Input and Parameters:
    ---------------------
    tsv       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """

    if len(tsv) % 2:
       tsv = tsv[1:]

    # Period and sampling frequency of input signal
    fs = 1. / len(tsv)

    ltemp = len(tsv) // 2
    fMirr = np.append(np.flip(tsv[:ltemp], axis=0), tsv)
    fMirr = np.append(fMirr,np.flip(tsv[-ltemp:], axis=0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T+1) / T

    # Spectral Domain discretization
    freqs = t - 0.5 - (1/T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[:T//2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])

    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K) * (i)
    elif init == 2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    else:
        omega_plus[0,:] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)

    # other inits
    uDiff = tol+np.spacing(1)
    n = 0
    sum_uk = 0
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)

    #*** Main loop for iterative updates***
    while ( uDiff > tol and  n < Niter-1 ):
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]

        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1.+Alpha[k]*(freqs - omega_plus[n,k])**2)

        # update first omega if not held at 0
        if not(DC):
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)

        # update of any other mode
        for k in np.arange(1,K):
            #accumulator
            sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]
            # mode spectrum
            u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n,k])**2)
            # center frequencies
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)

        # Dual ascent
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(np.sum(u_hat_plus[n+1,:,:],axis = 1) - f_hat_plus)

        # loop counter
        n = n+1

        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*np.dot((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i]),np.conj((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])))

        uDiff = np.abs(uDiff)


    # discard empty space if converged early
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:]

    idxs = np.flip(np.arange(1,T//2+1), axis=0)
    # Signal reconstruction
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T//2:T,:] = u_hat_plus[Niter-1,T//2:T,:]
    u_hat[idxs,:] = np.conj(u_hat_plus[Niter-1,T//2:T,:])
    u_hat[0,:] = np.conj(u_hat[-1,:])

    u = np.zeros([K,len(t)])
    for k in range(K):
        u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))

    # remove mirror part
    u = u[:, (T // 4):(3 * T // 4)]

    # # recompute spectrum
    # u_hat = np.zeros([u.shape[1], K], dtype=complex)
    # for k in range(K):
    #     u_hat[:,k] = np.fft.fftshift(np.fft.fft(u[k,:]))

    return u.T







def visibility_graph(tsv):

    g = nx.Graph()

    # convert list of magnitudes into list of tuples that hold the index
    tseries = []
    n = 0
    for magnitude in tsv:
        tseries.append( (n, magnitude ) )
        n += 1

    # contiguous time points always have visibility
    for n in range(0,len(tseries)-1):
        (ta, ya) = tseries[n]
        (tb, yb) = tseries[n+1]
        g.add_node(ta, mag=ya)
        g.add_node(tb, mag=yb)
        g.add_edge(ta, tb)

    for a, b in combinations(tseries, 2):
        # two points, maybe connect
        (ta, ya) = a
        (tb, yb) = b

        connect = True

        for tc, yc in tseries[ta+1:tb]:
            # other points, not a or b
            if tc != ta and tc != tb:
                # does c obstruct?
                if yc >= yb + (ya - yb) * ( (tb - tc) / (tb - ta) ):
                    connect = False

        if connect:
            g.add_edge(ta, tb)

    return g







