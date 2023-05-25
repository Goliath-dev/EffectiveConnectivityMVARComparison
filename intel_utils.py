# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:23:57 2022

@author: Admin
"""

import csv
import numpy as np
import scipy as sp
import mne
import glob
import matplotlib.pyplot as plt
import networkx as nx

fmin_arr = (4, 4,  8,  8,  10, 13, 20, 30, 30, 4)
fmax_arr = (8, 30, 13, 10, 13, 20, 30, 40, 45, 45)
# A weirdo for the sake of usability. 
freq_idcs = {(fmin, fmax): i for i, (fmin, fmax) in enumerate(zip(fmin_arr, fmax_arr))}

def modularity_curried(community_method):
    def func(G, weight):
        return nx.community.modularity(G, community_method(G), weight)
    return func
topology_methods = {'Shortest path length': nx.algorithms.average_shortest_path_length,
           'Clustering': nx.algorithms.average_clustering,
           'Modularity': modularity_curried(nx.algorithms.community.louvain_communities)}
topology_method_names = topology_methods.keys()

def read_csv(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        arr = np.array([line for line in reader])
    return arr

def get_topology(files, freq_of_interest, 
                 methods=topology_method_names):
    metrics_values = np.zeros((len(files), len(freq_of_interest), len(methods)))
    for i, file in enumerate(files):
        conn = np.load(file)
        for j, freq in enumerate(freq_of_interest):
            conn_freq = conn[j][conn[j] != 0]
            conn_matrix = sp.spatial.distance.squareform(conn_freq)
            G = nx.from_numpy_array(conn_matrix, create_using=nx.Graph)
            for k, method_name in enumerate(methods):
                    metric = topology_methods[method_name](G, weight='weight')
                    metrics_values[i, j, k] = metric
    return metrics_values
    
def get_Cuban_topology(freq_of_interest, conn_method,
                       topology_methods=topology_method_names):
    """
    Computes the topology metrics for the conncetivity graphs using the Cuban
    sample in frequency bands defined by freq_of_interest. 

    Parameters
    ----------
    freq_of_interest : Nx1 array
        An array of frequency band indices. Can be defined either by extracting
        indices from fmin_arr and fmax_arr manually or by using freq_idcs dictionary.
    conn_method : string
        A connectivity method to use. Available options are 'wPLI', 'PLV' and 'ciPLV'.
    topology_methods : Kx1 array of strings, optional
        Names of topology metrics to calculate. Available options are 
        'Shortest path length', 'Clustering' and 'Modularity'. The default is 
        topology_method_names, that is, all of these methods.

    Returns
    -------
    MxNxK array
        An array of values of topology metrics for M participants, N frequency
        bands and K topology metrics.

    """
    data_dir = f'E:\Work\MBS\EEG\Modelling\Intelligence\Cuban Map Project\Matrices\{conn_method}'
    files = glob.glob(data_dir + '\\CBM*.npy')
    return get_topology(files, freq_of_interest, topology_method_names)

def get_spectrum(files, fmin_arr=fmin_arr, fmax_arr=fmax_arr, picked_chs=['eeg']):
    psd_arr = np.zeros((len(files), len(fmin_arr)))
    for i, file in enumerate(files):
        epochs = mne.read_epochs(file)
        epoch_by_chs = epochs.pick(picked_chs)
        for j, (fmin, fmax) in enumerate(zip(fmin_arr, fmax_arr)):
            psd = epoch_by_chs.compute_psd(fmin=fmin, fmax=fmax, n_jobs=-1)
            psd = np.mean(psd, axis=(0, 1, 2))
            psd_arr[i, j] = psd
    return psd_arr
    
def get_Cuban_spectrum(fmin_arr=fmin_arr, fmax_arr=fmax_arr, picked_chs=['eeg']):
    """
    Computes the power spectral density using the data of the Cuban sample in 
    frequency bands defined by fmin_arr and fmax_arr. 

    Parameters
    ----------
    fmin_arr : Nx1 array, optional
        An array of lower bounds of frequency bands. The default is fmin_arr.
    fmax_arr : Nx1 array, optional
        An array of upper bounds of frequency bands. The default is fmax_arr.
    picked_chs : array of strings, optional
        Channels to compute PSD onto. Can be names or types of channels. If set
        types, all channels of the chosen type are picked. The default is ['eeg'].

    Returns
    -------
    MxN array
        An array of PSD values for M participants in N frequency bands. The result
        is averaged over epochs, channels and frequency points. 

    """
    data_dir = 'E:\Work\MBS\EEG\Modelling\Intelligence\Cuban Map Project\Preprocessed data'
    files = glob.glob(data_dir + '\\CBM*.fif')
    return get_spectrum(files, fmin_arr, fmax_arr, picked_chs)

def Raven_to_IQ(raven_arr, raven_IQ_map):
    raven_values = np.array([0 if el == 'NA' else int(el) for el in raven_arr])
    raven_sum = np.sum(raven_values)
    if raven_sum < 15 or raven_sum > 60: return np.nan
    return raven_IQ_map[raven_sum]

def get_Raven_intel():
    """
    Gets Raven intelligence score from the Ivanov's (or Zakharov's, God knows)
    file df_ML_full.csv. Assumes there exist Raven data\\RavenIQ.csv and
    Raven data\\df_ML_full.csv directories, the first one contains a rule on
    how to map Raven's score onto IQ values, the second one contains Raven's
    scores themselves. Takes only long Raven results. Strictly dataset- and
    project-specific, so don't expect any generalization - the only reason why
    I extracted this was that I was fed up with copy-pasting the same lines of
    code from script to script. 

    Returns
    -------
    raven_IQ_dict : str->int dict
        Returns a dictionary that maps participant IDs onto their IQ scores. 
        Can contain NaN if raw Raven scores sum up to unbelievably low (< 15) 
        or impossibly high value (> 60). 

    """
    # Form a rule on how to map Raven scores to IQ.
    
    raven_IQ = read_csv('Raven data\\RavenIQ.csv')
    raven_IQ_map = {row[0].astype(int): row[1].astype(int) for row in raven_IQ.T}
    
    # The first row is a header, so is omitted. The columns 0 through 61 are IDs
    # and raw Raven results. The short Raven (without first sections) is excluded. 
    
    raw_behav_raven = read_csv('Raven data\\df_ML_full.csv')[1:,0:61]
    raw_behav_raven = raw_behav_raven[raw_behav_raven[:, 1] != 'NA']
    
    # Sum up the raw Raven scores and build the resulting dict.
    
    raven_IQ_dict = {row[0]: Raven_to_IQ(row[1:], raven_IQ_map) for row in raw_behav_raven}
    return raven_IQ_dict

def get_WAIS_intel(return_PIQ=True):
    """
    Gets WAIS intelligence score from Cuban Brain Map Project, namely, 
    WAIS_III.csv file, assuming the file exists in the same directory the 
    intel_utils.py (this script file) does. Strictly dataset- and
    project-specific, so don't expect any generalization - the only reason why
    I extracted this was that I was fed up with copy-pasting the same lines of
    code from script to script. 

    Parameters
    ----------
    return_PIQ : bool, optional
        If True, returns PIQ (pefrofmance IQ) values, FSIQ (full-scale IQ)
        otherwise. The default is True.

    Returns
    -------
    PIQ_dict : str->int dict
        Returns a dictionary that maps participant IDs onto their IQ scores.

    """
    IQ_index = 3 if return_PIQ else 1
    behav_PIQ = read_csv('WAIS_III.csv')
    
    # Two first rows are title and header, so are omitted. 0th column is 
    # participant ID, 3rd one is PIQ, 1st one is FSIQ.
    
    PIQ_with_ID = behav_PIQ[2:, [0, IQ_index]]
    
    # For some participants there's no any intelligence data, so remove them.
    
    PIQ_with_ID_space_removed = PIQ_with_ID[PIQ_with_ID[:, 1] != '']
    PIQ_dict = {row[0]: row[1].astype(int) for row in PIQ_with_ID_space_removed}
    return PIQ_dict
 
def get_WAIS_age():
    data = read_csv('Demographic_data.csv')
    
    # Two first rows are title and header, so are omitted. 0th column is 
    # participant ID, 3rd one is PIQ, 1st one is FSIQ.
    
    age = data[2:, [0, 2]]
    
    # For some participants there's no any intelligence data, so remove them.

    age_dict = {row[0]: row[1].astype(int) for row in age}
    return age_dict
    
def plot_scatter_against_function(x_data, y_data, functions, axis_labels=None,
                                  legend=None, k_lim=0.1, scatter_colours=None,
                                  curve_colours=None, title=None):
    """
    Plots a scatter plot defined by x_data and y_data and a regular plot defined by
    functions. Typical use-case is a plotting least-squared fit against data it is
    fitted on. Allows to place several datasets and functions on the same plot,
    assuming shape consistency is satisfied.

    Parameters
    ----------
    x_data : MxN array
        A set of M datasets containing x-data and consisting of N points.
    y_data : MxN array
        A set of M datasets containing y-data and consisting of N points.
    functions : Px1 array of functions
        A set of P functions of type R->R. Must be broadcastable.
    axis_labels : 2x1 array of strings, optional
        A set of 2 string representing the axis labels. If None, no label is attached
        to the plot. The default is None.
    legend : (M+P)x1 array of strings, optional
        A set of M+P strings representing the plot legend. If None, no legend
        is attached to the plot. The default is None.
    k_lim : float, optional
        A spacing coefficient. If 0, the left and right border of the plot is
        built by the leftmost and rightmost points of x_data. The default is 0.1.
    scatter_colours : Mx1 array of strings, optional
        The colours of dots built by x_data and y_data. If None, the colours are chosen
        by default. The default is None.
    curve_colours : Px1 array of strings, optional
        The colours of curves built by functions. If None, the colours are chosen
        by default. The default is None.
    title : str, optional
        The tile of the plot.
    Returns
    -------
    None.

    """
    x_arr = np.array([])
    y_arr = np.array([])
    # Create flattened array out of all data arrays in order to calculate the limits
    for data in zip(x_data, y_data):
        x_arr = np.append(x_arr, data[0])
        y_arr = np.append(y_arr, data[1])

    x_min, x_max = np.min(x_arr), np.max(x_arr)
    y_min, y_max = np.min(y_arr), np.max(y_arr)

    x_lim_range = x_max - x_min
    y_lim_range = y_max - y_min
    xlim = [x_min - k_lim * x_lim_range,
            x_max + k_lim * x_lim_range] if x_lim_range > 0 else [-1, 1]
    ylim = [y_min - k_lim * y_lim_range,
            y_max + k_lim * y_lim_range] if y_lim_range > 0 else [-1, 1]
    plt.xlim(xlim)
    plt.ylim(ylim)

    for i, (x,y) in enumerate(zip(x_data, y_data)):
        if scatter_colours != None:
            plt.scatter(x, y, c = scatter_colours[i])
        else:
            plt.scatter(x, y)

    N = 100
    x = np.linspace(x_min, x_max, N)
    for i, f in enumerate(functions):
        if curve_colours != None:
            plt.plot(x, f(x), c = curve_colours[i])
        else:
            plt.plot(x, f(x))

    if axis_labels != None:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
    if legend != None: plt.legend(legend)
    if title != None: plt.title(title)
    plt.grid()
    plt.show()
