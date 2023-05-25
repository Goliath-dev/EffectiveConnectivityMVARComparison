# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:33:52 2023

@author: Admin
"""

# This file contains the SCoT library experiment. 

import numpy as np
import glob
import os
import time
import scot
import matplotlib.pyplot as plt
from itertools import combinations, product
from scipy import stats 

Chinese_source_ts_dir = 'Source Data'
Cuban_source_ts_dir = 'Source Data\\Cuban\\'

Chinese_corr_matrices_dir = 'CorrMatrices\\China'
Cuban_corr_matrices_dir = 'CorrMatrices\\Cuba'
both_corr_matrices_dir = 'CorrMatrices\\Both'

def freq_range_to_idx(freq_range, fs = 100, nfft = 512):
    freq_points = np.linspace(0, fs/2, nfft)
    min_idx = np.argmin(np.abs(freq_points - freq_range[0]))
    max_idx = np.argmin(np.abs(freq_points - freq_range[1]))
    return (min_idx, max_idx)

def calculate_conn_matrices(source_files, freq_ranges):
    matrices = {'DTF':[], 'ffDTF': [], 
               'dDTF': [], 'GDTF': [],
               'PDC': [], 'ffPDC': [],
               'PDCF': [], 'GPDC': []}
    for file in source_files:
        # subject = file.split('\\')[1].split('.')[0].removesuffix('_source_ts_aparc')
        source_ts = np.load(file)
        
        print(source_ts.shape)
        
        var_model = scot.var.VAR(40)
        var_model.fit(source_ts)
        conn_class = scot.connectivity.Connectivity(var_model.coef, var_model.rescov)
        methods = {'DTF':conn_class.DTF, 'ffDTF': conn_class.ffDTF, 
                   'dDTF': conn_class.dDTF, 'GDTF': conn_class.GDTF,
                   'PDC': conn_class.PDC, 'ffPDC': conn_class.ffPDC,
                   'PDCF': conn_class.PDCF, 'GPDC': conn_class.GPDC}
        for method in methods:
            conn = methods[method]()
            for freq_range in freq_ranges:
                min_idx, max_idx = freq_range_to_idx(freq_range)
                conn_matrix = np.mean(conn[:,:,min_idx:max_idx], axis=2)
                np.fill_diagonal(conn_matrix, 0)
                matrices[method].append((freq_range, conn_matrix))
    return matrices

def calculate_corr_matrices(glob_source_dir, N_files=60):
    freq_ranges = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
    source_files = glob.glob(glob_source_dir)[:N_files]
    matrices = calculate_conn_matrices(source_files, freq_ranges)
    # Just a size of a conncetivity matrix, 68 for the D-K atlas. 'DTF' is chosen 
    # arbitrarily, they are all of the same size.
    N = matrices['DTF'][0][1].shape[0] 
    corr_matrices = {(freq_range, method_pair): np.zeros((N, N)) 
                     for (freq_range, method_pair) in 
                     product(freq_ranges, combinations(matrices, 2))}
    for (freq_range, method_pair) in corr_matrices:
        series1 = np.array([matrix[1] for matrix in matrices[method_pair[0]] 
                   if matrix[0] == freq_range])
        series2 = np.array([matrix[1] for matrix in matrices[method_pair[1]] 
                   if matrix[0] == freq_range])
         
        for i in range(series1.shape[1]):
            for j in range(series1.shape[2]):
                corr_matrices[(freq_range, method_pair)][i, j] = stats.pearsonr(
                    series1[:, i, j], series2[:, i, j])[0]
    return corr_matrices

def plot_and_save(corr_matrices_dir, corr_matrices, is_both = False):
    for matrix in corr_matrices:
        plt.imshow(corr_matrices[matrix])
        plt.title(f'{matrix[0]} Hz, correlations between {matrix[1]}')
        plt.colorbar()
        plt.show()
        if not is_both:
            name = f'//{matrix[0][0]}_{matrix[0][1]}_{matrix[1][0]}_{matrix[1][1]}'
        else:
            name = f'//{matrix[0][0]}_{matrix[0][1]}_{matrix[1]}'
        np.save(corr_matrices_dir + name, corr_matrices[matrix])
    
start = time.time()

# Chinese sample.
# corr_matrices = calculate_corr_matrices(f'{Chinese_source_ts_dir}\\sub-*_source_ts_aparc.npy', N_files=60)
# plot_and_save(Chinese_corr_matrices_dir, corr_matrices)

# Cuban sample.
# corr_matrices = calculate_corr_matrices(f'{Cuban_source_ts_dir}\\CBM*_source_ts_aparc.npy', N_files=60)
# plot_and_save(Cuban_corr_matrices_dir, corr_matrices)

# Both samples.
freq_ranges = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
cuban_conn = calculate_conn_matrices(glob.glob(f'{Cuban_source_ts_dir}\\CBM*_source_ts_aparc.npy')[:60], freq_ranges)
chinese_conn = calculate_conn_matrices(glob.glob(f'{Chinese_source_ts_dir}\\sub-*_source_ts_aparc.npy')[:60], freq_ranges)
N = chinese_conn['DTF'][0][1].shape[0] 
corr_matrices = {(freq_range, method_pair): np.zeros((N, N))
                  for freq_range, method_pair in 
                  product(freq_ranges, product(chinese_conn.keys(), repeat=2))}
for freq_range, method_pair in corr_matrices:
    series1 = np.array([matrix[1] for matrix in cuban_conn[method_pair[0]] 
                if matrix[0] == freq_range])
    series2 = np.array([matrix[1] for matrix in chinese_conn[method_pair[1]] 
                if matrix[0] == freq_range])
    
    for i in range(series1.shape[1]):
        for j in range(series1.shape[2]):
            corr_matrices[(freq_range, method_pair)][i, j] = stats.pearsonr(
                series1[:, i, j], series2[:, i, j])[0]


plot_and_save(both_corr_matrices_dir, corr_matrices)    
stop = time.time()
print(stop - start)