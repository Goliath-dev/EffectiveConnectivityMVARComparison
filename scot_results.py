# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:42:13 2023

@author: Admin
"""

# This script get the results of scot_general and process them further.

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import glob

Chinese_corr_matrices_dir = 'CorrMatrices\\China'
Cuban_corr_matrices_dir = 'CorrMatrices\\Cuba'
both_corr_matrices_dir = 'CorrMatrices\\Both'

freq_ranges = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
letters = ['A', 'B', 'C', 'D', 'E', 'F']
meth_idx_map = {'DTF': 0, 'dDTF': 1, 'ffDTF': 2, 'GDTF': 3, 
                'PDC': 4, 'PDCF': 5, 'ffPDC': 6, 'GPDC': 7}

# Plotting the results regarding the Chinese and Cuban datasets separately.
def plot_single_dataset(dataset_dir):
    fig, axs = plt.subplots(3, 2, figsize = (20, 20), constrained_layout = True)
    axs = axs.flatten()
    median_freq_matrices = np.zeros((len(freq_ranges), 8, 8))
    for i, freq_range in enumerate(freq_ranges):
        files = glob.glob(dataset_dir + \
                                  f"\\{freq_range[0]}_{freq_range[1]}_*.npy")
              
        median_matrix = np.zeros((8, 8))
        for file in files:
            freq_and_meth = file.split('\\')[-1].split('.')[0].split('_')
            matrix = np.load(file)
            median = np.median(matrix[~np.isnan(matrix)])
            
            median_matrix[meth_idx_map[freq_and_meth[2]], 
                          meth_idx_map[freq_and_meth[3]]] = median
            median_matrix[meth_idx_map[freq_and_meth[3]], 
                          meth_idx_map[freq_and_meth[2]]] = median
        
        median_freq_matrices[i] = median_matrix
        labels =  meth_idx_map.keys()   
        print(np.median(median_matrix))
        
        axs[i].imshow(median_matrix)
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        for j in range(len(labels)):
            for k in range(len(labels)):
                if j != k:
                    axs[i].text(k, j, round(median_matrix[j, k], 2),
                                   ha="center", va="center", color="k", fontsize = 15)
        axs[i].set_xticks(np.arange(len(labels)), labels=labels, fontsize = 20)
        axs[i].set_yticks(np.arange(len(labels)), labels=labels, fontsize = 20)
        axs[i].text(-0.2, 0.9, letters[i], transform = axs[i].transAxes, fontsize = 30)
        axs[i].set_title(f'{freq_ranges[i][0]} - {freq_ranges[i][1]} Hz', fontsize = 25)

    median_freq_matrices_flat = np.array([np.triu(matrix).flatten() for matrix in median_freq_matrices])
    print(np.sum(median_freq_matrices_flat, axis=1))
    freq_distance_matrix = sp.spatial.distance_matrix(median_freq_matrices_flat, 
                                                      median_freq_matrices_flat)
    axs[5].imshow(freq_distance_matrix)
    distance_labels = [f'{freq_range[0]} - {freq_range[1]}' for freq_range in freq_ranges]
    for j in range(len(distance_labels)):
        for k in range(len(distance_labels)):
            if j != k:
                axs[5].text(k, j, round(freq_distance_matrix[j, k], 2),
                               ha="center", va="center", color="k", fontsize = 15)
    axs[5].set_xticks(np.arange(len(distance_labels)), labels=distance_labels, fontsize = 20)
    axs[5].set_yticks(np.arange(len(distance_labels)), labels=distance_labels, fontsize = 20)
    axs[5].text(-0.2, 0.9, letters[5], transform = axs[5].transAxes, fontsize = 30)
    axs[5].set_title('Distance matrix', fontsize = 25)


# Plotting the results for inter-dataset comparison.
def plot_inter_dataset():
    fig, axs = plt.subplots(3, 2, figsize = (20, 20), constrained_layout = True)
    axs = axs.flatten()
    median_freq_matrices = np.zeros((len(freq_ranges), 8, 8))
    for i, freq_range in enumerate(freq_ranges):
        files = glob.glob(both_corr_matrices_dir + \
                                  f"\\{freq_range[0]}_{freq_range[1]}_*.npy")
        medians = np.zeros((8, 8))
        for file in files:
            freq_and_meth = file.split('\\')[-1].split('.')[0].split('_')
            matrix = np.load(file)
            median = np.median(matrix[~np.isnan(matrix)])
            
            medians[meth_idx_map[freq_and_meth[2]], 
                          meth_idx_map[freq_and_meth[3]]] = median
        
        median_freq_matrices[i] = medians 
        axs[i].imshow(medians)
        print(np.median(medians))
        
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        labels =  meth_idx_map.keys()  
        for j in range(len(labels)):
            for k in range(len(labels)):
                axs[i].text(k, j, round(medians[j, k], 2),
                                ha="center", va="center", color="k", fontsize = 15)
        
        axs[i].set_xticks(np.arange(len(labels)), labels=labels, fontsize = 20)
        axs[i].set_yticks(np.arange(len(labels)), labels=labels, fontsize = 20)
        axs[i].text(-0.2, 0.9, letters[i], transform = axs[i].transAxes, fontsize = 30)
        axs[i].set_title(f'{freq_ranges[i][0]} - {freq_ranges[i][1]} Hz', fontsize = 25)
    
    # print(median_freq_matrices[-1] - median_freq_matrices[0])
    # median_freq_matrices[:, 1:3, :] = 0
    # median_freq_matrices[:, :, 1:3] = 0
    # median_freq_matrices[:, 7:8, :] = 0
    # median_freq_matrices[:, :, 7:8] = 0
    # print(median_freq_matrices[0])
    median_freq_matrices_flat = np.array([np.triu(matrix).flatten() for matrix in median_freq_matrices])
    freq_distance_matrix = sp.spatial.distance_matrix(median_freq_matrices_flat, 
                                                      median_freq_matrices_flat)
    axs[5].imshow(freq_distance_matrix)
    distance_labels = [f'{freq_range[0]} - {freq_range[1]}' for freq_range in freq_ranges]
    for j in range(len(distance_labels)):
        for k in range(len(distance_labels)):
            if j != k:
                axs[5].text(k, j, round(freq_distance_matrix[j, k], 2),
                               ha="center", va="center", color="k", fontsize = 15)
    axs[5].set_xticks(np.arange(len(distance_labels)), labels=distance_labels, fontsize = 20)
    axs[5].set_yticks(np.arange(len(distance_labels)), labels=distance_labels, fontsize = 20)
    axs[5].text(-0.2, 0.9, letters[5], transform = axs[5].transAxes, fontsize = 30)
    axs[5].set_title('Distance matrix', fontsize = 25)

# plot_single_dataset(Chinese_corr_matrices_dir)
plot_inter_dataset()