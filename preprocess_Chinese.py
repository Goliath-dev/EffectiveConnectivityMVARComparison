# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:29:06 2023

@author: Admin
"""

# This file runs preprocessing protocol, the same as in a Cuban sample in 
# the intelligence project, and applies it to the Chinese sample.

import source_utils
import mne
import autoreject
import glob
import os
import numpy as np
import time

start = time.time()

raw_dir = 'Data\\bids'
clean_epochs_dir = 'Clean Data'
label_dir = 'Label'
source_ts_dir = 'Source Data'
files = glob.glob(raw_dir + '\\sub-*\\ses-session1\\eeg\\sub-*_ses-session1_task-eyesclosed_eeg.vhdr')
names = [os.path.basename(file).split('.')[0].split('_')[0] for file in files]


for i, (file, name) in enumerate(zip(files, names)):
    raw = mne.io.read_raw_brainvision(file, preload=True)
    mne.rename_channels(raw.info, mapping = lambda name: 'CPz' if name == 'Cpz' else name)
    mne.rename_channels(raw.info, mapping = lambda name: 'Fpz' if name == 'FPz' else name)
    if os.path.exists(f'{label_dir}\\{name}_labels_aparc.npy'): continue
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw.filter(1, 45)
    raw.resample(100.)
    epochs = mne.make_fixed_length_epochs(raw, duration=6.0, preload=True, overlap=1.0)
    
    reject = autoreject.get_rejection_threshold(epochs, ch_types = ['eeg'], 
                                            random_state=42)
    
    n_components = epochs.info['nchan'] - len(epochs.info['bads'])
    ica = mne.preprocessing.ICA(n_components = n_components, method = 'infomax',
                                random_state=42)
    ica.fit(epochs)
    eog_channel = 'Fp2' # Looks controversial but makes perfect sense - 
    # MNE automatically uses EOG channel as a channel to perform ocular artifact 
    # correction, so there's no need to pass any channel to find_bads_eog in this case.
    # Instead, if there's no such a channel, use Fp2 for this. 
    eog_inds, eog_scores = ica.find_bads_eog(epochs, ch_name = eog_channel, 
                                    measure = 'correlation', threshold = 0.4)
    muscle_inds, muscle_scores = ica.find_bads_muscle(epochs)
    ica.apply(epochs, exclude = muscle_inds)
    
    ar = autoreject.AutoReject(random_state=42)
    clean_epochs, rej_log = ar.fit_transform(epochs, return_log=True)
    
    rsc = autoreject.Ransac(verbose = False, n_resample=100, min_channels=0.2, min_corr=0.9, n_jobs = -1, random_state=42)
    clean_epochs = rsc.fit_transform(clean_epochs)
    
    epochs.set_eeg_reference('average', projection=True)
    epochs.apply_proj()
    epochs.save(clean_epochs_dir + f'\\{name}_clean_epo.fif', overwrite=True, fmt = 'double')
    
    parcs = ['aparc']
    labels, ts = source_utils.fsaverage_time_courses(epochs, parc = parcs)
    for label_set, ts_set, parc in zip (labels, ts, parcs):
        np.save(f'{source_ts_dir}\\{name}_source_ts_{parc}', ts_set)
        np.save(f'{label_dir}\\{name}_labels_{parc}', label_set)

