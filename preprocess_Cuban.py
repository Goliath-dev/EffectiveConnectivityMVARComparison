# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:10:19 2023

@author: Admin
"""

# This file runs preprocessing protocol, the same as in a Cuban sample in 
# the intelligence project, and applies it to the Cuban sample itself. Differs
# from the one in the intelligence project a little bit in order to fit this 
# research better. Also differs from preprocess_Chinese.py because of the 
# differences in the samples (different coordinate frames and sometimes different
# channels count) I could've functionally decompose said scripts to reuse common
# code, but hey.

import source_utils
import mne
import autoreject
import glob
import os
import numpy as np
import time
import intel_utils

start = time.time()

raw_dir = 'E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\Cuban Map Project\\Data\\ds_bids_cbm_loris_24_11_21\\'
clean_epochs_dir = 'Clean Data\\Cuban'
label_dir = 'Label//Cuban'
source_ts_dir = 'Source Data//Cuban'
files = glob.glob(raw_dir + 'sub-CBM*\\eeg\\sub-CBM*_task-protmap_eeg.edf')

subjects = [file.removeprefix(raw_dir + 'sub-').split('\\')[0] for file in files]

behav = intel_utils.read_csv('E:\Work\MBS\EEG\Modelling\Intelligence\Cuban Map Project\WAIS_III.csv')
PIQ_with_ID = behav[2:, [0, 3]]
PIQ_with_ID_full = PIQ_with_ID[PIQ_with_ID[:, 1] != '']


N_files = 60
counter = 0
for i, (file, subject) in enumerate(zip(files, subjects)):
    raw = mne.io.read_raw_edf(file, preload=True)
    
    if os.path.exists(f'{label_dir}\\{subject}_labels_aparc.npy'): 
        counter += 1
        continue
    if subject not in PIQ_with_ID_full[:, 0]: continue
    if raw.times[-1] < 300: continue
    if len(raw.info['chs']) > 64: continue
    if ('121' in raw.info['ch_names'] or 
        '122' in raw.info['ch_names'] or 
        '123' in raw.info['ch_names']):
            raw.set_channel_types({'121': 'misc'})
            raw.set_channel_types({'122': 'misc'})
            # raw.set_channel_types({'123': 'misc'})
            # raw.info['bads'].extend(['121', '122', '123'])
            raw.info['bads'].append('121')
            raw.info['bads'].append('122')
            # raw.info['bads'].append('123')

    mne.rename_channels(raw.info, mapping = lambda name: 
                        name.removesuffix('-REF') if name.endswith('-REF') else name)
    mne.rename_channels(raw.info, mapping = lambda name: 
                        name.replace('Z', 'z'))
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw_rest = raw.crop(tmin=0.0, tmax=300.0)
    raw_rest.filter(1, 45)
    raw_rest.resample(100.)
    epochs = mne.make_fixed_length_epochs(raw_rest, duration=6.0, preload=True, overlap=1.0)
    
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
    epochs.save(clean_epochs_dir + f'\\{subject}_clean_epo.fif', overwrite=True, fmt = 'double')
    
    parcs = ['aparc']
    labels, ts = source_utils.fsaverage_time_courses(epochs, parc = parcs)
    for label_set, ts_set, parc in zip (labels, ts, parcs):
        np.save(f'{source_ts_dir}\\{subject}_source_ts_{parc}', ts_set)
        np.save(f'{label_dir}\\{subject}_labels_{parc}', label_set)
    counter += 1
    if counter > N_files: break