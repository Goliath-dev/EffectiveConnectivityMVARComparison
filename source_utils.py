# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:20:52 2023

@author: Admin
"""

from mne import compute_covariance, setup_source_space, make_forward_solution 
from mne import read_labels_from_annot, extract_label_time_course
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.datasets import fetch_fsaverage
import os.path as op
from collections.abc import Iterable

def fsaverage_time_courses(epochs, method='dSPM', parc='aparc', snr=3, epoch_count=None, verbose=False):
    noise_cov = compute_covariance(epochs, tmax=3., method=['shrunk', 'empirical'], verbose=verbose)
    print(noise_cov)
    fs_dir = fetch_fsaverage(verbose=verbose)
    subjects_dir = op.dirname(fs_dir)
    subject = 'fsaverage'
    trans = 'fsaverage'
    src = setup_source_space(subject=subject, subjects_dir=subjects_dir)
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    fwd = make_forward_solution(epochs.info, trans=trans, src=src, 
                                    eeg=True, bem=bem, mindist=5.0)
    inverse = make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
    lambda2 = 1. / snr ** 2
    stc = None
    if epoch_count == None:
        stc = apply_inverse_epochs(epochs, inverse, lambda2, method=method, verbose=verbose)
    else:
        stc = apply_inverse_epochs(epochs[:epoch_count], inverse, lambda2, method=method, verbose=verbose)
    
    labels = None
    time_courses = None
    if isinstance(parc, Iterable) and not (type(parc) is str):
        labels = []
        time_courses = []
        for p in parc:
            label_set = read_labels_from_annot(subject=subject, parc=p, subjects_dir=subjects_dir)
            labels.append(label_set)
            time_courses.append(extract_label_time_course(stc, label_set[:-1], src))
    else:
        labels = read_labels_from_annot(subject=subject, parc=parc, subjects_dir=subjects_dir)
        time_courses = extract_label_time_course(stc, labels[:1], src)
    return labels, time_courses