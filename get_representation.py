#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to get representation of triplet
"""

import pandas as pd
import numpy as np
import os

def get_triphone_acoustic_feat(folder_data, correspondance_file, window, stride, triphone_name):
    """
    window and stride are in seconds
    :param folder_data:
    :param correspondance_file:
    :param window:
    :param stride:
    :param triphone_name:
    :return:
    """
    df = pd.read_csv(correspondance_file, delimiter = '\t')
    info = df[df['index'] == triphone_name]
    file = info['#file'].iloc[0]
    file = file.split('/')[-1]
    #print(file)
    #print(folder_data)
    onset = info['onset'].iloc[0]
    offset = info['offset'].iloc[0]
    if onset == 'NA' or 'GLO' in triphone_name: # cas of OLLO database
        return np.load(os.path.join(folder_data, file + '.npy'))
    #print(onset)
    onset = float(onset)
    offset = float(offset)
    data = np.load(os.path.join(folder_data, file + '.npy'))
    # we suppose the time stamps is in the middle of the window
    begin  = window/2.

    onset_tab = int((onset - begin)/stride)
    offset_tab = int(int((offset - begin)/stride) + 1)

    return data[onset_tab:offset_tab + 1,:]

def get_triphone_dpgmm(folder_data, correspondance_file, window, stride, triphone_name):
    """
    window and stride are in seconds
    :param folder_data:
    :param correspondance_file:
    :param window:
    :param stride:
    :param triphone_name:
    :return:
    """
    df = pd.read_csv(correspondance_file, delimiter = '\t')
    info = df[df['index'] == triphone_name]
    file = info['#file'].iloc[0]
    file = file.split('/')[-1]
    #print(file)
    #print(folder_data)
    onset = info['onset'].iloc[0]
    offset = info['offset'].iloc[0]
    if onset == 'NA' or 'GLO' in triphone_name: # cas of OLLO database
        return np.loadtxt(os.path.join(folder_data, file + '.wav' + '_' + file + '.csv'), delimiter = ",")
    #print(onset)
    onset = float(onset)
    offset = float(offset)
    data = np.loadtxt(os.path.join(folder_data,  file + '.wav' + '_' + file + '.csv'), delimiter = ",")
    # we suppose the time stamps is in the middle of the window
    begin  = window/2.

    onset_tab = int((onset - begin)/stride)
    offset_tab = int(int((offset - begin)/stride) + 1)

    return data[onset_tab:offset_tab + 1,:]


def get_triphone_wav2vec(folder_data, triphone_name, layer_name):
    """ We need to return the representation with timexdim"""
    if 'conv' in layer_name:
        data = np.load(os.path.join(folder_data, triphone_name + '.npy'))
        return data.swapaxes(0,1) # we put time as first dimension
    else:
        data = np.load(os.path.join(folder_data, triphone_name + '.npy'))
        sh = data.shape
        data = data.reshape((sh[0], sh[-1]))
        return data