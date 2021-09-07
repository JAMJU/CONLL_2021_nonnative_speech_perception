#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute distance from representation
"""
import os
from dtw_experiment import compute_dtw
from get_representation import get_triphone_acoustic_feat, get_triphone_dpgmm, \
    get_triphone_wav2vec

def get_distances_and_delta(triphone_TGT, triphone_OTH, triphone_X, get_func, distance):
    TGT = get_func(triphone_TGT)
    OTH = get_func(triphone_OTH)
    X = get_func(triphone_X)

    TGTX = compute_dtw(TGT,X, distance, norm_div=True)
    OTHX = compute_dtw(OTH,X, distance, norm_div=True)

    delta = OTHX - TGTX

    return TGTX, OTHX, delta


def get_distance_for_triplets(filename_triplet_list, file_out, get_func, distance):
    f = open(filename_triplet_list, 'r')
    ind = f.readline().replace('\n', '').split('\t')
    print(ind)
    f_out = open(file_out, 'w')
    f_out.write('\t'.join(ind + ['TGTX', 'OTHX', 'delta', 'decision\n']))
    kee_dis = {}
    count = 0
    for line in f:
        if count % 100 == 0:
            print(count)
        count += 1
        new_line = line.replace('\n', '').split('\t')
        OTH_item = new_line[ind.index('OTH_item')]
        TGT_item = new_line[ind.index('TGT_item')]
        X_item = new_line[ind.index('X_item')]
        if TGT_item + ',' + OTH_item + ',' + X_item in kee_dis.keys():
            key = TGT_item + ',' + OTH_item + ',' + X_item
            TGTX = kee_dis[key][0]
            OTHX = kee_dis[key][1]
            delta = kee_dis[key][2]
        else:
            TGTX, OTHX, delta = get_distances_and_delta(triphone_TGT=TGT_item, triphone_OTH=OTH_item, triphone_X=X_item, get_func=get_func, distance = distance)
            kee_dis[TGT_item + ',' + OTH_item + ',' + X_item] = [TGTX, OTHX, delta]
        f_out.write('\t'.join(new_line + [str(TGTX), str(OTHX), str(delta), '1\n' if delta> 0. else '0\n']))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to compute distances')
    parser.add_argument('model', metavar='f_do', type=str,
                        help='model')
    parser.add_argument('layer', metavar='f_do', type=str,
                        help='layer wanted, only for neural network model')
    parser.add_argument('distance', metavar='f_do', type=str,
                        help='distance wanted, cosine for all model except for dpgmm where we use kl')
    parser.add_argument('folder_out', metavar='f_do', type=str,
                        help='folder where to put the distance file')
    args = parser.parse_args()

    triplet_file = 'all_triplets_list.csv'
    triphone_file = 'all_cleaned.csv'

    wav2vec_10k_path = os.path.join('ADDYOURPATH', args.layer)
    wav2vec_10k_en_path = os.path.join('ADDYOURPATH', args.layer)
    wav2vec_10k_fr_path = os.path.join('ADDYOURPATH', args.layer)
    mfccs_path = 'ADDYOURPATH'
    dpgmm_french_path = 'ADDYOURPATH'
    dpgmm_english_path = 'ADDYOURPATH'

    mfccs = lambda x: get_triphone_acoustic_feat(folder_data=mfccs_path, correspondance_file=triphone_file, window=0.025, stride = 0.010, triphone_name=x)
    dpgmm_english = lambda x: get_triphone_dpgmm(folder_data=dpgmm_english_path, correspondance_file=triphone_file,
                                                window=0.025, stride=0.010, triphone_name=x)
    dpgmm_french= lambda x: get_triphone_dpgmm(folder_data=dpgmm_french_path, correspondance_file=triphone_file,
                                                window=0.025, stride=0.010, triphone_name=x)
    wav2vec_10k = lambda x: get_triphone_wav2vec(folder_data=wav2vec_10k_path, triphone_name=x,
                                                    layer_name=args.layer)
    wav2vec_10k_en = lambda x: get_triphone_wav2vec(folder_data=wav2vec_10k_en_path, triphone_name=x,
                                                     layer_name=args.layer)
    wav2vec_10k_fr = lambda x: get_triphone_wav2vec(folder_data=wav2vec_10k_fr_path, triphone_name=x,
                                                    layer_name=args.layer)
    if args.model == 'mfccs':
        func = mfccs
    elif args.model == 'dpgmm_english':
        func = dpgmm_english
    elif args.model == 'dpgmm_french':
        func = dpgmm_french
    elif args.model == 'wav2vec_10k':
        func = wav2vec_10k
    elif args.model == 'wav2vec_10k_en':
        func = wav2vec_10k_en
    elif args.model == 'wav2vec_10k_fr':
        func = wav2vec_10k_fr

    get_distance_for_triplets(filename_triplet_list=triplet_file, file_out=os.path.join(args.folder_out, args.model + '_'+ args.layer + '_' + args.distance + 'triplet_distances.csv'),
                              get_func=func, distance=args.distance)