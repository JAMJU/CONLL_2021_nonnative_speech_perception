#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    evaluate model on native effect
"""

import pandas as pd
from scipy.stats import pearsonr
import numpy as np

def commpute_differences(filename_, value_comparison_french, value_comparison_english):
    """ Compute diff french minus english"""
    data = pd.read_csv(filename_, sep=',', encoding='utf-8')

    data_fr = data[data['language_indiv'] == 'french'].copy()
    data_en = data[data['language_indiv'] == 'english'].copy()

    # We normalize english and french side for the model so they are comparable
    data_fr[value_comparison_french] = data_fr[value_comparison_french]  / data_fr[value_comparison_french].std()
    data_en[value_comparison_english] = data_en[value_comparison_english] / data_en[value_comparison_english].std()

    # first we average on the triplet files for the participants, we use the non binarized results
    gf = data_fr.groupby(['filename', 'TGT', 'OTH', 'prev_phone', 'next_phone', 'language_stimuli'], as_index=False)
    ans_fr = gf.correct_answer.mean()
    val_fr = gf[value_comparison_french].mean()
    ans_fr[value_comparison_french] = val_fr[value_comparison_french]

    gf = data_en.groupby(['filename', 'TGT', 'OTH', 'prev_phone', 'next_phone', 'language_stimuli'], as_index=False)
    ans_en = gf.correct_answer.mean()
    val_en = gf[value_comparison_english].mean()
    ans_en[value_comparison_english] = val_en[value_comparison_english]

    if level == 'file':
        triplet_list = list(ans_fr['filename'].unique())
        diff_humans = []
        diff_models = []
        triplet_done = []
        for trip in triplet_list:
            if trip in triplet_done:
                continue
            other = trip
            if '_0' in trip:
                other.replace('_0', '_1')
            if '_1' in trip:
                other.replace('_1', '_0')
            triplet_done.append(trip)
            triplet_done.append(other)
            val_fr_human = (ans_fr[ans_fr['filename'] == trip].correct_answer.iloc[0] + ans_fr[ans_fr['filename'] == other].correct_answer.iloc[0])/2.
            val_en_human = (ans_en[ans_en['filename'] == trip].correct_answer.iloc[0] + ans_en[ans_en['filename'] == other].correct_answer.iloc[0])/2.
            # it is already averaged for the model
            val_fr_model = ans_fr[ans_fr['filename'] == trip][value_comparison_french].iloc[0]
            val_en_model = ans_en[ans_en['filename'] == trip][value_comparison_english].iloc[0]
            diff_humans.append(val_fr_human - val_en_human)
            diff_models.append(val_fr_model - val_en_model)
        return np.asarray(diff_models), np.asarray(diff_humans)

    # if it is not at the file level, we continue to average
    gf = ans_fr.groupby(['TGT', 'OTH', 'prev_phone', 'next_phone', 'language_stimuli'], as_index=False)
    ans_fr = gf.correct_answer.mean()
    val_fr = gf[value_comparison_french].mean()
    ans_fr[value_comparison_french] = val_fr[value_comparison_french]

    gf = ans_en.groupby(['TGT', 'OTH', 'prev_phone', 'next_phone', 'language_stimuli'], as_index=False)
    ans_en = gf.correct_answer.mean()
    val_en = gf[value_comparison_english].mean()
    ans_en[value_comparison_english] = val_en[value_comparison_english]

    # we continue at the contrast level
    gf = ans_fr.groupby(['TGT', 'OTH', 'language_stimuli'], as_index=False)
    ans_fr = gf.correct_answer.mean()
    val_fr = gf[value_comparison_french].mean()
    ans_fr[value_comparison_french] = val_fr[value_comparison_french]

    gf = ans_en.groupby(['TGT', 'OTH', 'language_stimuli'], as_index=False)
    ans_en = gf.correct_answer.mean()
    val_en = gf[value_comparison_english].mean()
    ans_en[value_comparison_english] = val_en[value_comparison_english]
    ans_en['contrast'] = ans_en['TGT'] + ',' + ans_en['OTH'] + ',' +  ans_en['language_stimuli']
    ans_fr['contrast'] = ans_fr['TGT'] + ',' + ans_fr['OTH'] + ',' + ans_fr['language_stimuli']
    triplet_list = list(ans_en.contrast.unique())
    diff_humans = []
    diff_models = []
    triplet_done = []
    for trip in triplet_list:
        if trip in triplet_done:
            continue
        # we average on TGT-OTH OTH-TGT
        other = trip.split(',')
        other = ','.join([other[1], other[0], other[2]])
        triplet_done.append(other)
        triplet_done.append(trip)

        val_fr_human = (ans_fr[ans_fr['contrast'] == trip].correct_answer.iloc[0] +
                        ans_fr[ans_fr['contrast'] == other].correct_answer.iloc[0]) / 2.
        val_en_human = (ans_en[ans_en['contrast'] == trip].correct_answer.iloc[0] +
                        ans_en[ans_en['contrast'] == other].correct_answer.iloc[0]) / 2.

        val_fr_model = (ans_fr[ans_fr['contrast'] == trip][value_comparison_french].iloc[0] +
                        ans_fr[ans_fr['contrast'] == other][value_comparison_french].iloc[0]) / 2.
        val_en_model = (ans_en[ans_en['contrast'] == trip][value_comparison_english].iloc[0] +
                        ans_en[ans_en['contrast'] == other][value_comparison_english].iloc[0]) / 2.
        diff_humans.append(val_fr_human - val_en_human)
        diff_models.append(val_fr_model - val_en_model)
    return np.asarray(diff_models), np.asarray(diff_humans)

def compute_correlation(diff_models, diff_humans, inverted):
    #print(diff_models)
    #print(diff_humans)
    if inverted:
        return pearsonr(-diff_models, diff_humans)
    else:
        return pearsonr(diff_models, diff_humans)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to add distances')
    parser.add_argument('predictor_file', metavar='f_do', type=str,
                        help='folder where the delta values files are')
    parser.add_argument('english_value', metavar='f_do', type=str,
                        help='predictor value to use for english')
    parser.add_argument('french_value', metavar='f_do', type=str,
                        help='predictor value to use for french')
    parser.add_argument('negative', metavar='f_do', type=str,
                        help='if predictor values are negatively correlated with humans accuracies')
    args = parser.parse_args()
    data_file = args.predictor_file
    inverted = True if args.negative == 'True' else False
    value_to_en = args.english_value
    value_to_fr = args.french_value

    diff_mod, diff_hum = commpute_differences(filename_=data_file,  value_comparison_french=value_to_fr,value_comparison_english=value_to_en)


    c = compute_correlation(diff_models=diff_mod, diff_humans=diff_hum, inverted=inverted)
    print('The native effect is: ', c, ' for ', value_to_en, 'and', value_to_fr)


