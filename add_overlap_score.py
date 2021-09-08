#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute overlap score
"""
import numpy as np


# list of phones used during the experiments
phone_english = ['ɪ', 'i', 'ʊ', 'u', 'oʊ', 'eɪ', 'ɛ', 'ʌ', 'æ', 'ɑ']
phone_french = ['i' ,'y' ,'u' ,'e','ɛ', 'ø', 'œ', 'a', 'o', 'ɔ', 'ɛ̃', 'ɔ̃', 'ɑ̃']


def get_v1_vectors(file_results_assimilation):
    """
    Compute the simple version of vector note and vote, just sum along all results for each (lang_stim, vowel, lang_indiv)
    :param file_results_assimilation: list of results for assimilation in tidy
    :return:
    """
    f = open(file_results_assimilation, 'r')
    ind = f.readline().split(',')
    dico_all = {'english':{}, 'french':{}}
    for line in f:
        new_line = line.replace('\n', '').split(',')
        language_indiv = new_line[ind.index('language_indiv')]
        lang_stimuli = new_line[ind.index('language_stimuli')]
        vowel_target = new_line[ind.index('#phone')]
        vowel_chosen = int(new_line[ind.index('code_assim')])
        grade = int(new_line[ind.index('grade')])

        class_stimuli = lang_stimuli + ';' + vowel_target

        if class_stimuli not in dico_all[language_indiv]:
            list_lang_phone = phone_french if language_indiv == 'french' else phone_english
            dico_all[language_indiv][class_stimuli] = {'note_all':[0 for a in list_lang_phone], 'vote_all':[0 for a in list_lang_phone]}

        dico_all[language_indiv][class_stimuli]['note_all'][vowel_chosen] += grade + 1
        dico_all[language_indiv][class_stimuli]['vote_all'][vowel_chosen] += 1

    results = {'english': {}, 'french':{}}
    for lang in ['english', 'french']:
        for class_stim in dico_all[lang]:
            summ = sum(dico_all[lang][class_stim]['vote_all'])
            results[lang][class_stim] = {'note':[n/max(1,v) for n,v in zip(dico_all[lang][class_stim]['note_all'], dico_all[lang][class_stim]['vote_all'])],
                                      'vote': [v/summ for v in dico_all[lang][class_stim]['vote_all']],
                                      'nb_votes':summ}
            results[lang][class_stim]['product'] = [n*v for n,v in zip(results[lang][class_stim]['note'], results[lang][class_stim]['vote'])]

    return results

def get_overlap_score(vector_1,vector_2):
    overlap = 0
    for dim in range(len(vector_1)):
        overlap += min(vector_1[dim], vector_2[dim])
    return overlap


def add_naive(assim_file, file_origin, file_out):
    res = get_v1_vectors(assim_file)

    f_in = open(file_origin, 'r')
    f_out = open(file_out, 'w')
    ind = f_in.readline().replace('\n', '')
    f_out.write(ind + ',' + 'overlap_score_naive' + '\n')
    ind = ind.split(',')
    for line in f_in:
        new_line = line.replace('\n', '').split(',')
        lang_stimuli = new_line[ind.index('language_stimuli')]
        language_indiv = new_line[ind.index('language_indiv')]
        TGT = new_line[ind.index('TGT')]
        OTH = new_line[ind.index('OTH')]

        # We use the vote vectors of the right language
        vector_TGT = np.asarray(res[language_indiv][lang_stimuli + ';' + TGT]['vote'])
        vector_OTH = np.asarray(res[language_indiv][lang_stimuli + ';' + OTH]['vote'])

        over_score = get_overlap_score(vector_OTH, vector_TGT)

        f_out.write(line.replace('\n', '') + ',' + str(over_score) + '\n')
    f_in.close()
    f_out.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to add distances')
    parser.add_argument('discrimination_file', metavar='f_do', type=str,
                        help='file where human discrimination results are')
    parser.add_argument('assimilationf_file', metavar='f_do', type=str,
                        help='file where human assimilation results are')
    parser.add_argument('file_out', metavar='f_do', type=str,
                        help='predictor file out with overlap score included')
    args = parser.parse_args()

    add_naive(assim_file=args.assimilation_file, file_origin=args.discrimination_file, file_out=args.file_out)