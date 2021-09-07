#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to test if gamma value predicts discrimination well
"""

import pandas as pd
from statsmodels.formula.api import probit
from sampling import get_dico_corres_file


def model_probit_binarized(data_file,  model, lang): # for the model, you have to add the +
    data_ = pd.read_csv(data_file, sep=',', encoding='utf-8')

    data_['binarized_answer'] = (data_['binarized_answer'] + 1.) / 2  # we transform -1 1 into 0 1
    data_['english'] = 1 - data_['language_indiv_code']
    data_['french'] = data_['language_indiv_code']

    nb_lines = len(data_)
    all_lines = list(range(nb_lines))

    # we select the language
    if lang == 'english':
        data = data_[data_['english'] == 1]#.iloc[lines_sampled] # we take all data for the simple model
    else:
        data = data_[data_['french'] == 1]

    # we normalize data
    for val in ['nb_stimuli'] + [mod.replace(' ', '')  for mod in model.split('+')]:
        data[val] = (data[val] -data[val].mean())/data[val].std()
        if 'ov' in val:
            data[val] = -data[val]
    model_probit = probit("binarized_answer ~ TGT_first_code + C(individual) + nb_stimuli  + " + model, data) #
    result_probit = model_probit.fit()

    return model_probit.loglike(result_probit.params)

def func_to_parallel(args):
    dico_lines = args[0]
    list_names= args[1]
    it = args[2]
    file_humans = args[3]
    list_log = [str(it)]
    language = args[4]
    for mod in list_names:
        print(mod)
        log= model_probit_binarized(data_file=file_humans, model=mod, lang =language )
        list_log.append(str(log))
        #list_auc.append(str(auc_result))
    return list_log


def iteration_model(filename, nb_it, outfile, french = True, english = True):
    dico_lines = get_dico_corres_file(filename, french=french, english=english)

    f_names = open(filename, 'r')
    line_names = f_names.readline().replace('\n', '').split(',')
    list_names = []
    start = False
    for nam in line_names:
        if start:
            list_names.append(nam)
        elif not start and nam == "language_stimuli_code":  # end of info start of models
            start = True

    f_names.close()
    print(list_names)
    out = open(outfile, 'a')
    out.write('nb,' + ','.join(list_names + ['mfccs_and_overlap_score',
                                            'dpgmm_en_and_overlap_score',
                                             'dpgmm_fr_and_overlap_score',
                                             'wav2vec_en_and_overlap_score',
                                             'wav2vec_fr_and_overlap_score',
                                             'wav2vec_10k_and_overlap_score',
                                             ]))
    out.write('\n')

    out.close()

    list_names += ['mfccs_cosine_delta + overlap_score_naive',
                   'overlap_score_naive + dpgmm_english',
                   'overlap_score_naive + dpgmm_french',
                   'overlap_score_naive + wav2vec_10k_en_transf4',
                   'overlap_score_naive + wav2vec_10k_fr_transf4',
                   'overlap_score_naive + wav2vec_10k_transf4',
                   ]

    print('Beginning')
    arguments = [[dico_lines, list_names, 0, filename, 'french' if french else 'english']]
    line = func_to_parallel(arguments)
    lines = [line]
    for li in lines:
        out = open(outfile , 'a')
        out.write(','.join(li))
        out.write('\n')
        out.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to analyze output from humans vs model\'s outputs using a probit model')
    parser.add_argument('file_humans', metavar='f_do', type=str,
                        help='file with outputs humans t give')
    parser.add_argument('outfile', metavar='f_do', type=str,
                        help='file with log likelihood answers')
    parser.add_argument('french', metavar='f_do', type=str,
                        help='if french participants used')
    parser.add_argument('english', metavar='f_do', type=str,
                        help='if english participants used')

    args = parser.parse_args()

    fr = True if args.french == 'True' else False
    en = True if args.english == 'True' else False
    print('french', fr,'english', en)

    iteration_model(filename=args.file_humans, nb_it=1, outfile=args.outfile, french=fr, english=en)
