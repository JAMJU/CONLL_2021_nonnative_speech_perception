#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute spearman correlation at the phone level and at the context level
"""
from scipy.stats import spearmanr
import pandas as pd





def get_spearman_phone(filename_data, value_evaluated):

    df = pd.read_csv(filename_data, delimiter = ',')

    # We separate participants
    df_french = df[df['language_indiv'] == 'french']
    df_english = df[df['language_indiv'] == 'english']
    all_values = []
    for dff in [df_french, df_english]:
        print(dff['language_indiv'].iloc[0])
        # We get only what we need
        dff = dff[['filename', 'TGT', 'OTH', 'prev_phone', 'next_phone', 'language_stimuli','binarized_answer', value_evaluated]]

        dff['binarized_answer'] = dff['binarized_answer'].astype(float)
        dff[value_evaluated] = dff[value_evaluated].astype(float)

        # We average over triplet first
        gf = dff.groupby(['filename', 'TGT', 'OTH', 'prev_phone', 'next_phone', 'language_stimuli'], as_index = False)
        ans_fr = gf.binarized_answer.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]

        # Then we average over context
        gf = ans_fr.groupby(['TGT', 'OTH', 'prev_phone', 'next_phone', 'language_stimuli'], as_index = False)
        ans_fr = gf.binarized_answer.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]

        # then we average over phone contrast
        gf = ans_fr.groupby(['TGT', 'OTH', 'language_stimuli'], as_index=False)
        ans_fr = gf.binarized_answer.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]

        # the we average over order TGT-OTH or the other way around
        res = ans_fr.copy()
        res['TGT'] = ans_fr['OTH']
        res['OTH'] = ans_fr['TGT']
        # print('ANS_FR###',ans_fr)
        # print('res', res)
        total = pd.concat([ans_fr, res], axis=0, )
        # print('TOTAL#####', total)
        gf = total.groupby(['TGT', 'OTH', 'language_stimuli'], as_index=False)
        ans_fr = gf.binarized_answer.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]

        rho_fr, p_fr = spearmanr(ans_fr['binarized_answer'], ans_fr[value_evaluated])
        #print(value_evaluated, rho_fr, p_fr)
        all_values.append([abs(rho_fr), p_fr])
    return all_values

def func_to_parallelize(args):
    file = args[0]
    list_names = args[1]
    it = args[2]
    data = args[3]
    # we sample
    list_res_french = [str(it)]
    list_res_english = [str(it)]
    for mod in list_names:

        corrs = get_spearman_phone(file,
                                   value_evaluated=mod)
        list_res_french.append(str(corrs[0][0]))
        list_res_english.append(str(corrs[1][0]))
    return list_res_french, list_res_english

def iteration_function(filename, outfile):
    data = pd.read_csv(filename)
    f_names = open(filename, 'r')
    line_names = f_names.readline().replace('\n', '').split(',')
    list_names = []
    start = False
    for nam in line_names:
        if start:
            list_names.append(nam)
        elif not start and nam == "language_stimuli_code": # end of info start of models
            start = True
    f_names.close()
    out_french = open(outfile+ '_french.csv', 'a')
    out_english = open(outfile + '_english.csv', 'a')
    out_french.write('nb,' + ','.join(list_names) + '\n')
    out_english.write('nb,' + ','.join(list_names) + '\n')
    out_french.close()
    out_english.close()
    print('Beginning')
    line= func_to_parallelize([filename, list_names,0, data])
    lines = [line]

    for li in lines:
        french, english = li
        out_french = open(outfile + '_french.csv', 'a')
        out_english = open(outfile + '_english.csv', 'a')
        out_french.write(','.join(french))
        out_english.write(','.join(english))
        out_french.write('\n')
        out_english.write('\n')
        out_french.close()
        out_english.close()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to compute cor')
    parser.add_argument('file', metavar='f_do', type=str,
                        help='model')
    parser.add_argument('outfile', metavar='f_do', type=str,
                        help='beginning out file')
    args = parser.parse_args()

    data_ = args.file

    iteration_function(filename = data_, outfile = args.outfile)














