#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created april 2021
    by Juliette MILLET
    add delta_values to human_results
"""
import os

def add_values(file_human, files_with_delta, folder_results,  file_out):
    results = {}
    for file in files_with_delta:
        name = file.split('.')[0]
        name = name.split('triplet')[0]
        f_res = open(os.path.join(folder_results, file), 'r')
        ind = f_res.readline().replace('\n', '').split('\t')
        for line in f_res:
            new_line = line.replace('\n', '').split('\t')
            trip = new_line[ind.index('filename')]
            print(trip)
            if trip not in results:
                results[trip] = {}
            results[trip][name] = new_line[ind.index('delta')]
        f_res.close()

    f_human = open(file_human, 'r')
    f_out = open(file_out, 'w')

    ind = f_human.readline().replace('\n', '')
    f_out.write(ind )
    for file in files_with_delta:
        name = file.split('.')[0]
        name = name.split('triplet')[0]
        f_out.write(',' + name)
    f_out.write('\n')

    ind = ind.split(',')

    for line in f_human:
        new_line = line.replace('\n', '').split(',')
        trip = new_line[ind.index('filename')]
        f_out.write(','.join(new_line))
        for file in files_with_delta:
            name = file.split('.')[0]
            name = name.split('triplet')[0]
            f_out.write(',' + results[trip][name])
        f_out.write('\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to add distances')
    parser.add_argument('folder_results', metavar='f_do', type=str,
                        help='folder where the delta values files are')
    parser.add_argument('file_in', metavar='f_do', type=str,
                        help='predictor file to modify')
    parser.add_argument('file_out', metavar='f_do', type=str,
                        help='predictor file out')
    args = parser.parse_args()
    files_values = [file for file in os.listdir(args.folder_results)]
    add_values(file_human=args.file_in, files_with_delta=files_values, file_out=args.file_out, folder_results=args.folder_results)