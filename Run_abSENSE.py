#!/usr/bin/env python
'''abSENSE: a method to interpret undetected homologs'''

import argparse
import csv
from datetime import datetime
import os
import numpy as np
from abSENSE import abSENSE

def tsv_to_dict(file_name, data_type=float):
    '''Converts a 2 column TSV to a dictionary of strings to floats'''
    with open(file_name, "r") as stream:
        reader = csv.reader(stream, delimiter="\t")
        my_dict = {row[0]: data_type(row[1]) for row in reader}
        return my_dict

def load_spiecial_distances(distfile, includeonly=None):
    '''Reads the species distance file and filters for includeonly'''
    distances = {}
    with open(distfile, "r") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row in reader:
            if includeonly is None or row[0] in includeonly:
                distances[row[0]] = as_float(row[1])
    return distances

def as_float(value):
    '''Casts value as float, else False'''
    try:
        return float(value)
    except:
        return False

def bit_score_reader(scorefile, species):
    '''CSV DictReader for bitscores'''
    with open(scorefile, 'r') as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            yield row[reader.fieldnames[0]], {k: as_float(row[k]) for k in species}

def csv_writer(file_path, fieldnames):
    '''CSV DictWriter for output'''
    stream = open(file_path, 'w', newline="\n")
    writer = csv.DictWriter(stream, fieldnames, delimiter="\t")
    writer.writeheader()
    return writer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ABSense arguments:')

    parser.add_argument(
      "--distfile",
      type=str,
      required=True,
      help="Required. Name of file containing pairwise evolutionary distances between " +
           "focal species and each of the other species")
    parser.add_argument(
      "--scorefile",
      type=str,
      required=True,
      help="Required. Name of file containing bitscores between focal species gene and " +
           "orthologs in other species")
    parser.add_argument(
      "--Eval",
      default=0.001,
      type=float,
      help="Optional. E-value threshold. Scientific notation (e.g. 10E-5) accepted. " +
           "Default 0.001.")
    parser.add_argument("--includeonly",
      type=str,
      help="Optional. Species whose orthologs' bitscores will be included in fit; " +
           "all others will be omitted. Default is all species. Format as species names, " +
           "exactly as in input files, separated by commas (no spaces).")
    parser.add_argument(
      "--genelenfile",
      type=str,
      help="Optional. File containing lengths (aa) of all genes to be analyzed. Used to " +
           "accurately calculate E-value threshold. Default is 400aa for all genes. Only " +
           "large deviations will qualitatively affect results.")
    parser.add_argument("--dblenfile",
      type=str,
      help="Optional. File containing size (aa) of databases on which the anticipated " +
          "homology searches will be performed. Species-specific. Used to accurately " +
          "calculate E-value threshold. Default is 400aa/gene * 20,000 genes for each " +
          "species, intended to be the size of an average proteome. Only large deviations " +
          "will significantly affect results.")
    parser.add_argument(
      "--predall",
      type=bool,
      default=False,
      help="Optional. True: Predicts bitscores and P(detectable) of homologs in all " +
           "species, including those in which homologs were actually detected. Default " +
           "is False: only make predictions for homologs that seem to be absent.")
    starttime = datetime.now().strftime("%m.%d.%Y_%H.%M")
    parser.add_argument(
      "--out",
      type=str,
      default=f'ABSense_results_{starttime}',
      help="Optional. Name of directory for output data. Default is date and time when " +
           "analysis was run.")

    args = parser.parse_args()
    np.random.seed(1)
    # For testing
    np.random.multivariate_normal = lambda n, *x: [0.5] * len(n)
    np.random.normal = lambda *x: 0.5

    spiecial_distance = load_spiecial_distances(args.distfile, args.includeonly)

    if args.includeonly:
        include = args.includeonly.split(",")
    else:
        include = list(spiecial_distance.keys())

    my_abs = abSENSE(spiecial_distance, args.Eval)

    try:
        os.mkdir(args.out)
    except FileExistsError:
        pass

    spec_header = ["Gene"] + include
    #out_path = lambda p: os.path.join(args.out, p)
    def out_path(file_name):
        '''Helper function for file paths'''
        return os.path.join(args.out, file_name)
    mloutputfile = csv_writer(out_path('Predicted_bitscores'), spec_header)
    lowboundoutputfile = csv_writer(out_path('Bitscore_99PI_lowerbound_predictions'), spec_header)
    highboundoutputfile = csv_writer(out_path('Bitscore_99PI_higherbound_predictions'), spec_header)
    pvaloutputfile = csv_writer(out_path('Detection_failure_probabilities'), spec_header)
    #outputfileparams = csv_writer(out_path('Parameter_values'), ["Gene", "a", "b"])
    spec_files = [mloutputfile, lowboundoutputfile, highboundoutputfile, pvaloutputfile]

    for gene, bit_scores in bit_score_reader(args.scorefile, include):
        results = my_abs.test_orthology(bit_scores)
        for i in range(4):
            row = {k: round(v[i], 2) for k, v in results.items()}
            row["Gene"] = gene
            spec_files[i].writerow(row)
