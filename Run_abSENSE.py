#!/usr/bin/env python
'''abSENSE: a method to interpret undetected homologs'''

import argparse
import csv
from datetime import datetime
from math import exp, log
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm



class ABSense:
    '''a method that calculates the probability that a homolog of a given gene
    would fail to be detected by a homology search (using BLAST or a similar
    method) in a given species, even if the homolog were present and evolving normally.'''
    def __init__(self, spiecial_distances, ethresh):
        self.gene_length = ABSense._gen_lookup(default=400)
        self.db_size = ABSense._gen_lookup(default=8000000)
        self.spiecial_distance = spiecial_distances
        self.ethresh = ethresh

    @staticmethod
    def _gen_lookup(lookup_dict=None, default=None):
        if lookup_dict is None:
            return lambda x: default
        if default is None:
            return lambda x: lookup_dict[x]
        return lambda x: lookup_dict.get(x, default)

    @staticmethod
    def predict(x, a, b):
        '''curve to fit'''
        return a * np.exp(-b * x)

    @staticmethod
    def estimate_noise(distance, a, b):
        '''Gaussian noise (a function of distance, a, b)'''
        exp_d = exp(-b * distance)
        return np.sqrt(a * (1 - exp_d) * exp_d)

    @staticmethod
    def find_p_i(testvals, currx, bitthresh):
        '''Function to take each of the sampled a, b values and use them to
        sample directly from the distribution of scores taking into account the
        Gaussian noise (a function of distance, a, b).
        This gives an empirical estimate of the prediction interval'''
        p_i_samples = []
        for a_vals, b_vals in testvals:
            detval = ABSense.predict(currx, a_vals, b_vals)
            estnoise = ABSense.estimate_noise(currx, a_vals, b_vals)

            if estnoise > 0:
                p_i_samples += [detval + np.random.normal(0, estnoise) for i in range(200)]
            else:
                p_i_samples.append(detval)

        mean = np.mean(p_i_samples)
        std = np.std(p_i_samples)

        pval = norm.cdf(bitthresh, mean, std)
        (lowint, highint) = norm.interval(0.99, mean, std)

        return lowint, highint, pval

    def bitscore_threshold(self, gene, species):
        '''Compute detectable bitscore threshold'''
        return -log(self.ethresh / (self.gene_length(gene) * self.db_size(species)), 2)

    def set_gene_length(self, gene_dict=None, default=None):
        '''Add gene length data'''
        self.gene_length = ABSense._gen_lookup(gene_dict, default)

    def set_db_size(self, db_dict=None, default=None):
        '''Add species genome size data'''
        self.db_size = ABSense._gen_lookup(db_dict, default)

    def test_orthology(self, bit_scores, gene=None):
        '''Computes orthology probabilities for a gene'''
        orthologs = [s for s, b in bit_scores.items() if b > 0]

        truncdistances = [self.spiecial_distance[s] for s in orthologs]
        genebitscores = [bit_scores[s] for s in orthologs]

        (a, b), covar = curve_fit(
            ABSense.predict,
            truncdistances,
            genebitscores,
            bounds=((-np.inf, 0), (np.inf, np.inf)))

        preds = {s: ABSense.predict(self.spiecial_distance[s], a, b) for s in bit_scores}

        testvals = [np.random.multivariate_normal([a, b], covar) for i in range(200)]

        results = {s: ABSense.find_p_i(
          testvals,
          self.spiecial_distance[s],
          self.bitscore_threshold(gene,s)) for s in bit_scores}

        return {s: (preds[s], *results[s]) for s in bit_scores}

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
    with open(file_path, 'w', newline="\n") as stream:
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

    my_abs = ABSense(spiecial_distance, args.Eval)

    os.mkdir(args.out)
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
