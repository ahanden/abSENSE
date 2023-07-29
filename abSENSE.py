#!/usr/bin/env python
'''abSENSE: a method to interpret undetected homologs'''

from math import exp, log
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

class abSENSE:
    '''a method that calculates the probability that a homolog of a given gene
    would fail to be detected by a homology search (using BLAST or a similar
    method) in a given species, even if the homolog were present and evolving normally.'''
    def __init__(self, spiecial_distances, ethresh):
        self.gene_length = abSENSE._gen_lookup(default=400)
        self.db_size = abSENSE._gen_lookup(default=8000000)
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
            detval = abSENSE.predict(currx, a_vals, b_vals)
            estnoise = abSENSE.estimate_noise(currx, a_vals, b_vals)

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
        self.gene_length = abSENSE._gen_lookup(gene_dict, default)

    def set_db_size(self, db_dict=None, default=None):
        '''Add species genome size data'''
        self.db_size = abSENSE._gen_lookup(db_dict, default)

    def test_orthology(self, bit_scores, gene=None):
        '''Computes orthology probabilities for a gene'''
        orthologs = [s for s, b in bit_scores.items() if b > 0]

        truncdistances = [self.spiecial_distance[s] for s in orthologs]
        genebitscores = [bit_scores[s] for s in orthologs]

        (a, b), covar = curve_fit(
            abSENSE.predict,
            truncdistances,
            genebitscores,
            bounds=((-np.inf, 0), (np.inf, np.inf)))

        preds = {s: abSENSE.predict(self.spiecial_distance[s], a, b) for s in bit_scores}

        testvals = [np.random.multivariate_normal([a, b], covar) for i in range(200)]

        results = {s: abSENSE.find_p_i(
          testvals,
          self.spiecial_distance[s],
          self.bitscore_threshold(gene,s)) for s in bit_scores}

        return {s: (preds[s], *results[s]) for s in bit_scores}
