#!/usr/bin/env python

import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import norm
import sys
import os
import argparse
import csv
from datetime import datetime
from math import exp, log

class abSENSE:
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
    return a * np.exp(-b * x)

  @staticmethod
  def estimate_noise(x, a, b):
    y = exp(-b * x)
    return np.sqrt(a * (1 - y) *  y)

  @staticmethod
  def PI_find(testvals, currx, bitthresh):
    #{s: bit_thresh(gene, s) for s in spiecial_distance}
    PIsamples = []
    for a_vals, b_vals in testvals:
      detval = abSENSE.predict(currx, a_vals, b_vals)
      estnoise = abSENSE.estimate_noise(currx, a_vals, b_vals)

      if estnoise > 0:
        PIsamples += [detval + np.random.normal(0, estnoise) for i in range(200)]
      else:
        PIsamples.append(detval)

    mean = np.mean(PIsamples)
    std = np.std(PIsamples)

    pval = norm.cdf(bitthresh, mean, std)
    (lowint, highint) = norm.interval(0.99, mean, std)

    return lowint, highint, pval

  def bitscore_threshold(self, gene, species):
    return -log(self.ethresh / (self.gene_length(gene) * self.db_size(species)), 2)

  def set_gene_length(self, gene_dict=None, default=None):
    self.gene_length = abSENSE._gen_lookup(gene_dict, default)

  def set_db_size(self, db_dict=None, default=None):
    self.db_size = abSENSE._gen_lookup(db_dict, default)

  def test_orthology(self, bit_scores, gene=None):
    orthologs = [s for s, b in bit_scores.items() if b > 0]

    truncdistances = [self.spiecial_distance[s] for s in orthologs]
    genebitscores = [bit_scores[s] for s in orthologs]
    bi_thresh_dict = {s: self.bitscore_threshold(gene, s) for s in bit_scores}

    (a, b), covar = curve_fit(
      abSENSE.predict,
      truncdistances,
      genebitscores,
      bounds=((-np.inf, 0), (np.inf, np.inf)))

    preds = {s: abSENSE.predict(self.spiecial_distance[s], a, b) for s in bit_scores}
    testvals = [np.random.multivariate_normal([a, b], covar) for i in range(200)]
    results = {s: abSENSE.PI_find(testvals, self.spiecial_distance[s], self.bitscore_threshold(gene,s)) for s in bit_scores}

    return {s: (preds[s], *results[s]) for s in bit_scores}



def tsv_to_dict(file_name, data_type=float):
  with open(file_name, "r") as stream:
    reader = csv.reader(stream, delimiter="\t")
    my_dict = {row[0]: data_type(row[1]) for row in reader}
    return my_dict

def load_spiecial_distances(distfile, includeonly=None):
  distances = {}
  with open(distfile, "r") as stream:
    reader = csv.reader(stream, delimiter="\t")
    for row in reader:
      if includeonly is None or row[0] in includeonly:
        distances[row[0]] = as_float(row[1])
  return distances

def as_float(v):
  try:
    return float(v)
  except:
    return False

def bit_score_reader(scorefile, species):
  with open(scorefile, 'r') as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    for row in reader:
      yield row[reader.fieldnames[0]], {k: as_float(row[k]) for k in species}

def csv_writer(file_path, fieldnames):
  stream = open(file_path, 'w', newline="\n")
  writer = csv.DictWriter(stream, fieldnames, delimiter="\t")
  writer.writeheader()
  return writer

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='abSENSE arguments:')

  parser.add_argument("--distfile", type=str, required=True, help="Required. Name of file containing pairwise evolutionary distances between focal species and each of the other species")
  parser.add_argument("--scorefile", type=str, required=True, help="Required. Name of file containing bitscores between focal species gene and orthologs in other species")
  parser.add_argument("--Eval", default=0.001, type=float, help="Optional. E-value threshold. Scientific notation (e.g. 10E-5) accepted. Default 0.001.")
  parser.add_argument("--includeonly", type=str, help="Optional. Species whose orthologs' bitscores will be included in fit; all others will be omitted. Default is all species. Format as species names, exactly as in input files, separated by commas (no spaces).")
  parser.add_argument("--genelenfile", type=str, help="Optional. File containing lengths (aa) of all genes to be analyzed. Used to accurately calculate E-value threshold. Default is 400aa for all genes. Only large deviations will qualitatively affect results.")
  parser.add_argument("--dblenfile", type=str, help="Optional. File containing size (aa) of databases on which the anticipated homology searches will be performed. Species-specific. Used to accurately calculate E-value threshold. Default is 400aa/gene * 20,000 genes for each species, intended to be the size of an average proteome. Only large deviations will significantly affect results.")
  parser.add_argument("--predall", type=bool, default=False, help="Optional. True: Predicts bitscores and P(detectable) of homologs in all species, including those in which homologs were actually detected. Default is False: only make predictions for homologs that seem to be absent.")
  starttime = datetime.now().strftime("%m.%d.%Y_%H.%M")
  parser.add_argument("--out", type=str, default=f'abSENSE_results_{starttime}', help="Optional. Name of directory for output data. Default is date and time when analysis was run.")

  args = parser.parse_args()
  np.random.seed(1)

  spiecial_distance = load_spiecial_distances(args.distfile, args.includeonly)

  if args.includeonly:
    include = args.includeonly.split(",")
  else:
    include = list(spiecial_distance.keys())

  my_abs = abSENSE(spiecial_distance, args.Eval)

  os.mkdir(args.out)
  spec_header = ["Gene"] + include
  mloutputfile = csv_writer(os.path.join(args.out, 'Predicted_bitscores'), spec_header)
  lowboundoutputfile = csv_writer(os.path.join(args.out, 'Bitscore_99PI_lowerbound_predictions'), spec_header)
  highboundoutputfile = csv_writer(os.path.join(args.out, 'Bitscore_99PI_higherbound_predictions'), spec_header)
  pvaloutputfile = csv_writer(os.path.join(args.out, 'Detection_failure_probabilities'), spec_header)
  #outputfileparams = csv_writer(os.path.join(args.out, 'Parameter_values'), ["Gene", "a", "b"])
  spec_files = [mloutputfile, lowboundoutputfile, highboundoutputfile, pvaloutputfile]

  for gene, bit_scores in bit_score_reader(args.scorefile, include):
    results = my_abs.test_orthology(bit_scores)
    for i in range(4):
      row = {k: round(v[i], 2) for k, v in results.items()}
      row["Gene"] = gene
      spec_files[i].writerow(row)