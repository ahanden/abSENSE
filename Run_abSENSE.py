#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import glob
import sys
import random
from scipy.optimize import curve_fit
from scipy.stats import chi2
import inspect
import math
import re
import sys
import glob
import os
import warnings
import argparse
import csv
from datetime import datetime
from scipy import stats

###### Define user inputs #####

class OutputWriter:
  def __init__(self, output_dir_path):
    self.output_dir_path = output_dir_path

  def _setup_stream(self, file_name):
    path = os.path.join(self.output_dir_path, file_name)
    stream = open(path, "w")
    writer = csv.writer(stream, delimiter="\t")
    return {
      "stream": stream,
      "writer": writer
    }

  def _write_header(output, speciesorder, invordervec, description):
    # Write headers to file (first column will have the gene name;
    # subsequent columns have prediction info)
    output["writer"].writerow([f'# {description}'])
    header = [speciesorder[species] for species in invordervec]
    output["writer"].writerow(["Gene"] + header)

  def open(self):
    os.mkdir(self.output_dir_path)

    self.mloutput = self._setup_stream('Predicted_bitscores')
    self.lowboundoutput = self._setup_stream('Bitscore_99PI_lowerbound_predictions')
    self.highboundoutput = self._setup_stream('Bitscore_99PI_higherbound_predictions')
    self.pvaloutputfile = self._setup_stream('Detection_failure_probabilities')
    self.outputfileparams = self._setup_stream('Parameter_values')

    self._write_header(
      self.mloutput,
      speciesorder,
      invordervec,
      'This file contains maximum likelihood bitscore predictions for each tested gene in each species')
    self._write_header(
      self.lowboundoutput,
      speciesorder,
      invordervec,
      'This file contains the lower bound of the 99% bitscore prediction interval for each tested gene in each species')
    self._write_header(
      self.highboundoutput,
      speciesorder,
      invordervec,
      'This file contains the upper bound of the 99% bitscore prediction interval for each tested gene in each species')
    self._write_header(
      self.pvaloutput,
      speciesorder,
      invordervec,
      'This file contains the probability of a homolog being undetected at the specified significance threshold (see run info file) in each tested gene in each species')

    self.outputfileparams.write('# This file contains the best-fit parameters (performed using only bitscores from species not omitted from the fit; see run info file) for a and b for each gene\n')
    self.outputfileparams.write('Gene')
    write_cell(self.outputfileparams, 'a')
    write_cell(self.outputfileparams, 'b', True)

  def close(self):
    self.mloutput["stream"].close()
    self.lowboundoutput["stream"].close()
    self.highboundoutput["stream"].close()
    self.pvaloutput["stream"].close()
    self.outputfileparams["stream"].close()

def read_tsv(file_name):
  return np.genfromtxt(file_name, dtype=str, delimiter='\t')

## Find species db sizes in the right order, from either file or manual input (done above)
def find_total_lengths(speciesorder, speciesdblengths, speciesdblengthfilefound):
  speciestotallengths = []
  for species in speciesorder:
    found = False
    for j, length in enumerate(speciesdblengths):
      if species in length[0]:
        speciestotallengths.append(float(length[1]))
        found = True
    if not found:
      if speciesdblengthfilefound:
        sys.exit('One or more species names in your database size file do not match species names in distance file! The first I encountered was ' + speciesorder[i] + '. Quitting. \n')
  return speciestotallengths

## curve to fit
func = lambda x, a, b: a * np.exp(-b * x)

def write_cell(stream, content, new_line=False):
  stream.write(f'\t{content}' + ("\n" if new_line else ""))

def isfloat(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

## function to, where possible, use maximum likelihood estimates of a and
## b parameter plus estimated covariance matrix to directly sample from
## the probability distribution of a and b (assume Gaussian with mean of
## max likelihood estimates and given covariance structure)
def parameter_CI_find(mla, mlb, covar):
  if True in np.isinf(covar):
    return 'failed'

  testavals = []
  testbvals = []

  # Can definitely rewrite this better
  for i in range(200):
    (a, b) = np.random.multivariate_normal([mla, mlb], covar)
    testavals.append(a)
    testbvals.append(b)
  return testavals, testbvals

## function to take each of the sampled a, b values and use them to sample
## directly from the distribution of scores taking into account the
## Gaussian noise (a function of distance, a, b)
## this gives an empirical estimate of the prediction interval
def PI_find(testavals, testbvals, currx):
  # sample from score distribution: Gaussian with mean a, b and noise
  # determined by distance (currx), a, b
  PIsamples = []
  for a_vals, b_vals in zip(testavals, testbvals):
    detval = func(currx, a_vals, b_vals)

    x = math.exp(-1 * b_vals * currx)
    estnoise = np.sqrt(a_vals * (1 - x) *  x)

    if estnoise > 0:
      PIsamples += [detval + np.random.normal(0, estnoise) for i in range(200)]
    else:
      PIsamples.append(detval)

  mean = np.mean(PIsamples)
  std = np.std(PIsamples)

  # calculate this analytically from std estimate
  pval = stats.norm.cdf(bitthresh, mean, std)

  # calculate 99% CI
  (lowint, highint) = stats.norm.interval(0.99, mean, std)

  return lowint, highint, pval

def order_species(include , bitscores, speciesorder):
  ordervec = []
  pred_spec_locs = []

  for i, species in enumerate(speciesorder):
    found = False
    for j, score in enumerate(bitscores):
      if species in score:
        ordervec.append(j)
        found = True
        if species in include:
          pred_spec_locs.append(i)
    if not found:
      sys.exit('One or more species names in header of bitscore file do not match species names in header of distance file! The first I encountered was ' + speciesorder[i] + '. Quitting. \n')
  return ordervec, pred_spec_locs

def invert_vec(bitscores, speciesorder):
  invordervec = [] # first index is location of first species in file in speciesorder; and so on
  for b in bitscores:
    for j, species in enumerate(speciesorder):
      if b in species:
        invordervec.append(j)
  return invordervec

def write_header(stream, speciesorder, invordervec, description):
  # Write headers to file (first column will have the gene name; subsequent columns have prediction info)
  stream.write(f'# {description}\n')
  write_cell(stream, 'Gene')
  for species in invordervec:
    write_cell(stream, speciesorder[species])
  stream.write('\n')

'''
TODO
def log_run_info(file_path, starttime, args, pred_specs, ethresh):
  stream = open(file_path, 'w')
  stream.write(f'''
abSENSE analysis run on {starttime}
Input bitscore file: {args.scorefile}
Input distance file: {args.distfile}
Gene length file: {args.genelenfile}
Database length file: {args.dblenfile}
Species used in fit: {"All (default)" if len(pred_specs) == 0 else " ".join(pred_specs)}
E-value threshold: {ethresh} (default)
''')
  if not genelengthfilefound:
    runinfofile.write(f'Gene length (for all genes): {defgenelen} (default)\n')
  if not speciesdblengthfilefound:
    runinfofile.write(f'Database length (for all species): {defdbsize} (default)\n')
  runinfofile.close()
'''
def get_seqlen(gene, genelengthfilefound, genelengths):
  if genelengthfilefound:
    for lengths in genelengths[1:]:
      if gene in lengths:
        return float(lengths[1])
    sys.exit(f'Gene {gene} not found in specified gene length file! Quitting \n')
  return defgenelen

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
  parser.add_argument("--out", type=str, default='abSENSE_results_' + starttime, help="Optional. Name of directory for output data. Default is date and time when analysis was run.")

  args = parser.parse_args()

  distancefile = np.transpose(read_tsv(args.distfile))
  speciesorder = distancefile[0]
  rawdistances = distancefile[1].astype(float)

  bitscores = read_tsv(args.scorefile)
  genelist = bitscores[1:,0] # skip header

  ethresh = args.Eval

  defgenelen = float(400)
  genelengthfilefound = args.genelenfile is not None
  if genelengthfilefound:
    genelengths = read_tsv(args.genelenfile)

  defdbsize = float(8000000)
  speciesdblengthfilefound = args.dblenfile is not None
  if speciesdblengthfilefound:
    speciesdblengths = read_tsv(args.dblenfile)
  else:
    speciesdblengths = np.transpose(np.vstack((speciesorder, [float(defdbsize)]*len(speciesorder))))

  pred_specs = re.split(',', args.includeonly) if args.includeonly else []
  ordervec, pred_spec_locs = order_species(pred_specs , bitscores[0], speciesorder)
  invordervec = invert_vec(bitscores[0], speciesorder)

  speciestotallengths = find_total_lengths(speciesorder, speciesdblengths, speciesdblengthfilefound)

  os.mkdir(args.out)
  output = OutputWriter(args.out)
  output.open()

  runinfofile = open(outpath('Run_info'), 'w')
  runinfofile.write(f'''
abSENSE analysis run on {starttime}
Input bitscore file: {args.scorefile}
Input distance file: {args.distfile}
Gene length file: {args.genelenfile}
Database length file: {args.dblenfile}
Species used in fit: {"All (default)" if len(pred_specs) == 0 else " ".join(pred_specs)}
E-value threshold: {ethresh} (default)
''')
  if not genelengthfilefound:
    runinfofile.write(f'Gene length (for all genes): {defgenelen} (default)\n')
  if not speciesdblengthfilefound:
    runinfofile.write(f'Database length (for all species): {defdbsize} (default)\n')
  runinfofile.close()

  # Ignore warning that sometimes happen as a result of stochastic sampling but that doesn't affect overall computation
  warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

  print('Running!')

  #file_names = [
  #  mloutputfile,
  #  lowboundoutputfile,
  #  highboundoutputfile,
  #  outputfileparams,
  #  pvaloutputfile ]

  for gene in genelist:
    # report current position and gene name
    print(f'gene: {gene}')

    # print current gene to output file
    #for f in file_names:
    #  f.write(gene)

    # make new arrays to put truncated (not using omitted species) scores,
    # distances
    genebitscores = []
    truncdistances = []

    # make new arrays to indicate which species/distances are ambiguous
    # orthology but have a homolog; don't predict these later
    ambigdists =  []

    # if gene length input file given, look for length of current gene
    # if not given, assume default value (specified above)
    seqlen = get_seqlen(gene, genelengthfilefound, genelengths)

    # put scores for current gene in bitscore file in right order
    orderedscores = []
    for species in ordervec: # ordervec starts at 1
      orderedscores.append(bitscores[i+1][species]) ##
    # i + 1 because header skipped in gene list formation, so one behind now
    # append score of species and corresponding distance of species to gene-specific distance, score vectors if:
    # score isn't 0 (can't distinguish absence from loss from bad data etc)
    # score isn't 'N/A' or some other string (ie indicator of unclear orthology or otherwise absent, or generally unclear what it is)
    # score isn't from species that is to be excluded from fit
    for k in range(len(orderedscores)):
      if isfloat(orderedscores[k]) and orderedscores[k] != '0':
        if len(pred_spec_locs) == 0 or k in pred_spec_locs:
          genebitscores.append(float(orderedscores[k]))
          truncdistances.append(rawdistances[k])
        elif orderedscores[k] == 'N/A':
          ambigdists.append(rawdistances[k])

    if len(truncdistances) > 2:
      try:
        (a, b), covar = curve_fit(func, truncdistances, genebitscores, bounds=((-np.inf, 0), (np.inf, np.inf)))
      except RuntimeError:
        for file_stream in [mloutputfile, highboundoutputfile, lowboundoutputfile, pvaloutputfile]:
          for j in range(len(rawdistances) - 1):
            write_cell(file_stream, "analysis_error")
          write_cell(file_stream, "analysis_error", True)
        write_cell(outputfileparams, "analysis_error")
        write_cell(outputfileparams, "analysis_error", True)
        continue
      parout = parameter_CI_find(a, b, covar)
      if parout != 'failed':
        testavals, testbvals = parout
        for j in range(len(rawdistances)):
          species = invordervec[j]
          distance = rawdistances[species]
          bitthresh = -math.log(ethresh / (seqlen * speciestotallengths[species]), 2)
          prediction = round(func(distance, a, b), 2)
          lowprediction, highprediction, pval = PI_find(testavals, testbvals, distance)
          highprediction = round(highprediction, 2)
          lowprediction = round(lowprediction, 2)
          pval = round(pval, 2)

          if distance in truncdistances:
            if args.predall:
              '(Ortholog_detected)'
            else:
              stat = 'Ortholog_detected'
          elif distance in ambigdists:
            stat = 'Homolog_detected(orthology_ambiguous)'
          else:
            stat = ''

          if stat == '(Ortholog_detected)':
            realscore = genebitscores[truncdistances.index(distance)]
            write_cell(mloutputfile, f'{prediction}(Ortholog_detected:{realscore})')
          else:
            write_cell(mloutputfile, f'{prediction}{stat}')
          write_cell(highboundoutputfile, f'{highprediction}{stat}')
          write_cell(lowboundoutputfile, f'{lowprediction}{stat}')
          write_cell(pvaloutputfile, f'{pval}{stat}')

        mloutputfile.write('\n')
        highboundoutputfile.write('\n')
        lowboundoutputfile.write('\n')
        pvaloutputfile.write('\n')
        write_cell(outputfileparams, a)
        write_cell(outputfileparams, b, True)
      else:
        for j in range(len(rawdistances)):
          prediction = round(func(rawdistances[j], a, b), 2)
          write_cell(mloutputfile, prediction)
          write_cell(highboundoutputfile, 'analysis_error')
          write_cell(lowboundoutputfile, 'analysis_error')
          write_cell(pvaloutputfile, 'analysis_error')
        mloutputfile.write('\n')
        highboundoutputfile.write('\n')
        write_cell(outputfileparams, 'analysis_error')
        write_cell(outputfileparams, 'analysis_error', True)
        lowboundoutputfile.write('\n')
        pvaloutputfile.write('\n')
    else:
      for j in range(0, len(rawdistances)):
        write_cell(mloutputfile, 'not_enough_data')
        write_cell(highboundoutputfile, 'not_enough_data')
        write_cell(lowboundoutputfile, 'not_enough_data')
        write_cell(pvaloutputfile, 'not_enough_data')
      mloutputfile.write('\n')
      highboundoutputfile.write('\n')
      lowboundoutputfile.write('\n')
      pvaloutputfile.write('\n')
      write_cell(outputfileparams, 'not_enough_data')
      write_cell(outputfileparams, 'not_enough_data', True)
  mloutputfile.close()
  highboundoutputfile.close()
  lowboundoutputfile.close()
  outputfileparams.close()
  pvaloutputfile.close()
