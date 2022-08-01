import os
import numpy
import pandas
import re

from metax import Exceptions, Utilities, MatrixManager
from .. import Utilities as BUtilities


def get_group(args, entry_whitelist=None):
    groups = {}
    keys = []
    if len(args.grouping) > 1    and args.grouping[1] != "GTEx_sQTL":
        raise  Exceptions.InvalidArguments("Unsupported grouping options")

    lines = Utilities.generate_from_any_plain_file(args.grouping[0], skip_n =1)
    for i,g in enumerate(lines):
        comps = g.strip().split()
        group = comps[1]
        entry = comps[2]
        
	# In multi-tissue context, entry will have the form:
	# <tissue_name>.intron_<chr>_<spos>_<epos>
        # However, the entry_whitelist still has introns without
	# the <tissue_name>. prefix, so let's check for the existence of a .
	# and only take the intron without the tissue when checking the whitelist.
        # if "." in entry:
            # entry_no_tiss = entry.split(".")[1]
        # else: 
            # entry_no_tiss = entry
        if entry_whitelist and entry not in entry_whitelist:
            continue

        if not group in groups:
            groups[group] =[]
            keys.append(group)
        groups[group].append(entry)

    return keys, groups


def get_covariance_source(path, whitelist=None):
    covariance_source = {}
    lines = Utilities.generate_from_any_plain_file(path, skip_n=1)
    for l in lines:
        comps = l.strip().split()
        r1 = comps[1]
        r2 = comps[2]

        if whitelist and (r1 not in whitelist or r2 not in whitelist):
            continue

        v = comps[3]
        if not r1 in covariance_source:
            covariance_source[r1] = {}
        # if not r2 in covariance_source[r1]: # This version of the line can remove covariances. . .
        if not r2 in covariance_source:
            covariance_source[r2] = {}
        covariance_source[r1][r2] = numpy.float64(v)
    return covariance_source


def get_associations(args):
    if args.associations_folder is not None:
        return(get_associations_from_multiple(args))
    associations = {}

    if len(args.associations) < 2 or args.associations[1] != "SPrediXcan":
        raise Exceptions.InvalidArguments("Unsupported association options")

    if args.tiss_list is not None:
        tiss_list = pandas.read_csv(args.tiss_list, header=None)[0].tolist()
        print("Tissue list.")
        print(tiss_list)
    lines = Utilities.generate_from_any_plain_file(args.associations[0], skip_n=1)
    for l in lines:
        comps = l.strip().split(",")
        entry = comps[0]
        if args.tiss_list is not None:
            tiss = entry.split(".")[0] # Textbook hardcoding right here, apologies
            if tiss not in tiss_list:
                continue
        association = comps[2]
        if association != "NA":
            associations[entry] = numpy.float64(association)

    return associations

def get_associations_from_multiple(args):
    file_regexp = re.compile(args.associations_file_pattern) if args.associations_file_pattern else  None
    if args.associations_tissue_pattern:
        tissue_regexp = re.compile(args.associations_tissue_pattern)
    else:
        raise Exceptions.InvalidArguments("Need a way to identify which association file is with which tissue")

    associations = {}
    files = BUtilities.contentsWithRegexpFromFolder(args.associations_folder, file_regexp)
    for assoc_file in files:
        tiss = tissue_regexp.search(os.path.basename(assoc_file)).groups()[0]
        if args.tiss_list is not None:
            if tiss not in tiss_list:
                continue
        lines = Utilities.generate_from_any_plain_file(os.path.join(args.associations_folder, assoc_file), skip_n=1)
        for l in lines:
            comps = l.strip().split(",")
            entry = comps[0]
            association = comps[2]
            if association != "NA":
                associations["{}.{}".format(tiss, entry)] = numpy.float64(association)
    return associations




def build_model_structure(model, whitelist=None):
    structure = {}
    for t in model.weights.itertuples():

        if "." in t.rsid:
            rsid_no_tiss = t.rsid.split(".")[1]
        else:
            rsid_no_tiss = t.rsid

        if whitelist and rsid_no_tiss not in whitelist:
            continue
        if t.gene not in structure:
            structure[t.gene] = {}
        structure[t.gene][t.rsid] = t.weight
    return structure


def get_features_and_variants(group, model_structure):
    variants = set()
    features = set()
    for feature in group:
        if feature in model_structure:
            variants.update(model_structure[feature].keys())
            features.add(feature)
    return features, variants


def get_weight_matrix(model_structure, features_, variants_):
    matrix = []
    for feature in features_:
        model = model_structure[feature]
        w = []
        for variant in variants_:
            if variant in model:
                w.append(model[variant])
            else:
                w.append(0)
        matrix.append(w)
    return numpy.matrix(matrix, dtype=numpy.float64)


def get_transcriptome_correlation(covariance_source, model_structure, features, variants, group_name):
    variants_ = sorted(variants)
    features_ = sorted(features)
    genotype_covariance = MatrixManager._to_matrix_2(covariance_source, variants_)
    weight = get_weight_matrix(model_structure, features_, variants_)
    transcriptome_covariance = numpy.dot(numpy.dot(weight, genotype_covariance), weight.T)
    zeros = numpy.where(transcriptome_covariance == 0)
    num_zeros = zeros[0].shape[0]
    if num_zeros > 0:
        print("{} is affected by SNPs with 0 covar".format(group_name))
    variances = numpy.diag(transcriptome_covariance)
    normalization = numpy.sqrt(numpy.outer(variances, variances))
    transcriptome_correlation = numpy.divide(transcriptome_covariance, normalization)
    return features_, variants_, transcriptome_correlation
