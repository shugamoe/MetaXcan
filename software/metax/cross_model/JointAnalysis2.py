import numpy

from metax import Exceptions, Utilities, MatrixManager


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
        if not r2 in covariance_source[r1]:
            covariance_source[r2] = {}
        covariance_source[r1][r2] = float(v)
    return covariance_source


def get_associations(args):
    associations = {}

    if len(args.associations) < 2 or args.associations[1] != "SPrediXcan":
        raise Exceptions.InvalidArguments("Unsupported association options")

    lines = Utilities.generate_from_any_plain_file(args.associations[0], skip_n=1)
    for l in lines:
        comps = l.strip().split(",")
        entry = comps[0]
        association = comps[2]
        if association != "NA":
            associations[entry] = float(association)

    return associations


def build_model_structure(model, whitelist=None):
    structure = {}
    for t in model.weights.itertuples():
        if whitelist and t.rsid not in whitelist:
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


def get_transcriptome_correlation(covariance_source, model_structure, features, variants):
    variants_ = sorted(variants)
    features_ = sorted(features)
    genotype_covariance = MatrixManager._to_matrix_2(covariance_source, variants_)
    weight = get_weight_matrix(model_structure, features_, variants_)
    transcriptome_covariance = numpy.dot(numpy.dot(weight, genotype_covariance), weight.T)
    variances = numpy.diag(transcriptome_covariance)
    normalization = numpy.sqrt(numpy.outer(variances, variances))
    transcriptome_correlation = numpy.divide(transcriptome_covariance, normalization)
    return features_, variants_, transcriptome_correlation