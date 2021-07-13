import logging
import string
import pandas

from .. import Constants
from .. import PredictionModel
from ..gwas import Utilities as GWASUtilities

EA, NEA = Constants.EFFECT_ALLELE, Constants.NON_EFFECT_ALLELE
EA_BASE, NEA_BASE = EA + "_BASE", NEA + "_BASE"

def align_data_to_alleles(data, base, left_on, right_on):
    merged = pandas.merge(data, base, left_on=left_on, right_on=right_on, suffixes=("", "_BASE"))

    alleles_1 = pandas.Series([set(e) for e in zip(merged[EA], merged[NEA])])
    alleles_2 = pandas.Series([set(e) for e in zip(merged[EA_BASE], merged[NEA_BASE])])
    eq = alleles_1 == alleles_2
    merged = merged[eq]
    if eq.shape[0] == 0:
        return merged

    flipped = merged[EA] != merged[EA_BASE]
    Z = Constants.ZSCORE
    if Z in merged:
        merged.loc[flipped, Z] = - merged.loc[flipped, Z]
    B = Constants.BETA
    if B in merged:
        merged.loc[flipped, B] = - merged.loc[flipped, B]

    merged.loc[flipped, EA] = merged.loc[flipped, EA_BASE]
    merged.loc[flipped, NEA] = merged.loc[flipped, NEA_BASE]

    return merged

def model_and_intersection_with_gwas_from_args(args):
    gwas= GWASUtilities.load_plain_gwas_from_args(args)
    PF = PredictionModel.WDBQF
    logging.log(9, "loading %s", args.model_db_path)
    model = PredictionModel.load_model(args.model_db_path, args.model_db_snp_key)
    base = model.weights[[PF.K_RSID, PF.K_EFFECT_ALLELE, PF.K_NON_EFFECT_ALLELE]].drop_duplicates()

    # Multi-Tissue IntronXCan has the db file with rsids of the form:
    # <tissue_name_.<SNP_info>
    # gwas file will only have <SNP_info> so we create and use a separate rsid
    # column with only snpinfo in it.
    if "." in base['rsid'][0]:
        base['rsid_no_tiss'] = base['rsid'].str.split(".", expand=True)[1]
        b = align_data_to_alleles(gwas, base, Constants.SNP, 'rsid_no_tiss')
    else:
        b = align_data_to_alleles(gwas, base, Constants.SNP, PF.K_RSID)
    return model, set(b[Constants.SNP])

def gwas_models_intersection_from_args(args):
    gwas= GWASUtilities.load_plain_gwas_from_args(args)
    paths = PredictionModel._model_paths(args.models_folder, args.models_name_filter)
    PF = PredictionModel.WDBQF
    intersection = set()
    for db_path in sorted(paths):
        logging.log(9, "loading %s", db_path)
        model = PredictionModel.load_model(db_path, args.model_db_snp_key)
        base = model.weights[[PF.K_RSID, PF.K_EFFECT_ALLELE, PF.K_NON_EFFECT_ALLELE]].drop_duplicates()
        b = align_data_to_alleles(gwas, base, Constants.SNP, PF.K_RSID)
        intersection.update(b[Constants.SNP])
    return intersection

try:
    #python 3
    complement_translation = "CGTA".maketrans({"C": "G", "G": "C", "T":"A", "A": "T"})
except:
    #python 2
    complement_translation = string.maketrans("CGTA", "GCAT")

def match_alleles(l_non_effect_allele, l_effect_allele, r_non_effect_allele, r_effect_allele):
    if len(l_effect_allele) == 1 and len(l_non_effect_allele) == 1:
        if l_effect_allele == r_effect_allele and  l_non_effect_allele == r_non_effect_allele:
            return 1, 1
        if l_effect_allele == r_non_effect_allele and l_non_effect_allele == r_effect_allele:
            return -1, 1

        t_r_non_effect = r_non_effect_allele.translate(complement_translation)
        t_r_effect = r_effect_allele.translate(complement_translation)

        if l_effect_allele == t_r_effect and  l_non_effect_allele == t_r_non_effect:
            return 1, -1
        if l_effect_allele == t_r_non_effect and l_non_effect_allele == t_r_effect:
            return -1, -1

        return None, None
    else:
        if l_effect_allele == r_effect_allele and  l_non_effect_allele == r_non_effect_allele:
            return 1, 1

        t_r_non_effect = r_non_effect_allele.translate(complement_translation)
        t_r_effect = r_effect_allele.translate(complement_translation)

        if l_effect_allele == t_r_effect and  l_non_effect_allele == t_r_non_effect:
            return 1, -1

        return None, None

