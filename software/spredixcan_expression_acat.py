import os
from glob import glob
import pandas as pd
import numpy as np
import re

def calc_1step_acat(results_dir,
                    study_pattern,
                    output,
                    tissue_pattern="__PM__(.*)\.csv",
                    replace_pval=True,
                    tiss_list=None,
                    split_acat=False
                    ):

    rfiles = glob(os.path.join(results_dir, study_pattern))
    pvals_dict, zscores_dict = {}, {}
    tissues_used = []
    if tiss_list is not None:
        whitelist = list(pd.read_csv(tiss_list, header=None)[0])
    for study_tiss_file in rfiles:
        print(study_tiss_file)
        tissue_regexp = re.compile(tissue_pattern)
        tissue = tissue_regexp.search(os.path.basename(study_tiss_file)).groups()[0]
        if tiss_list is not None:
            if tissue not in whitelist:
                continue
            else:
                tissues_used.append(tissue)
        print(tissue)

        if split_acat:
            df = pd.read_csv(study_tiss_file, usecols=["gene", "pvalue", "zscore"])
        else:
            df = pd.read_csv(study_tiss_file, usecols=["gene", "pvalue"])

        if len(df) == 0:
            continue
        
        for index, row in df.iterrows():
            pval = row["pvalue"]
            gene = row["gene"]
            if replace_pval is True:
                if pval == 1:
                    pval = pval - 1e-5

            if gene not in pvals_dict:
                pvals_dict[gene] = {"tissue": [tissue],
                                    "pval": [pval]}
            elif tissue not in pvals_dict[gene]["tissue"]:
                pvals_dict[gene]["tissue"].append(tissue)
                pvals_dict[gene]["pval"].append(pval)
            else:
                print("What's going on?")
                print(tissue)
                print(pval)

            # Grab zscore if splitting acat
            if split_acat:
                zscore = row["zscore"]
                if gene not in zscores_dict:
                    zscores_dict[gene] = {"tissue": [tissue],
                                        "zscore": [zscore]}
                elif tissue not in zscores_dict[gene]["tissue"]:
                    zscores_dict[gene]["tissue"].append(tissue)
                    zscores_dict[gene]["zscore"].append(zscore)
                else:
                    print("What's going on?")
                    print(tissue)
                    print(pval)

    if split_acat:
        out_cols = ["group", "acat", "acat_zscore_pos", "acat_zscore_neg", 
                    "acat_zscore_zero", "zscore_pos_count", "zscore_neg_count", "zscore_zero_count", 
                    "breast_pvalue", "group_size", "available_results"]
        results = []
        for gene in pvals_dict:
            acat, acat_zscore_pos, acat_zscore_neg, acat_zscore_zero, zscore_pos_count, \
                    zscore_neg_count, zscore_zero_count, \
                breast_pvalue, group_size, available_results = acat_by_direction(pvals_dict[gene], zscores_dict[gene])
            results.append((gene, acat, acat_zscore_pos, acat_zscore_neg, 
                acat_zscore_zero, zscore_pos_count, zscore_neg_count, zscore_zero_count, breast_pvalue,
                group_size, available_results))
        results = pd.DataFrame(results, columns=out_cols)
    else:
        out_cols = ["group", "acat", "breast_pvalue", "group_size", "available_results"]
        results = []
        for gene in pvals_dict:
            acat, breast_pvalue, group_size, available_results = single_acat(pvals_dict[gene])
            results.append((gene, acat, breast_pvalue, group_size, available_results))
        results = pd.DataFrame(results, columns=out_cols)

    if split_acat:
        write_gene_tiss_breakdown(pvals_dict, zscores_dict, og_output=output)
    results.to_csv(output, index=False, sep="\t")
    print("List of tissues used")
    print(tissues_used)
    return

def write_gene_tiss_breakdown(pvals_dict, zscores_dict, og_output):
    out_cols = ["group", "tissue", "zscore" , "pvalue"]

    results = []
    for gene in pvals_dict.keys():
        group = gene
        for tiss_index, cur_tissue in enumerate(pvals_dict[gene]['tissue']):
            tissue = cur_tissue

            pval = pvals_dict[gene]['pval'][tiss_index]

            tissue_z_index = zscores_dict[gene]["tissue"].index(cur_tissue)
            zscore = zscores_dict[gene]['zscore'][tissue_z_index]
            
            results.append((group, tissue, zscore, pval))

    results = pd.DataFrame(results, columns=out_cols)
    oname = "gene_tiss_breakdown_{}".format(os.path.basename(og_output))
    results.to_csv(oname, index=False, sep="\t")
    print("Results written at: {}".format(oname))
    return

def acat_by_direction(pval_dict, zscore_dict):
    try:
        breast_index = pval_dict["tissue"].index("Breast_Mammary_Tissue")
        breast_pvalue = pval_dict["pval"][breast_index]
    except ValueError:
        breast_pvalue = None

    zscore_pos, zscore_neg, zscore_zero = [], [], []
    pval_pos, pval_neg, pval_zero = [], [], []
    for tiss_zscore, tiss_pval in zip(zscore_dict['zscore'], pval_dict['pval']):
        if tiss_zscore > 0:
            pval_pos.append(tiss_pval)
        elif tiss_zscore < 0:
            pval_neg.append(tiss_pval)
        elif tiss_zscore == 0:
            pval_zero.append(tiss_pval)

    # Get statistics on retrievable tissues
    group_size = len(pval_dict["pval"]) # Total tissues
    available_results = len(pval_pos) + len(pval_neg) + len(pval_zero) # Tissues useable in ACAT
    
    zscore_pos_count, zscore_neg_count, zscore_zero_count = len(pval_pos), \
            len(pval_neg), len(pval_zero)

    acat_zscore_pos = simple_acat(pval_pos)
    acat_zscore_neg = simple_acat(pval_neg)
    acat_zscore_zero = simple_acat(pval_zero)

    # all_acat_scores = [acat_zscore_pos, acat_zscore_neg, acat_zscore_zero]
    # num_acat_scores = [zscore_pos_count, zscore_neg_count, zscore_zero_count]
    # all_viable_acat = np.array([ele for ele in all_acat_scores if ele is not None], dtype=np.float64)
    # weights_viable_acat = np.array([ele / available_results for ele in num_acat_scores if ele > 0], dtype=np.float64)
    # pieces_acat = simple_acat(all_viable_acat, weights=weights_viable_acat)

    acat, _, _, _ = single_acat(pval_dict) # The differences are incredibly small, so no need to calculate pieces_acat now
    # Comment when confident that ACAT is working as intended
    # diff = og_acat - pieces_acat
    # print("{} difference of og_acat - pieces_acat".format(diff))

    return acat, acat_zscore_pos, acat_zscore_neg, acat_zscore_zero, \
               zscore_pos_count, zscore_neg_count, zscore_zero_count, breast_pvalue, group_size, available_results
#             acat, acat_zscore_pos, acat_zscore_neg, acat_zscore_zero, zscore_zero_count, \
#                 breast_pvalue, group_size, available_results = split_acat(pvals_dict[gene], zscores_dict[gene])

    
def single_acat(pval_dict):
    try:
        breast_index = pval_dict["tissue"].index("Breast_Mammary_Tissue")
        breast_pvalue = pval_dict["pval"][breast_index]
    except ValueError:
        breast_pvalue = None

    group_size = len(pval_dict["pval"]) # Total tissues
    pval_arr = np.array(pval_dict["pval"], dtype=np.float64)
    pval_arr = pval_arr[np.logical_not(np.isnan(pval_arr))]
    available_results = pval_arr.shape[0] # Tissues useable in ACAT

    acat = simple_acat(pval_arr)

    return acat, breast_pvalue, group_size, available_results

def simple_acat(pvals, weights=None):
    if len(pvals) == 0:
        return(None)
    # No NANs
    if type(pvals) is list:
        pval_arr = np.array(pvals, dtype=np.longdouble)
    else:
        pval_arr = pvals

    if weights is None:
        weights = np.repeat(1, pval_arr.shape[0])

    s = np.sum(weights * np.tan((0.5 - pval_arr) * np.pi))
    acat = 0.5 - np.arctan(s / np.sum(weights)) / np.pi
    return(acat)

def run(args):
    if os.path.exists(args.output):
        print("Output exists. Remove it or delete it.")
        return

    calc_1step_acat(args.results_dir, args.study_pattern, args.output, tiss_list=args.tiss_list,
            split_acat=args.split_acat) 

if __name__ == "__main__":
    import  argparse
    parser = argparse.ArgumentParser("Combine the sqtl associations of a study across all relevant tissues.")

    parser.add_argument("--results_dir", help="Directory of spredixcan eqtl results")
    parser.add_argument("--study_pattern", help="Pattern to match study's eqtl files")
    parser.add_argument("--tiss_list", default=None, help="File with list of tissues to filter by, one column only, no header. Tissues must match exactly the <tissue_name> in the associations file tissue/intron format <tissue_name>.<intron_id>.")
    parser.add_argument("--split_acat", action='store_true', default=False, help="Also show ACAT by zscore direction")
    parser.add_argument("--output", help="Name of the output")
    args = parser.parse_args()

    run(args)
