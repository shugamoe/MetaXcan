import os
import logging 
import numpy
import pandas
from scipy import stats
from timeit import default_timer as timer

from metax import Utilities, Logging
from metax.cross_model import JointAnalysis2
from metax.misc import GWASAndModels, Math, KeyedDataSource
from metax.gwas import Utilities as GWASUtilities
from metax.cross_model import Utilities as SMultiXcanUtilities


########################################################################################################################

def run(args):
    start = timer()
    if os.path.exists(args.output):
        logging.info("Output exists. Remove it or delete it.")
        return

    Utilities.ensure_requisite_folders(args.output)

    cutoff = SMultiXcanUtilities._cutoff(args)

    logging.info("Acquiring gwas-variant intersection")
    model, intersection = GWASAndModels.model_and_intersection_with_gwas_from_args(args)
    model_structure = JointAnalysis2.build_model_structure(model, intersection)

    logging.info("Acquiring groups")
    group_keys, groups = JointAnalysis2.get_group(args, set(model_structure.keys()))

    logging.info("Acquiring covariance source")
    covariance_source = JointAnalysis2.get_covariance_source(args.covariance, intersection)

    logging.info("Acquiring associations")
    associations = JointAnalysis2.get_associations(args)
    if args.acat:
        associations_p = KeyedDataSource.load_data_from_assoc(args, "gene", "pvalue", sep=",")

    reporter = Utilities.PercentReporter(logging.INFO, len(group_keys))

    results = []
    for i,group_name in enumerate(group_keys):
        if args.debug:
            # if group_name not in ['ENSG00000160796.16', 'ENSG00000265531.3']: # DEBUG only, remove before commit
            if group_name not in ['ENSG00000265531.3']: # DEBUG only, remove before commit
                continue
        if args.single_gene and group_name != args.single_gene:
            continue
        reporter.update(i+1, "processed %d %% of groups")
        if args.MAX_M and  i>args.MAX_M-1:
            logging.info("Early abort")
            break

        logging.log(8, "Processing group for %s", group_name)
        group_name, chi2_p, n_variants, n_features, n_indep, tmi, status, mtiss_acat, breast_acat, n_tiss, n_features_pval_mod = group_name, None, None, None, None, None, None, None, None, None, None
        try:
            group = [x for x in groups[group_name] if x in associations]

            features, variants = JointAnalysis2.get_features_and_variants(group, model_structure)
            if not features or not variants:
                continue
            n_features, n_variants = len(features), len(variants)
            features_, variants_, transcriptome_correlation = JointAnalysis2.get_transcriptome_correlation(covariance_source, model_structure, features, variants, group_name)

            zscores = numpy.array([associations[x] for x in features_], dtype=numpy.float64)
            cutoff_ = cutoff(transcriptome_correlation)
            inv, n_indep, eigen = Math.capinv(transcriptome_correlation, cutoff_, args.regularization, group_name, args.cutoff_sumrule) 

            w = float(numpy.dot(numpy.dot(zscores, inv), zscores))
            chi2_p = stats.chi2.sf(w, n_indep)

            # import pickle as pkl # DEBUG ONLY REMOVE BEFORE COMMIT
            # with open('/gpfs/data/gao-lab/sing_hub/data/intronxcan_debug/{}/ixc_eg_gene_introns_by_tiss.pkl'.format(group_name), 'wb') as file:
            #     pkl.dump(features_, file)
            # with open('/gpfs/data/gao-lab/sing_hub/data/intronxcan_debug/{}/ixc_eg_gene_zscores_by_tiss.pkl'.format(group_name), 'wb') as ofile:
            #     pkl.dump(zscores, ofile)

            tmi = numpy.trace(numpy.dot(transcriptome_correlation, inv))
            if args.acat:
                if args.tiss_rank:
                    tiss_acat_dict, n_tiss, n_features_pval_mod, tiss_introns_dict, tiss_pvals_dict = calc_mtiss_acat(associations_p, features_, variants_, group_name, list(zscores), tiss_rank=args.tiss_rank, tiss_list=args.tiss_list)
                else:
                    mtiss_acat, breast_acat, n_tiss, n_features_pval_mod = calc_mtiss_acat(associations_p, features_, variants_, group_name, list(zscores), tiss_list=args.tiss_list)
        except Exception as e:
            #TODO: improve error tracking
            logging.log(8, "{} Exception: {}".format(group_name, str(e)))
            status = "Error"
            if args.acat:
                if args.tiss_rank:
                    tiss_acat_dict, n_tiss, n_features_pval_mod, tiss_introns_dict, tiss_pvals_dict = calc_mtiss_acat(associations_p, features_, variants_, group_name, list(zscores), tiss_rank=args.tiss_rank, tiss_list=args.tiss_list)
                else:
                    mtiss_acat, breast_acat, n_tiss, n_features_pval_mod = calc_mtiss_acat(associations_p, features_, variants_, group_name, list(zscores), tiss_list=args.tiss_list)
        if args.acat:
            if args.tiss_rank:
                for tiss in tiss_acat_dict:
                    for int_idx, intron in enumerate(tiss_introns_dict[tiss]):
                        results.append((group_name, chi2_p, n_variants, n_features, n_indep, tmi, status, tiss, intron, tiss_pvals_dict[tiss][int_idx], tiss_acat_dict[tiss], n_tiss, n_features_pval_mod))
            else:
                results.append((group_name, chi2_p, n_variants, n_features, n_indep, tmi, status, mtiss_acat, breast_acat, n_tiss, n_features_pval_mod))
        else:
            results.append((group_name, chi2_p, n_variants, n_features, n_indep, tmi, status))
    
    out_cols = ["group", "pvalue", "n_variants", "n_features", "n_indep", "tmi", "status"]
    if args.acat:
        if args.tiss_rank:
            out_cols += ["tissue", "intron", "intron_pval", "tissue_acat", "n_tiss", "n_features_pval_mod"]
        else:
            out_cols += ["mtiss_acat", "breast_acat", "n_tiss", "n_features_pval_mod"]
    results = pandas.DataFrame(results, columns=out_cols)

    results = results.sort_values(by="pvalue")
    if len(results) == 0 & (args.single_gene is not None):
        logging.info("Couldn't calculate gene: {}".format(args.single_gene))
    logging.info("Saving results")
    Utilities.save_dataframe(results, args.output)

    end = timer()
    logging.info("Successfully processed grouped SMultiXcan in %s seconds" % (str(end - start)))

def calc_mtiss_acat(associations_p, features_, variants_, group_name, zscores, output_debug_file=False,
        tiss_rank=False, tiss_list=None):
    tiss_pvals_dict = {}
    features_tissues_ = []
    features_introns_ = []
    tiss_introns_dict = {}

    pvals = [associations_p[x] for x in features_]
    def mod_val(features_, val=1, mod_val=1 - 1E-5):
        val_left_in = True
        num_mod = 0
        while val_left_in:
            try:
                mod_index = pvals.index(val)
                pvals[mod_index] = mod_val
                num_mod += 1
            except ValueError:
                val_left_in = False
        return num_mod
    n_features_mod = mod_val(features_) # Uncomment to remove introns with pval=1

    if tiss_list is not None:
        whitelist = list(pandas.read_csv(tiss_list, header=None)[0])
    for tiss_intron in features_:
        tiss, intron = tiss_intron.split(".")
        if tiss_list is not None:
            if tiss not in whitelist:
                continue
        features_tissues_.append(tiss)
        features_introns_.append(intron)
        if tiss not in tiss_pvals_dict:
            tiss_pvals_dict[tiss] = [associations_p[tiss_intron]]
            tiss_introns_dict[tiss] = [intron]
        else:
            tiss_pvals_dict[tiss].append(associations_p[tiss_intron])
            tiss_introns_dict[tiss].append(intron)

    def single_acat(pval_list):
        pval_arr = numpy.array(pval_list, dtype=numpy.float64)
        s = numpy.sum(numpy.tan((0.5 - pval_arr) * numpy.pi))
        acat = 0.5 - numpy.arctan(s / pval_arr.shape[0]) / numpy.pi
        return acat

    tiss_acat_dict = {}
    for tiss in tiss_pvals_dict.keys():
        tiss_acat_dict[tiss] = single_acat(tiss_pvals_dict[tiss])
    
    if tiss_rank:
        pass
    else:
        final_acat = single_acat([tiss_acat_dict[tiss] for tiss in tiss_acat_dict.keys()])
    n_tiss = len(tiss_pvals_dict.keys())

    if 'Breast_Mammary_Tissue' in tiss_acat_dict.keys():
        breast_acat = tiss_acat_dict['Breast_Mammary_Tissue']
    else:
        breast_acat = None
    
    if output_debug_file:
        debug_odir = "/gpfs/data/gao-lab/sing_hub/data/airlock/acat_2step/debug/zscore_remove/"
        # tissue	gene	intron	zscore	pvalue	1tiss_acat_pvalue
        oput_df = pandas.DataFrame({'tissue': features_tissues_,
                                    'gene': group_name,
                                    'intron': features_introns_,
                                    'zscore': zscores,
                                    'pvalue': [associations_p[intron] for intron in features_],
                                    '1tiss_acat_pvalue': [tiss_acat_dict[tiss] for tiss in features_tissues_]}) 
        oput_df = oput_df[['tissue', 'gene', 'intron', 'zscore', 'pvalue', '1tiss_acat_pvalue']]
        oput_df.to_csv("{}{}_BCAC_overall_2step_acat_debug.csv".format(debug_odir, group_name), index=False)
    if tiss_rank:
        return(tiss_acat_dict, n_tiss, n_features_mod, tiss_introns_dict, tiss_pvals_dict)
    else:
        return(final_acat, breast_acat, n_tiss, n_features_mod)
    

if __name__ == "__main__":
    import  argparse
    parser = argparse.ArgumentParser("Compute MultiXcan on multiple features, given a grouping (i.e. MulTiXcan on all splicing events -features-  of a given gene -group-)")

    parser.add_argument("--gwas_folder", help="name of folder containing GWAS data. All files in the folder are assumed to belong to a single study." 
                                              "If you provide this, you are likely to need to pass a --gwas_file_pattern argument value.")
    parser.add_argument("--gwas_file_pattern", help="Pattern to recognice GWAS files in folders (in case there are extra files and you don't want them selected).")
    parser.add_argument("--gwas_file", help="Load a single GWAS file. (Alternative to providing a gwas_folder and gwas_file_pattern)")
    GWASUtilities.add_gwas_arguments_to_parser(parser)

    parser.add_argument("--model_db_path", help="path to model file")
    parser.add_argument("--model_db_snp_key", help="Specify a key to use as snp_id")
    parser.add_argument("--covariance", help="path to file containing covariance data")
    
    # Use these together
    parser.add_argument("--associations", nargs="+", help="Entries association")
    parser.add_argument("--associations_folder", help="Name of folder containing entries association data.")
    parser.add_argument("--associations_file_pattern", help="Pattern to recognize associations files in folders (in case there are extra files and you don't want them selected).")
    parser.add_argument("--associations_tissue_pattern", help="Pattern to recognize the tissues corresponding to an association file")

    parser.add_argument("--grouping", nargs="+", help="File containing groups of features.", default = [])

    parser.add_argument("--cutoff_sumrule", help="If True, keep eigenvalue j if sum from i=0 to i=j >= 95% of sum of eigenvalues.", default=None, type=float)
    parser.add_argument("--cutoff_condition_number", help="condition number of eigen values to use when truncating SVD", default=None, type=float)
    parser.add_argument("--cutoff_eigen_ratio", help="ratio to use when truncating SVD", default=None, type=float)
    parser.add_argument("--cutoff_threshold", help="threshold of variance when truncating SVD", default=None, type=float)
    parser.add_argument("--cutoff_trace_ratio", help="ratio to use when truncating SVD", default=None, type=float)
    parser.add_argument("--regularization", help="Add a regularization term to correct for expression covariance matrix singularity", default=None, type=float)

    parser.add_argument("--MAX_M", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--verbosity", help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything", default=10, type=int)
    parser.add_argument("--tiss_list", default=None, help="File with list of tissues to filter by, one column only, no header. Tissues must match exactly the <tissue_name> in the associations file tissue/intron format <tissue_name>.<intron_id>.")
    parser.add_argument("--acat", action='store_true', default=False, help="Whether or not to provide a p-value calculated using ACAT only (by tissue for the gene with ACAT and then combining by all tissue ACAT p-values using ACAT).")

    parser.add_argument("--single_gene", help="Specify that you want to confine the analysis to a single gene. Typically used after COJO in a conditioned TWAS procedure.")

    parser.add_argument("--debug", action='store_true', default=False, help="Activate debug sections.")
    parser.add_argument("--tiss_rank", action='store_true', default=False, help="Rank tissues by how many genes identified.")
    args = parser.parse_args()

    Logging.configureLogging(args.verbosity)

    run(args)
