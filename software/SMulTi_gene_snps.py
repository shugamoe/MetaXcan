import os
import logging 
import numpy
import pandas
from datetime import datetime
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
    if args.gene_whitelist:
        gene_whitelist = pandas.read_table(args.gene_whitelist[0], usecols=[args.gene_whitelist[1]])[args.gene_whitelist[1]].to_list()
    logging.info("Acquiring gwas-variant intersection")
    model, intersection = GWASAndModels.model_and_intersection_with_gwas_from_args(args)
    model_structure = JointAnalysis2.build_model_structure(model, intersection)

    logging.info("Acquiring groups")
    group_keys, groups = JointAnalysis2.get_group(args, set(model_structure.keys()))

    logging.info("Acquiring associations")
    associations = JointAnalysis2.get_associations(args)

    reporter = Utilities.PercentReporter(logging.INFO, len(group_keys))

    # Imputed SNPs don't have effect sizes or standard errors, but they
    # recorded the position window of the original SNPs, with effect sizes,
    # that were used to impute their zscores, go get those
    if args.get_og_for_imputed:
        logging.info(f"Loading COJO MA File | {datetime.now()}")
        cojo_ma_df = pandas.read_csv(args.get_og_for_imputed, sep=" ", usecols=['chromosome_position', 'effect_size'])
        # cojo_ma_df = cojo_ma_df[['chromosome_position', 'effect_size']]
        cojo_ma_df['position'] = cojo_ma_df['chromosome_position'].str.split("_", expand=True)[1].astype(int)
        cojo_ma_df['chr_num'] = cojo_ma_df["chromosome_position"].str.extract(r"(\d{1,2})").astype(int)

        logging.info(f"Loading Imputation Ranges | {datetime.now()}")
        snp_typed_imp_pos = KeyedDataSource.load_data_dual(args.gwas_file, "panel_variant_id", "typed_min_pos", 
                             "typed_max_pos", sep="\t", numeric=True) 
        logging.info(f"Imputation Ranges Loaded | {datetime.now()}")

    results = []
    for i, group_name in enumerate(group_keys):
        if args.gene_whitelist:
            if group_name not in gene_whitelist:
                continue
        reporter.update(i+1, "processed %d %% of groups")
        if args.MAX_M and  i>args.MAX_M-1:
            logging.info("Early abort")
            break

        logging.log(8, "Processing group for %s", group_name)
        try:
            ofile = None
            group = [x for x in groups[group_name] if x in associations]

            features, variants = JointAnalysis2.get_features_and_variants(group, model_structure)
            if not features or not variants:
                logging.info(f"{group_name} has no features or variants.")
                continue
            variants_ = sorted(variants)

            variants_no_tiss_ = [tiss_variant.split(".")[1] for tiss_variant in variants_]
            variants_no_tiss_split_ = [variant.split("_") for variant in variants_no_tiss_]
            variants_chr_pos_ = [f"{split[0]}_{split[1]}" for split in variants_no_tiss_split_]
            gene_df = pandas.DataFrame({"chromosome_position": variants_chr_pos_})
            gene_df = gene_df.drop_duplicates() # Remove SNPs that were used across tissues

            all_pred_ofile = "{}{}.snplist.all.for.prediction".format(args.snp_list_prefix, group_name)
            gene_df[['chromosome_position']].to_csv(all_pred_ofile, header=False, index=False)

            if args.get_og_for_imputed:
		# gene_df_imp_snps gives us the 'imputed' SNPs that we don't
		# have original data for (effect_size, standard error) COJO
		# needs this information to run, so instead, we go through each
		# gene and get the position windows for the original SNPs (that
		# have effect_size, standard error)
                gene_df_imp_snps = cojo_ma_df[cojo_ma_df['effect_size'].isna()].merge(gene_df, on="chromosome_position")
                gene_df_og_snps = cojo_ma_df.dropna().merge(gene_df, on="chromosome_position")
                
                if len(gene_df_imp_snps) == 0: # All SNPs for the introns for a gene are original SNPs
                    ofile = "{}{}.snplist".format(args.snp_list_prefix, group_name)
                    gene_df_og_snps[['chromosome_position']].to_csv(ofile, header=False, index=False)
                    results.append((group_name, ofile))
                    continue
                
		# Retrieve the original SNPs in the windows of original SNPs
		# used to impute the imputed SNPS
                var_chr_num =  int(gene_df_imp_snps['chr_num'][0])
                
                og_snps_for_imp_df = None 
                for imp_snp in gene_df_imp_snps['chromosome_position'].tolist():
                    full_name_index = variants_chr_pos_.index(imp_snp)
                    full_name = variants_no_tiss_[full_name_index]

                    min_og_snp_pos = snp_typed_imp_pos[full_name]["typed_min_pos"]
                    max_og_snp_pos = snp_typed_imp_pos[full_name]["typed_max_pos"]
                    
                    og_snps_for_imp = cojo_ma_df[cojo_ma_df['chr_num'] == var_chr_num].dropna()
                    og_snps_for_imp = og_snps_for_imp[og_snps_for_imp['position'].between(min_og_snp_pos, max_og_snp_pos)]
                    
                    if og_snps_for_imp_df is None:
                        og_snps_for_imp_df = og_snps_for_imp
                    else:
                        og_snps_for_imp_df = pandas.concat([og_snps_for_imp_df, og_snps_for_imp])
                        og_snps_for_imp_df = og_snps_for_imp_df.drop_duplicates() # Windows might overlap for different SNPs
		# Combine original SNPs from original variant list and original
		# SNPs needed for imputation of imputed snps
                all_og_snp_df = pandas.concat([og_snps_for_imp_df[['chromosome_position']], 
                                              gene_df_og_snps[['chromosome_position']]]) 
                all_og_snp_df = all_og_snp_df.drop_duplicates()
                ofile = "{}{}.snplist".format(args.snp_list_prefix, group_name)
                all_og_snp_df[['chromosome_position']].to_csv(ofile, header=False, index=False)
            else: 
                ofile = "{}{}.snplist".format(args.snp_list_prefix, group_name)
                gene_df[['chromosome_position']].to_csv(ofile, header=False, index=False)
        
        except Exception as e:
            #TODO: improve error tracking
            logging.log(8, "{} Exception: {}".format(group_name, str(e)))
            status = "Error"
            # results.append((group_name, ofile))
        else:
            results.append((group_name, ofile))
    
    out_cols = ["group", "snp_list_path"]
    results = pandas.DataFrame(results, columns=out_cols)

    logging.info("Saving results")
    Utilities.save_dataframe(results, args.output)

    end = timer()
    logging.info("Successfully wrote SNP lists for genes in %s seconds" % (str(end - start)))


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

    parser.add_argument("--associations", nargs="+", help="Entries association")
    parser.add_argument("--associations_folder", help="Name of folder containing entries association data.")
    parser.add_argument("--associations_file_pattern", help="Pattern to recognize associations files in folders (in case there are extra files and you don't want them selected).")
    parser.add_argument("--associations_tissue_pattern", help="Pattern to recognize the tissues corresponding to an association file")

    parser.add_argument("--grouping", nargs="+", help="File containing groups of features.", default = [])

    # parser.add_argument("--cutoff_sumrule", help="If True, keep eigenvalue j if sum from i=0 to i=j >= 95% of sum of eigenvalues.", default=None, type=float)
    # parser.add_argument("--cutoff_condition_number", help="condition number of eigen values to use when truncating SVD", default=None, type=float)
    # parser.add_argument("--cutoff_eigen_ratio", help="ratio to use when truncating SVD", default=None, type=float)
    # parser.add_argument("--cutoff_threshold", help="threshold of variance when truncating SVD", default=None, type=float)
    # parser.add_argument("--cutoff_trace_ratio", help="ratio to use when truncating SVD", default=None, type=float)
    # parser.add_argument("--regularization", help="Add a regularization term to correct for expression covariance matrix singularity", default=None, type=float)

    parser.add_argument("--MAX_M", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--snp_list_prefix")
    parser.add_argument("--verbosity", help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything", default=10, type=int)
    parser.add_argument("--tiss_list", help="File with list of tissues to filter by, one column only, no header. Tissues must match exactly the <tissue_name> in the associations file tissue/intron format <tissue_name>.<intron_id>.")
    parser.add_argument("--get_og_for_imputed", help="For SNPs that were imputed, write the SNP list of the original genes used to impute those SNPs. Input an .ma file for COJO, any row without na will be considered an 'original' gene.")
    parser.add_argument("--gene_whitelist", nargs="+", default = [], help="Path to tab separated file with genes to run, followed by the column name with ENSG* style gene names.")

    args = parser.parse_args()

    Logging.configureLogging(args.verbosity)

    run(args)
