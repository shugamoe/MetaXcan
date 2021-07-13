import os
import logging
import numpy
import pandas
from scipy import stats
from timeit import default_timer as timer

from metax import Utilities, Logging
from metax.cross_model import JointAnalysis2
from metax.misc import GWASAndModels, Math
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

    reporter = Utilities.PercentReporter(logging.INFO, len(group_keys))

    results = []
    for i,group_name in enumerate(group_keys):
        reporter.update(i+1, "processed %d %% of groups")
        if args.MAX_M and  i>args.MAX_M-1:
            logging.info("Early abort")
            break

        logging.log(8, "Processing group for %s", group_name)
        group_name, chi2_p, n_variants, n_features, n_indep, tmi, status  = group_name, None, None, None, None, None, None
        try:
            group = [x for x in groups[group_name] if x in associations]

            features, variants = JointAnalysis2.get_features_and_variants(group, model_structure)
            if not features or not variants:
                continue
            n_features, n_variants = len(features), len(variants)
            features_, variants_, transcriptome_correlation = JointAnalysis2.get_transcriptome_correlation(covariance_source, model_structure, features, variants)

            zscores = numpy.array([associations[x] for x in features_], dtype=numpy.float64)
            cutoff_ = cutoff(transcriptome_correlation)
            inv, n_indep, eigen = Math.capinv(transcriptome_correlation, cutoff_, args.regularization)

            w = float(numpy.dot(numpy.dot(zscores, inv), zscores))
            chi2_p = stats.chi2.sf(w, n_indep)
            tmi = numpy.trace(numpy.dot(transcriptome_correlation, inv))
        except:
            #TODO: improve error tracking
            status = "Error"
        results.append((group_name, chi2_p, n_variants, n_features, n_indep, tmi, status))
    results = pandas.DataFrame(results, columns=["group", "pvalue", "n_variants", "n_features", "n_indep", "tmi", "status"])
    results = results.sort_values(by="pvalue")
    logging.info("Saving results")
    Utilities.save_dataframe(results, args.output)

    end = timer()
    logging.info("Successfully processed grouped SMultiXcan in %s seconds" % (str(end - start)))

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
    parser.add_argument("--grouping", nargs="+", help="File containing groups of features.", default = [])

    parser.add_argument("--cutoff_condition_number", help="condition number of eigen values to use when truncating SVD", default=None, type=float)
    parser.add_argument("--cutoff_eigen_ratio", help="ratio to use when truncating SVD", default=None, type=float)
    parser.add_argument("--cutoff_threshold", help="threshold of variance when truncating SVD", default=None, type=float)
    parser.add_argument("--cutoff_trace_ratio", help="ratio to use when truncating SVD", default=None, type=float)
    parser.add_argument("--regularization", help="Add a regularization term to correct for expression covariance matrix singularity", default=None, type=float)

    parser.add_argument("--MAX_M", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--verbosity", help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything", default=10, type=int)

    args = parser.parse_args()

    Logging.configureLogging(args.verbosity)

    run(args)
