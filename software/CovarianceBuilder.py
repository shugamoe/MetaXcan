#! /usr/bin/env python
import os
import logging
import pandas

from timeit import default_timer as timer

from metax import __version__
from metax import Logging
from metax import Exceptions
from metax import PredictionModel
from metax import Utilities
from metax import MatrixManager
from metax.genotype import  Utilities as GenotypeUtilities
from metax.genotype import GenotypeAnalysis

def run(args):
    if os.path.exists(args.snp_covariance_output):
        logging.info("%s already exists, you have to move it or delete it if you want it done again", args.snp_covariance_output)
        return

    start = timer()

    logging.info("Loading models...")
    model_manager = PredictionModel.load_model_manager(args.models_folder, name_pattern=args.models_pattern)
    all_snps = model_manager.get_rsids()

    logging.info("processing genotype")
    for chromosome, metadata, dosage in GenotypeUtilities.genotype_by_chromosome_from_args(args, all_snps):
        logging.log(9, "Processing chromosome %s", str(chromosome))
        covariance_results = pandas.DataFrame()

        context = GenotypeAnalysis.GenotypeAnalysisContext(metadata, dosage, model_manager)
        genes = context.get_genes()
        reporter = Utilities.PercentReporter(9, len(genes))
        reporter.update(0, "%d %% of genes processed so far in chromosome " + str(chromosome))
        for i,gene in enumerate(genes):
            logging.log(7, "%d/%d:%s", i+1, len(genes), gene)
            cov_data = GenotypeAnalysis.get_prediction_covariance(context, gene)
            cov_data = MatrixManager._flatten_matrix_data([cov_data])
            cov_data = Utilities.to_dataframe(cov_data, GenotypeAnalysis.COVARIANCE_COLUMNS, to_numeric="ignore", fill_na="NA")
            covariance_results = pandas.concat([covariance_results, cov_data])

            reporter.update(i, "%d %% of genes processed so far in chromosome "+str(chromosome))

        reporter.update(len(genes), "%d %% of genes processed so far in chromosome " + str(chromosome))

        logging.log(9, "writing chromosome results")
        Utilities.save_dataframe(covariance_results, args.snp_covariance_output,
                                    mode="w" if chromosome ==1 else "a",
                                    header=chromosome==1)

    end = timer()
    logging.info("Ran covariance builder in %s seconds" % (str(end - start)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CovarianceBuilder.py %s:'
        'Collect and process covariance of genotypes' % (__version__))

    parser.add_argument("--models_folder", help="Path to folder with prediction models")
    parser.add_argument("--models_pattern", help="Regexp to extarct models with")
    parser.add_argument("--gtex_genotype_file", help="Path to gtex genotype file")
    parser.add_argument("--gtex_snp_file", help="Path to snp annotation file")
    parser.add_argument("--gtex_release_version", help="none(which is v6p) or V8")
    parser.add_argument("--dosage_genotype_folder", help="Path to dosage folder")
    parser.add_argument("--dosage_genotype_pattern", help="Regexp-like pattern to select files")
    parser.add_argument("--snp_covariance_output", help="where you want the output", default=None)
    parser.add_argument("--verbosity", help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything", default = "10")
    parser.add_argument("--throw", action="store_true", help="Throw exception on error", default=False)

    args = parser.parse_args()

    Logging.configureLogging(int(args.verbosity))

    if args.throw:
        run(args)
    else:
        try:
            run(args)
        except Exceptions.ReportableException as e:
            logging.error(e.msg)
        except Exception as e:
            logging.info("Unexpected error: %s" % str(e))