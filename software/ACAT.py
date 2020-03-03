import os
import logging
import math

import numpy
import pandas

from metax import Logging, Utilities
from metax.misc import KeyedDataSource
from metax.cross_model import JointAnalysis2

def run(args):
    if os.path.exists(args.output):
        logging.info("Output exists. Remove it or delete it.")
        return

    logging.info("Loading associations")
    associations = KeyedDataSource.load_data(args.associations[0], "gene", "pvalue", sep=",")

    logging.info("Loading groups")
    group_keys, groups = JointAnalysis2.get_group(args)

    r = []
    logging.info("Processing")
    for key in group_keys:
        group = [x for x in groups[key] if x in associations]
        p = numpy.array([associations[x] for x in group if associations[x] != "NA" and not numpy.isnan(associations[x])], dtype=numpy.float)
        if len(p) == 0:
            continue
        s = numpy.sum(numpy.tan((0.5 - p) * numpy.pi))
        acat = 0.5 - math.atan(s/p.shape[0])/numpy.pi
        r.append((key, acat, len(group), len(p)))

    r = pandas.DataFrame(r, columns=["group", "acat", "group_size", "available_results"])
    r = r.sort_values(by="acat")
    logging.info("Saving results")
    Utilities.save_dataframe(r, args.output)

    logging.info("Successfully processed acat")

if __name__ == "__main__":
    import  argparse
    parser = argparse.ArgumentParser("Compute Acat on multiple features, given a grouping (i.e. MulTiXcan on all splicing events -features-  of a given gene -group-)")

    parser.add_argument("--associations", nargs="+", help="Entries association")
    parser.add_argument("--grouping", nargs="+", help="File containing groups of features.", default = [])

    parser.add_argument("--output")
    parser.add_argument("--verbosity", help="Log verbosity level. 1 is everything being logged. 10 is only high level messages, above 10 will hardly log anything", default=10, type=int)

    args = parser.parse_args()

    Logging.configureLogging(args.verbosity)

    run(args)