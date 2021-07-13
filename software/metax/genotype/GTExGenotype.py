import gzip
import logging
import io

import numpy

from .Genotype import GF
from .. import Utilities
from . import  Helpers


def parse_gtex_variant(variant):
    comps = variant.split("_")
    return comps[0:4]

def gtex_geno_header(gtex_file):
    with io.TextIOWrapper(gzip.open(gtex_file, "r"), newline="") as file:
        header = file.readline().strip().split()
    return header

def gtex_geno_lines(gtex_file, gtex_snp_file, snps=None, gtex_release_version=None, impute_to_mean=False,
                    member_file=None):

    logging.log(9, "Loading GTEx snp file")
    #TODO: change to something more flexible to support V7 naming

    gtex_snp = Helpers.gtex_snp(gtex_snp_file, gtex_release_version)

    logging.log(9, "Processing GTEx geno")
    with io.TextIOWrapper(gzip.open(gtex_file, "r"), newline="") as file:
            
        for i,line in enumerate(file):
            if i==0:
                if member_file is not None:
                    import pandas as pd
                    # Always want varID column from header
                    col_list = ['varID'] + list(pd.read_csv(member_file, sep="\t")['indv_id']) 
                else:
                    col_list = line.strip().split()

                def filter_members(gtex_id):
                    if gtex_id in col_list:
                        return True
                    else:
                        return False

                header = line.strip().split()
                col_mask = list(map(filter_members, header))

                og_num_members = len(header) - 1 
                cur_num_members = sum(col_mask) - 1
                
                if og_num_members != cur_num_members: 
                    logging.log(9, "Filtered # of members from {} -> {} ".format(og_num_members, cur_num_members))
                continue #skip header. This line is not needed but for conceptual ease of mind

            comps = numpy.array(line.strip().split())[col_mask].tolist()
            assert len(comps) == cur_num_members + 1
            variant = comps[0]

            if not variant in gtex_snp:
                continue

            rsid = gtex_snp[variant]
            if snps and not rsid in snps:
                continue

            data = parse_gtex_variant(variant)
            if impute_to_mean:
                _d = numpy.array([x for x in comps[1:] if x != "NA"], dtype=numpy.float64)
                _m = numpy.mean(_d)
                dosage = numpy.array([x if x != "NA" else _m for x in comps[1:]], dtype=numpy.float64)
                frequency = _m/2
            else:
                dosage = numpy.array(comps[1:], dtype=numpy.float64)
                frequency = numpy.mean(dosage)/2

            yield [rsid] + data + [frequency] + list(dosage)

def gtex_geno_by_chromosome(gtex_file, gtex_snp_file, snps=None, gtex_release_version=None, impute_to_mean=False,member_file=None):
    buffer = []
    last_chr = None

    def _buffer_to_data(buffer):
        _data = list(zip(*buffer))

        _metadata = _data[0:GF.FIRST_DOSAGE]
        rsids = _metadata[GF.RSID]

        chr_ = _metadata[GF.CHROMOSOME][0].replace("chr", "")
        chromosome = int(chr_) # I know I am gonna regret this cast UPDATE: I did regret it!
        _metadata = list(zip(*_metadata))
        metadata = Utilities.to_dataframe(_metadata, ["rsid", "chromosome", "position", "ref_allele", "alt_allele", "frequency"], to_numeric="ignore")

        dosage = list(zip(*_data[GF.FIRST_DOSAGE:]))
        dosage_data = {}
        #TODO: 142_snps are not unique, several rows go into the same value, improve this case handling
        for i in range(0, len(rsids)):
            rsid = rsids[i]
            if rsid in dosage_data:
                ind = metadata[metadata.rsid == rsid].index.tolist()[0]
                metadata = metadata.drop(ind)

            dosage_data[rsid] = dosage[i]
        metadata["number"] = list(range(0, len(metadata)))
        metadata = metadata.set_index("number")

        return chromosome, metadata, dosage_data

    logging.log(8, "Starting to process lines")
    for line in gtex_geno_lines(gtex_file, gtex_snp_file, snps, gtex_release_version, impute_to_mean, member_file):

        chromosome = line[GF.CHROMOSOME]
        if last_chr is None: last_chr = chromosome

        if last_chr != chromosome:
            yield _buffer_to_data(buffer)
            buffer = []

        last_chr = chromosome
        buffer.append(line)

    yield _buffer_to_data(buffer)
