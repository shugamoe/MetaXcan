import os
import gzip
import io
import pandas
import numpy as np
import re
# /gpfs/data/gao-lab/Julian/software/MetaXcan/software/metax/cross_model/Utilities.py

from .. import Utilities
from .. import Utilities as BUtilities

def try_parse(string, fail=None, float128=False):
    if not float128:
        try:
            return float(string)
        except Exception:
            return fail;
    else:
        try:
            return np.float128(string)
        except Exception:
            return fail;

def skip_na(key, value):
    skip = (not value or value == "NA")
    return skip

def skip_non_rsid_value(key, value):
    return not "rs" in value

def dot_to_na(value):
    return "NA" if value == "." else value

def load_data(path, key_name, value_name, white_list=None, value_white_list=None, numeric=False, should_skip=None, value_conversion=None, key_filter=None, sep=None, float128=False):
    data = {}
    c_key=None
    c_value=None
    for i, line in enumerate(Utilities.generate_from_any_plain_file(path)):
        if i==0:
            header = line.strip().split(sep)
            c_key = header.index(key_name)
            c_value = header.index(value_name)
            continue

        comps = line.strip().split(sep)
        key = comps[c_key]
        if white_list and not key in white_list:
            continue
        if key_filter and key_filter(key):
            continue

        value = comps[c_value]
        if value_white_list and not value in value_white_list:
            continue

        if should_skip and should_skip(key, value):
            continue

        if value_conversion:
            value = value_conversion(value)
        elif numeric:
            value = try_parse(value)
        elif float128:
            value = try_parse(value, float128=True)

        if value != None:
            data[key] = value

    return data

def load_data_from_assoc(args, key_name, value_name,
        white_list=None, value_white_list=None, numeric=True,
        should_skip=None, value_conversion=None, key_filter=None, sep=None,
        float128=False):
    if args.tiss_list is not None:
        tiss_list = pandas.read_csv(args.tiss_list, header=None)[0].tolist()
        print("Tissue list.")
        print(tiss_list)
    if args.associations:
        return(load_data(args.associations[0], key_name, value_name, sep=sep, numeric=numeric))

    file_regexp = re.compile(args.associations_file_pattern) if args.associations_file_pattern else  None
    if args.associations_tissue_pattern:
        tissue_regexp = re.compile(args.associations_tissue_pattern)
    else:
        raise Exceptions.InvalidArguments("Need a way to identify which association file is with which tissue")

    data = {}
    files = BUtilities.contentsWithRegexpFromFolder(args.associations_folder, file_regexp)
    for assoc_file in files:
        tiss = tissue_regexp.search(os.path.basename(assoc_file)).groups()[0]
        if args.tiss_list is not None:
            if tiss not in tiss_list:
                continue
        lines = Utilities.generate_from_any_plain_file(os.path.join(args.associations_folder, assoc_file))
        for i, line in enumerate(lines):
            if i==0:
                header = line.strip().split(sep)
                c_key = header.index(key_name)
                c_value = header.index(value_name)
                continue

            comps = line.strip().split(sep)
            key = comps[c_key]
            if white_list and not key in white_list:
                continue
            if key_filter and key_filter(key):
                continue

            value = comps[c_value]
            if value_white_list and not value in value_white_list:
                continue

            if should_skip and should_skip(key, value):
                continue

            if value_conversion:
                value = value_conversion(value)
            elif numeric:
                value = try_parse(value)
            elif float128:
                value = try_parse(value, float128=True)

            if value != None:
                data["{}.{}".format(tiss, key)] = value
    return data

# Rushed modification of load_data to get two columns at a time in one pass
def load_data_dual(path, key_name, value_1_name, value_2_name, white_list=None, value_white_list=None, numeric=False, should_skip=None, value_conversion=None, key_filter=None, sep=None, float128=False):
    data = {}
    c_key=None
    c_value_1=None
    c_value_2=None
    for i, line in enumerate(Utilities.generate_from_any_plain_file(path)):
        if i==0:
            header = line.strip().split(sep)
            c_key = header.index(key_name)
            c_value_1 = header.index(value_1_name)
            c_value_2 = header.index(value_2_name)
            continue

        comps = line.strip().split(sep)
        key = comps[c_key]
        if white_list and not key in white_list:
            continue
        if key_filter and key_filter(key):
            continue

        value_1 = comps[c_value_1]
        value_2 = comps[c_value_2]

        if value_conversion:
            value_1 = value_conversion(value_1)
            value_2 = value_conversion(value_2)
        elif numeric:
            value_1 = try_parse(value_1)
            value_2 = try_parse(value_2)
        elif float128:
            value_1 = try_parse(value_1, float128=True)
            value_2 = try_parse(value_2, float128=True)

        if value_1:
            data[key] = {value_1_name: value_1, value_2_name: value_2}

    return data

def load_data_column(path, column_name):
    def _ogz(p):
        return  io.TextIOWrapper(gzip.open(p, "r"), newline="")
    _o = _ogz if ".gz" in path else open
    data = []
    c_key=None
    with _o(path) as file:
        for i, line in enumerate(file):
            if i==0:
                header = line.strip().split()
                c_key = header.index(column_name)
                continue

            comps = line.strip().split()
            value = comps[c_key]
            data.append(value)
    return data

def to_data_frame(data, keys, key_column, value_column, to_numeric=None):
    ids = [x for x in keys if x in data]
    data = [(x, data[x]) for x in ids]
    if len(data) == 0:
        return pandas.DataFrame({key_column: [], value_column: []})
    data = Utilities.to_dataframe(data, [key_column, value_column], to_numeric)
    return data
