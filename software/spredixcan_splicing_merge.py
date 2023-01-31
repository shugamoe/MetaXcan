import os
import pandas as pd
import glob


def get_tiss_name(path):
    fname = os.path.basename(path) 
    parts = fname.split("__")
    tname_and_csv = parts[2]
    tname = tname_and_csv.split(".")[0]
    return tname
 

def combine_files(input_list, output_name):
    GOT_BASE = False
    
    for p_group in input_list:
        tissue = get_tiss_name(p_group)
        df = pd.read_csv(p_group)
        df['gene'] = "{}.".format(tissue) + df['gene']
        df['gene_name'] = "{}.".format(tissue) + df['gene_name']
        # IntronXCan wants missing zscore values to be "NA"
        df['zscore'] = df['zscore'].fillna('NA') 
     
        if not GOT_BASE:
            base_df = df
            GOT_BASE = True
        else:
            base_df = base_df.append(df)
    
    base_df.to_csv(output_name, sep=",", index=False)

    return None

def run(args):
    if os.path.exists(args.output):
        print("Output exists. Remove it or delete it.")
        return

    input_list = glob.glob(args.spred_sqtl_pattern)
    combine_files(input_list, args.output)
    
if __name__ == "__main__":
    import  argparse
    parser = argparse.ArgumentParser("Combine the sqtl associations of a study across all relevant tissues.")
    parser.add_argument("--spred_sqtl_pattern", help="Pattern to match study's sqtl files")
    parser.add_argument("--output", help="Name of the output")
    args = parser.parse_args()
    run(args)
