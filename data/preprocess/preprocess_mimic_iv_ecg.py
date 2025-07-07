import pandas as pd
import numpy as np
import os
from signal_analysis import calculate_waveforms
from tqdm import tqdm
import wfdb
import argparse

def prepare(args):
    data_dir = args.data_dir
    df = pd.read_csv(os.path.join(data_dir, "record_list.csv"))
    
    missing_path_file = os.path.join(data_dir, "missing_path.txt")
    if os.path.exists(missing_path_file):
        with open(missing_path_file, "r") as f:
            missing_paths= [line.strip() for line in f if line.strip()]
        df = df[~df["path"].isin(missing_paths)]
        print(f"Filtered out {len(missing_paths)} records with missing paths.")
    else:
        print("No missing path file found, proceeding with all records.")
    
    print("Start calculating waveform data...")
    data_dict = calculate_waveforms(data_dir, df.path.values)
    
    for key in data_dict.keys():
        df[key] = data_dict[key]
        df.to_csv(os.path.join(data_dir, "new_record_list.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    args = parser.parse_args()

    prepare(args)