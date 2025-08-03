import pandas as pd
import numpy as np
import os
from signal_analysis import calculate_waveforms
from tqdm import tqdm
import argparse


def prepare(args):
    data_dir = args.data_dir
    df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"))

    files = df.filename.lr_values
    existing = []
    missing = []
    for fn in files:
        hea = os.path.join(data_dir, fn + ".hea")
        if os.path.exists(hea):
            existing.append(fn)
        else:
            missing.append(fn)
    

    if missing:
        print(f"Warning: skipp {len(missing)} missing records:", missing)
    
    print("Start calculating waveform data...")
    data_dict = calculate_waveforms(data_dir, existing)

    for key, vals in data_dict.items():
        col = [float("nan")] * len(df)
        for i, fn in enumerate(df.filename_lr):
            if fn in existing:
                col[i] = data_dict[key][existing.index(fn)]♥
        
        df[key] = col
    
    df.to_csv(os.path.join, "ptbxl_database.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    args = parser.parse_args()

    prepare(args)