import os
import sys

project_root = "/scratch/m/M.Theocharakis/paper/img-compression-mps/"
src = os.path.join(project_root, "src")
sys.path.insert(0, src)

# from src.compression.mps_ND import NDMPS
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from copy import deepcopy
import argparse
from benchmark import run_full_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
parser.add_argument("--type", type=str, required=True)
args = parser.parse_args()

start = args.start
end = args.end
tpe = args.type



filename = f'ds003799_{start}_{end}_100_100steps_to_0p1_{tpe}_ses3.json'
cutoff_list = np.linspace(0, 0.1, 100)[1:]

run_full_benchmark("/scratch/m/M.Theocharakis/paper/img-compression-mps/Data/ses-3", 
    cutoff_list, filename, "MRI", tpe, start, end, ".gz", shape = None)
