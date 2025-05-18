"""
This is a notebook to run the benchmarks and evaluate the data at the same time
"""


#%%
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# Automatically find the project root (assumes "src" is in the project)
current_file = Path(__file__).resolve() if '__file__' in locals() else Path.cwd()
project_root = current_file
while not (project_root / "src").exists() and project_root != project_root.parent:
    project_root = project_root.parent

# Ensure we found the correct project root
if not (project_root / "src").exists():
    raise FileNotFoundError("Could not find project root containing 'src' directory.")

# Set the working directory to the project root
os.chdir(project_root)



from src.evaluation.benchmark import run_full_benchmark

#%%
"""
This one is for MRI images
"""
D_path = "Data/ds003799-2.0.0"
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
#run_full_benchmark_3D(D_path, cutoff_list, 'results_dict_test.json')
cutoff_list = np.linspace(0, 0.1, 10)[1:]
run_full_benchmark(D_path, cutoff_list, 'pedestrians_0_1_10steps_to_0.1_PSNR_test.json', "MRI", 'DCT', 0, 1, '.npz')

# %%
"""
This one is for MRI slices
"""
D_path = "Data/MRI Dataset"

cutoff_list = np.linspace(0, 0.3, 100)[1:]
run_full_benchmark(D_path, cutoff_list, 'MRI_slice_0_to_12_100_steps_to_03_Std.json', "MRI_Slice", "Std", 0, -1)

# %%
