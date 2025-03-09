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
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
D_path = D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/MRI Dataset'
#run_full_benchmark_3D(D_path, cutoff_list, 'results_dict_test.json')
cutoff_list = np.linspace(0, 0.1, 100)[1:]
run_full_benchmark(D_path, cutoff_list, 'MRI_5_to_10_100_steps_to_01_test.json', "MRI", 5, 10)

# %%
"""
This one is for MRI slices
"""
D_path = D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/MRI Dataset'
#run_full_benchmark_3D(D_path, cutoff_list, 'results_dict_test.json')
cutoff_list = np.linspace(0, 0.2, 100)[1:]
run_full_benchmark(D_path, cutoff_list, 'MRI_slice_0_to_13_100_steps_to_02.json', "MRI_Slice", 0, 13)

# %%
