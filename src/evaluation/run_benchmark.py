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
This one is for Video images
"""
D_path = "Data/pedestrians"
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'

cutoff_list = np.linspace(0, 0.1, 100)[1:]
run_full_benchmark(D_path, cutoff_list, 'Pedestrians_0_10_100steps_to_0p1_Std.json', "Video", "Std", 0, 10, ".npz", shape = (200,144,216))

# %%
"""
This one is for MRI images
"""
D_path = "Data/MRI Dataset"
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'

cutoff_list = np.linspace(0, 0.1, 100)[1:]
run_full_benchmark(D_path, cutoff_list, 'ds000003_0_13_100steps_to_0p1_Std.json', "MRI", "Std", 0, -1, ".gz", shape = None)
#%%

"""
This one is for fMRI images
"""
D_path = "Data/fMRI_Datatset"
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'

cutoff_list = np.linspace(0, 0.1, 100)[1:]
run_full_benchmark(D_path, cutoff_list, 'fMRI_0_5_100steps_to_0p1_Std.json', "fMRI", "Std", 0, 5, ".gz", shape = None)
#%%
"""
This one is for MRI slices
"""
D_path = "Data/ds003799-2.0.0"

cutoff_list = np.linspace(0, 0.3, 100)[1:]
run_full_benchmark(D_path, cutoff_list, 'MRI_slice_ds003799_0_to_50_100_steps_to_03_Std.json', "MRI_Slice", "Std", 0, 50, ".gz", shape= None)

# %%
