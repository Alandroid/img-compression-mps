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
D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
#D_path = '/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/Data/fMRI_Datatset'
cutoff_list = np.linspace(0, 0.1, 10)[1:]
#run_full_benchmark_3D(D_path, cutoff_list, 'results_dict_test.json')
run_full_benchmark(D_path, cutoff_list, 'results_fMRI_extended_json_test.json', "fMRI", 0, 1)

# %%

def calc_mean_std(dict):
    ssims = np.array(dict["ssim_list"])
    comps = np.array(dict["compressionratio_list"])
    max_common_fac = np.min(1/comps[:,-1])
    min_common_fac = np.max(1/comps[:,0])
    common_comp_facs = np.linspace(min_common_fac, max_common_fac, 20)
    interpolated_ssim = []
    for x, y in zip(1/comps, ssims):
        interp_func = interp1d(x, y, kind="linear", bounds_error=False)
        interpolated_ssim.append(interp_func(common_comp_facs))
    return np.mean(interpolated_ssim, axis = 0), np.std(interpolated_ssim, axis = 0), np.array(common_comp_facs)
#%%
with open("/Users/maxge/Documents/Studium/München/02_SS 2024/QEL/Block encoding generalization/img-compression-mps/src/evaluation/results/results_dict_MRI_Slice_1.json", "r") as f:
    results_Slice = json.load(f)

# %%
slice_mean, slice_std, slice_comp_facs = calc_mean_std(results_Slice)
plt.errorbar(slice_comp_facs, slice_mean, yerr = slice_std, label = "MRI Slice")
# %%

with open("results/results_dict_MRI_2.json", "r") as f:
    results_MRI = json.load(f)
# %%
MRI_mean, MRI_std, MRI_comp_facs = calc_mean_std(results_MRI)
plt.errorbar(MRI_comp_facs, MRI_mean, yerr = MRI_std, label = "MRI")
plt.errorbar(slice_comp_facs, slice_mean, yerr = slice_std, label = "MRI Slice")
plt.legend()
# %%
with open("results/results_fMRI_test.json", "r") as f:
    results_fMRI = json.load(f)
# %%
fMRI_mean, fMRI_std, fMRI_comp_facs = calc_mean_std(results_fMRI)
plt.errorbar(fMRI_comp_facs, fMRI_mean, yerr = fMRI_std, label = "fMRI")
plt.errorbar(MRI_comp_facs, MRI_mean, yerr = MRI_std, label = "MRI")
plt.errorbar(slice_comp_facs, slice_mean, yerr = slice_std, label = "MRI Slice")
plt.legend()
# %%
