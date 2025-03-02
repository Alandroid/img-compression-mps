import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
notebook_dir = Path.cwd()
src_folder = str(notebook_dir.parents[0])
sys.path.append(src_folder)

from compression.utils_ND import calc_mean_std


# Load MRI Slice results
with open("results/results_dict_MRI_SLICE_1.json", "r") as f:
    results_slice = json.load(f)

slice_mean, slice_std, slice_comp_facs = calc_mean_std(results_slice)
plt.errorbar(slice_comp_facs, slice_mean, yerr=slice_std, label="MRI Slice")

# Load MRI results
with open("results/results_dict_MRI_2.json", "r") as f:
    results_mri = json.load(f)

mri_mean, mri_std, mri_comp_facs = calc_mean_std(results_mri)
plt.errorbar(mri_comp_facs, mri_mean, yerr=mri_std, label="MRI")
plt.errorbar(slice_comp_facs, slice_mean, yerr=slice_std, label="MRI Slice")
plt.legend()

# Load fMRI results
with open("results/results_fMRI_test.json", "r") as f:
    results_fmri = json.load(f)

fmri_mean, fmri_std, fmri_comp_facs = calc_mean_std(results_fmri)
plt.errorbar(fmri_comp_facs, fmri_mean, yerr=fmri_std, label="fMRI")
plt.errorbar(mri_comp_facs, mri_mean, yerr=mri_std, label="MRI")
plt.errorbar(slice_comp_facs, slice_mean, yerr=slice_std, label="MRI Slice")
plt.legend()
plt.xlabel("Compression Factor")
plt.ylabel("SSIM")
plt.title("SSIM vs Compression Factor")
plt.show()
