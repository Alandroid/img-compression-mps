#%%
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
with open("results/results_dict_MRI_SLICE_1.json", "r") as f:
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
