#%%
import matplotlib.pyplot as plt
from my_analysis import Define_Bins_for_Compression_Factor, load_all_dicts, bootstrap_error_bars
import json
from collections import defaultdict
import numpy as np
import os
# %%
def locate_data(path_in):
    batched_files = []
    full_files = []
    files = os.listdir(path_in)
    for file in files:
        if 'batch' in file:
            batched_files.append(file)
        else:
            full_files.append(file)
    
    return batched_files, full_files

def load_dicts_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        psnr_dict = data["psnr_dict"]
        compression_ratio_DCT = data["compression_ratio_DCT"]
    return psnr_dict, compression_ratio_DCT

def MY_Define_Bins_for_Compression_Factor(compression_factor_dict,
                                       error_method="bootstrap",
                                       limit_xaxis =40, 
                                       bins = 50,
                                       n_bootstrap=1000):
    
    bin_edges = np.geomspace(1e-2, limit_xaxis + 1, bins + 1) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    binned_ssim = defaultdict(list)

    for cf, ssim_values in compression_factor_dict.items():
        try:
            # Convert compression factor key to float
            cf = float(cf) * 1024*768/(1024 * 1024)
        except ValueError:
            # Skip invalid or non-numeric compression factor keys
            print(f"Skipping invalid compression factor: {cf}")
            continue

        bin_index = None  # Initialize bin index as None

        # Iterate through bin edges to find the correct bin for cf
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= cf < bin_edges[i + 1]:
                bin_index = i
                break  # Exit loop once the bin is found

        # If a valid bin is found, add the SSIM values to the corresponding bin
        if bin_index is not None and 0 <= bin_index < bins:
            binned_ssim[bin_index].extend(ssim_values)

    mean_ssim, lower_ci, upper_ci = [], [], []
    for b in range(bins):
        ssim_values = binned_ssim[b]
        if ssim_values:
            if error_method == "bootstrap":
                mean, lower, upper = bootstrap_error_bars(np.array(ssim_values), n_bootstrap=n_bootstrap)
            # else:
            #     mean, lower, upper = gaussian_error_bars(np.array(ssim_values))
            mean_ssim.append(mean)
            lower_ci.append(lower)
            upper_ci.append(upper)
        else:
            mean_ssim.append(np.nan)
            lower_ci.append(np.nan)
            upper_ci.append(np.nan)

    return bin_centers, mean_ssim, lower_ci, upper_ci

def plot_results(compression_factors):
    plots_data = []
    labels = ['Image size = (1024, 1024) (Padded)', 
              'Imafe size = (512, 512)',
              'Image size = (1024, 712) (Rectangular)',
              'Image size = (1024, 1024)']
    for key in compression_factors:
        if key == '1024':
            bin_centers, mean_ssim, lower_ci, upper_ci = MY_Define_Bins_for_Compression_Factor(compression_factors[key])
        else:
            bin_centers, mean_ssim, lower_ci, upper_ci = Define_Bins_for_Compression_Factor(compression_factors[key])

        plots_data.append((bin_centers, mean_ssim, lower_ci, upper_ci))

    plt.figure(figsize=(10, 5))  

    for i, (bin_centers, mean_ssim, lower_ci, upper_ci) in enumerate(plots_data):
        plt.scatter(bin_centers, mean_ssim, s=8, label=labels[i])
        plt.errorbar(
            bin_centers, mean_ssim,
            yerr=[np.array(mean_ssim) - np.array(lower_ci), np.array(upper_ci) - np.array(mean_ssim)],
            fmt='.', ecolor='red', capsize=2
        )

    plt.xlabel("Compression Factor (1 / Compression Ratio)")
    plt.ylabel("SSIM")
    plt.xlim(0, 16)
    plt.title("SSIM vs. Compression Factor with Error Bars")
    plt.legend()
    plt.grid(True)

    plt.show()  

def extract_compression_factor_dict(ssim_dict, compression_ratio, call):
    compression_factor_dict = defaultdict(list)

    rescale_list = [1024*1024/(1024*768), 512*512/(1024*768), 1] # quick fix for comperable results
                                                                # We should fix the calculation of the
                                                                # compression ratio in the class
    for key in ssim_dict:
        for i in range (len(compression_ratio[key])):
            cr_rescaled = compression_ratio[key][i] * rescale_list[call]

            compression_factor = 1/cr_rescaled
            ssim = ssim_dict[key][i]

            compression_factor_dict[compression_factor].append(ssim)
    
    return compression_factor_dict

def load_1024_compression_factor_dict(path_in, batched_files):
    files = [path_in+i for i in  os.listdir(path_in) if i in batched_files]
    combined_psnr_dict, combined_compression_ratio_DCT, combined_compression_factor_dict = load_all_dicts(files)

    return combined_compression_factor_dict

def create_compression_factors_dict(path_in, batched_filenames, full_filenames):
    compression_factors = {}
    call = 0
    for filename in full_filenames:
        print(filename)
        ssim_dict, compression_ratio = load_dicts_from_txt(path_in+filename)
        compression_factor = extract_compression_factor_dict(ssim_dict, compression_ratio, call)

        compression_factors[filename] = compression_factor
        call += 1

    compression_factors['1024'] = load_1024_compression_factor_dict(path_in, batched_filenames)

    return compression_factors

def main():
    path_in = './data/'
    batched_files, full_files = locate_data(path_in)

    compression_factors = create_compression_factors_dict(path_in=path_in,
                                                        batched_filenames=batched_files,
                                                        full_filenames=full_files)

    plot_results(compression_factors)

if __name__ == '__main__':
    main()

# %%
