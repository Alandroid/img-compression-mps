#%%
import matplotlib.pyplot as plt
from my_analysis import Define_Bins_for_Compression_Factor, load_all_dicts
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

def plot_results(compression_factors):
    plots_data = []
    labels = ['Image size = (1024, 1024) (Padded)', 
              'Imafe size = (512, 512)',
              'Image size = (1024, 712) (Rectangular)',
              'Image size = (1024, 1024)']
    for key in compression_factors:
        print(key)
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
    plt.title("SSIM vs. Compression Factor with Error Bars")
    plt.legend()
    plt.grid(True)

    plt.show()  

def extract_compression_factor_dict(ssim_dict, compression_ratio):
    compression_factor_dict = defaultdict(list)

    for key in ssim_dict:
        for i in range (len(compression_ratio[key])):
            compression_factor = 1/compression_ratio[key][i]
            ssim = ssim_dict[key][i]

            compression_factor_dict[compression_factor].append(ssim)
    
    return compression_factor_dict

def load_1024_compression_factor_dict(path_in, batched_files):
    files = [path_in+i for i in  os.listdir(path_in) if i in batched_files]
    combined_psnr_dict, combined_compression_ratio_DCT, combined_compression_factor_dict = load_all_dicts(files)

    return combined_compression_factor_dict

def create_compression_factors_dict(path_in, batched_filenames, full_filenames):
    compression_factors = {}
    for filename in full_filenames:
        ssim_dict, compression_ratio = load_dicts_from_txt(path_in+filename)
        compression_factor = extract_compression_factor_dict(ssim_dict, compression_ratio)

        compression_factors[filename] = compression_factor

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
