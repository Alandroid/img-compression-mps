#%%
from mps import BWMPS
from utils import *
import os
from collections import defaultdict
from PIL import Image
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#%%

def get_batches(path_in: str):
    images = os.listdir(path_in)
    images.sort()
    batches = defaultdict(list)
    batch_size = 100

    for i, image in enumerate(images):
        key = i // batch_size + 1  # Compute the key based on batch size
        batches[key].append(image)

    return batches

def process_batch(path_in, batch, new_size = (256*3, 256*4)):
    mps_list = []
    for i, filename in enumerate(batch):
        print(f"Making mps list of image {i}")
        file = os.path.join(path_in, filename)
        
        with Image.open(file) as img:
            img_array = resize_image(img, new_size)
        
        mps = BWMPS.from_matrix(img_array, norm = False ,mode = 'DCT')
        mps_list.append([mps, img_array/img_array.max()])

    return mps_list

def compress_mps(mps_list, cutoff_list, limit_xaxis = 40):
    compression_ratio_DCT = defaultdict(dict)
    ssim_dict = defaultdict(dict)
    compression_factor_dict = defaultdict(list)

    for i, (mps, img) in enumerate(mps_list):
        print(f"Working on image{i}")
        compression_ratio_DCT[f'Image{i}'] = []
        ssim_dict[f'Image{i}'] = []

        for cutoff in (cutoff_list):
            mps.compress(cutoff)
            final_matrix = mps.mps_to_matrix()
            compression_ratio = mps.compression_ratio()


            compression_ratio_DCT[f'Image{i}'].append(compression_ratio)

            compression_factor = 1 / compression_ratio
            ssim_value = output_image_quality(img, final_matrix, metric="ssim")


            ssim_dict[f'Image{i}'].append(ssim_value)
            compression_factor_dict[compression_factor].append(ssim_value)

            if compression_factor > limit_xaxis:
                break
        
        print("."*50)
        
    
    return compression_ratio_DCT, ssim_dict, compression_factor_dict

def save_dicts_to_txt(compression_ratio_dict, psnr_dict, compression_factor_dict, batch,
                      path='/home/myron/data/'):
    
    file_name = f'dicts_data{batch}.txt'
    with open(path+file_name, 'w') as file:
        data = {
            "psnr_dict": psnr_dict,
            "compression_ratio_DCT": compression_ratio_dict,
            "compression_factor_dict": compression_factor_dict
        }
        json.dump(data, file)

def load_dicts_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        psnr_dict = data["psnr_dict"]
        compression_ratio_DCT = data["compression_ratio_DCT"]
        compression_factor_dict = data["compression_factor_dict"]
    return psnr_dict, compression_ratio_DCT, compression_factor_dict

def load_all_dicts(file_paths):
    combined_psnr_dict = {}
    combined_compression_ratio_DCT = {}
    combined_compression_factor_dict = {}
    
    for file_path in file_paths:
        psnr_dict, compression_ratio_DCT, compression_factor_dict = load_dicts_from_txt(file_path)
        combined_psnr_dict.update(psnr_dict)
        combined_compression_ratio_DCT.update(compression_ratio_DCT)
        combined_compression_factor_dict.update(compression_factor_dict)
    
    return combined_psnr_dict, combined_compression_ratio_DCT, combined_compression_factor_dict

def Create_Data():
    path_in = '/home/myron/DIV2K_BW'
    batches = get_batches(path_in)
    cutoff_list = np.arange(0, 4, 0.005).tolist()
    for batch in batches:
        print(f"Wornking on Batch {batch}!")
        mps_list = process_batch(path_in, batches[batch], new_size = (1024,1024))

        compression_ratio_dict, ssim_dict, compression_factor_dict = compress_mps(mps_list,
                                                                             cutoff_list)
        
        save_dicts_to_txt(compression_ratio_dict, ssim_dict, compression_factor_dict, batch)

def bootstrap_error_bars(data, n_bootstrap=1000, alpha=0.05):
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))

    ci_lower = np.percentile(means, alpha / 2 * 100)
    ci_upper = np.percentile(means, (1 - alpha / 2) * 100)
    return np.mean(data), ci_lower, ci_upper


def Define_Bins_for_Compression_Factor(compression_factor_dict,
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
            cf = float(cf)
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

def plot_results(bin_centers, mean_ssim, lower_ci, upper_ci):
    plt.figure(figsize=(10, 5))
    plt.scatter(bin_centers, mean_ssim, s=8, color="blue", label="Average SSIM")
    plt.errorbar(bin_centers, mean_ssim, yerr=[np.array(mean_ssim) - np.array(lower_ci), np.array(upper_ci) - np.array(mean_ssim)], fmt='.', ecolor='red', capsize=2)
    plt.xlabel("Compression Factor (1 / Compression Ratio)")
    plt.ylabel("SSIM")
    plt.title("SSIM vs. Compression Factor with Error Bars")
    plt.legend()
    plt.grid(True)
    # plt.savefig(output_plot_path)
    plt.show()


def Miros_results(loaded_compression_ratio_DCT, loaded_ssim_dict):
    common_compression_factors = np.arange(0.5, 30.005, 0.005)

    interpolated_ssim_values = []

    for i in range(len(loaded_compression_ratio_DCT)):
        cr = np.array(loaded_compression_ratio_DCT[f'Image{i}'])

        ssim_ = np.array(loaded_ssim_dict[f'Image{i}'])

        boolean_list = np.where(1/cr < 100)[0]
        cf = 1/cr[boolean_list]

        ssim_ = ssim_[boolean_list]

        
        interp_func = interp1d(cf, ssim_, kind='linear', bounds_error=False, fill_value=np.nan)

        ssim_interp = interp_func(common_compression_factors)
        interpolated_ssim_values.append(ssim_interp)

    interpolated_ssim_values = np.array(interpolated_ssim_values)
    mean_ssim = np.nanmean(interpolated_ssim_values, axis=0)
    std_ssim = np.nanstd(interpolated_ssim_values, axis=0)

    return common_compression_factors, mean_ssim, std_ssim

def Plot_Miros_Results(common_compression_factors, mean_ssim, std_ssim):
    plt.figure(figsize=(12, 6))
    plt.plot(common_compression_factors, mean_ssim, label='Average ssim', color='blue')
    plt.fill_between(
        common_compression_factors,
        mean_ssim - std_ssim,
        mean_ssim + std_ssim,
        color='blue',
        alpha=0.2,
        label='±1 Standard Deviation')

    plt.xlabel('Compression Factor')
    plt.ylabel('SSIM')
    # plt.ylim(15,35)
    plt.title('Average SSIM vs. Compression Factor Img_size = (1024, 1024)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Create_Data()
    path_in = './data/'
    files = [path_in+i for i in  os.listdir(path_in)]
    combined_psnr_dict, combined_compression_ratio_DCT, combined_compression_factor_dict = load_all_dicts(files)
    bin_centers, mean_ssim, lower_ci, upper_ci = Define_Bins_for_Compression_Factor(combined_compression_factor_dict)

    #common_compression_factors, mean_ssim, std_ssim = Miros_results(combined_compression_ratio_DCT, combined_psnr_dict)

    #Plot_Miros_Results(common_compression_factors, mean_ssim, std_ssim)

    plot_results(bin_centers, mean_ssim, lower_ci, upper_ci)


if __name__ == "__main__":
    main()


