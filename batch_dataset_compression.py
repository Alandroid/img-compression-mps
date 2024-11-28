import os
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from mps import BWMPS
from utils import resize_image, output_image_quality



def resize_image(img, new_size):
    return np.array(img.resize(new_size, Image.LANCZOS))


# Batch processing for resizing images
def simple_resize_batch(path_in, new_size=(1024, 1024), batch_size=10, max_number_images=None):
    files = os.listdir(path_in)
    files.sort()  # Ensure consistent processing order

    total_images_processed = 0  # Counter for total processed images
    for i in range(0, len(files), batch_size):
        if max_number_images and total_images_processed >= max_number_images:
            print(f"Reached the maximum number of images: {max_number_images}. Stopping...")
            break

        batch_files = files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{len(files) // batch_size + 1}")
        mps_list = []

        for filename in batch_files:
            if max_number_images and total_images_processed >= max_number_images:
                break

            try:
                image_path = os.path.join(path_in, filename)
                with Image.open(image_path) as img:
                    img_array = resize_image(img, new_size)

                mps = BWMPS.from_matrix(img_array, norm=False, mode="DCT")
                mps_list.append([mps, img_array / img_array.max()])  # Normalize pixel values for SSIM
                total_images_processed += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        yield mps_list


# Error bar calculation functions
def gaussian_error_bars(data, alpha=0.05):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z = 1.96  # 95% confidence interval
    return mean, mean - z * std / np.sqrt(len(data)), mean + z * std / np.sqrt(len(data))


def bootstrap_error_bars(data, n_bootstrap=1000, alpha=0.05):
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))

    ci_lower = np.percentile(means, alpha / 2 * 100)
    ci_upper = np.percentile(means, (1 - alpha / 2) * 100)
    return np.mean(data), ci_lower, ci_upper


# Process images and group results into bins
def process_images_with_bins(
    mps_list, cutoff_list, limit_xaxis=40, error_method="bootstrap", n_bootstrap=1000, bins=50
):
    compression_factor_dict = defaultdict(list)

    for i, (mps, img_array) in enumerate(mps_list):
        print(f"Processing image {i+1}...")
        for cutoff in cutoff_list:
            mps.compress(cutoff)
            final_matrix = mps.mps_to_matrix()

            compression_ratio = mps.compression_ratio()
            compression_factor = 1 / compression_ratio
            ssim_value = output_image_quality(img_array, final_matrix, metric="ssim")
            print(f"Compression factor: {compression_factor:.2f}, SSIM: {ssim_value:.3f}")

            compression_factor_dict[compression_factor].append(ssim_value)

            if compression_factor > limit_xaxis:
                break

    # Define bins for compression factors
    bin_edges = np.geomspace(1e-2, limit_xaxis + 1, bins + 1) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    binned_ssim = defaultdict(list)
    for cf, ssim_values in compression_factor_dict.items():
        bin_index = np.digitize(cf, bin_edges) - 1
        if 0 <= bin_index < bins:
            binned_ssim[bin_index].extend(ssim_values)

    mean_ssim, lower_ci, upper_ci = [], [], []
    for b in range(bins):
        ssim_values = binned_ssim[b]
        if ssim_values:
            if error_method == "bootstrap":
                mean, lower, upper = bootstrap_error_bars(np.array(ssim_values), n_bootstrap=n_bootstrap)
            else:
                mean, lower, upper = gaussian_error_bars(np.array(ssim_values))
            mean_ssim.append(mean)
            lower_ci.append(lower)
            upper_ci.append(upper)
        else:
            mean_ssim.append(np.nan)
            lower_ci.append(np.nan)
            upper_ci.append(np.nan)

    return bin_centers, mean_ssim, lower_ci, upper_ci


# Save batch results
def save_batch_results(batch_idx, bin_centers, mean_ssim, lower_ci, upper_ci, output_folder):
    data = {
        "batch_idx": batch_idx,
        "bin_centers": bin_centers.tolist(),
        "mean_ssim": mean_ssim,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }
    output_file = os.path.join(output_folder, f"batch_{batch_idx}_results.json")
    with open(output_file, "w") as f:
        json.dump(data, f)
    print(f"Batch {batch_idx} results saved.")


# Combine batch results
def combine_batches(output_folder):
    import glob

    all_bin_centers, all_mean_ssim, all_lower_ci, all_upper_ci = [], [], [], []
    batch_files = sorted(glob.glob(os.path.join(output_folder, "batch_*_results.json")))

    for batch_file in batch_files:
        with open(batch_file, "r") as f:
            data = json.load(f)
            all_bin_centers.append(data["bin_centers"])
            all_mean_ssim.append(data["mean_ssim"])
            all_lower_ci.append(data["lower_ci"])
            all_upper_ci.append(data["upper_ci"])

    final_mean_ssim = np.nanmean(all_mean_ssim, axis=0)
    final_lower_ci = np.nanmean(all_lower_ci, axis=0)
    final_upper_ci = np.nanmean(all_upper_ci, axis=0)
    final_bin_centers = all_bin_centers[0]

    return final_bin_centers, final_mean_ssim, final_lower_ci, final_upper_ci


# Plot results
def plot_results(bin_centers, mean_ssim, lower_ci, upper_ci, output_plot_path):
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

# Main function
def main():
    path_in = '../DIV2K_BW/'
    output_folder = "./"
    os.makedirs(output_folder, exist_ok=True)

    new_size = (1024, 1024)
    cutoff_list = np.arange(0, 4, 0.005).tolist()
    batch_size = 10
    max_number_images = 50
    bins = 50

    for batch_idx, batch in enumerate(simple_resize_batch(path_in, new_size, batch_size, max_number_images)):
        bin_centers, mean_ssim, lower_ci, upper_ci = process_images_with_bins(batch, cutoff_list, bins=bins)
        save_batch_results(batch_idx, bin_centers, mean_ssim, lower_ci, upper_ci, output_folder)

    bin_centers, mean_ssim, lower_ci, upper_ci = combine_batches(output_folder)
    plot_results(bin_centers, mean_ssim, lower_ci, upper_ci, "/final_plot_ssim.png")


if __name__ == "__main__":
    main()
