
import sys
import os
from src.compression.mps_ND import NDMPS
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import src.compression.utils_ND as ut
import pickle
import json

def find_specific_files(directory_path, file_extension=None):
    """
    Finds all files in a directory with a specific file extension.
    """
    files = []
    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if file_extension is None or filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

def load_tensors(files):
    """
    Loads the data from the file name list.
    """
    data_list = []
    for i, file in enumerate(files):
        print(f"Loading file {i+1}/{len(files)}")
        img = nib.load(file)
        img_data = img.get_fdata()
        data_list.append(img_data)
    return data_list

def conv_to_mps(data_list):
    """
    Conversts the data list to a list of MPS objects.
    """
    mps_list = []
    for i, data in enumerate(data_list):
        print(f"Converting file {i+1}/{len(data_list)}")
        mps = NDMPS.from_tensor(data, norm = False, mode="DCT")
        mps_list.append(mps)
    return mps_list

def conv_to_tensors(mps_list):
    """
    Converts a list of MPS objects back to tensors.
    """
    data_list = []
    for i, mps in enumerate(mps_list):
        print(f"Converting file {i+1}/{len(mps_list)}")
        data = mps.to_tensor()
        data_list.append(data)
    return data_list

def compress_list(mps_list, compression_factors):
    """
    Applies compression to a list of MPS objects.
    """
    for mps in mps_list:
        mps.compress(compression_factors)

def calc_compression_ratio(mps_list):
    """
    Computes the compression ratio for a list of MPS objects.
    """
    compression_ratios = []
    for i, mps in enumerate(mps_list):
        compression_ratios.append(mps.compression_ratio())
    return compression_ratios

def benchmark_SSIM(mps_list, original_tensor_list):
    ssim_list = []
    for i, mps in enumerate(mps_list):
        if mps.dim == 4:
            ssim_list.append(ut.avg_SSIM_4D(mps.to_tensor(), original_tensor_list[i]))
        if mps.dim == 3:
            ssim_list.append(ut.avg_SSIM_3D(mps.to_tensor(), original_tensor_list[i]))
        elif mps.dim == 2:
            ssim_list.append(ut.compute_ssim_2D(mps.to_tensor(), original_tensor_list[i]))
    return ssim_list

def get_bond_dimensions(mps_list):
    """
    Retrieves bond dimensions for a list of MPS objects.
    """
    bond_dimensions = []
    for mps in mps_list:
        bond_dimensions.append(mps.bond_sizes())
    return bond_dimensions

def get_shapes(data_list):
    """
    Retrieves the shapes of tensors in a list.
    """
    shapes = []
    for data in data_list:
        shapes.append(data.shape)
    return shapes

def run_benchmark(mps_list, original_tensors_list, cutoff_list):
    ssim_list = []
    compressionratio_list = []
    bonddim_list = []
    compressionratio_list.append(calc_compression_ratio(mps_list))
    ssim_list.append(benchmark_SSIM(mps_list, original_tensors_list))
    for i, cutoff in enumerate(cutoff_list):
        print(i)
        compress_list(mps_list, cutoff)
        bonddim_list.append(get_bond_dimensions(mps_list))
        ssim_list.append(benchmark_SSIM(mps_list, original_tensors_list))
        compressionratio_list.append(calc_compression_ratio(mps_list))
    return np.array(ssim_list).T, np.array(compressionratio_list).T, bonddim_list

def run_full_benchmark(Dataset_path, cutoff_list, result_file, Datatype = "MRI", start = 0, end=-1):
    results_dict = {}
    results_dict["Datatype"] = Datatype
    if end == -1:
        files = find_specific_files(Dataset_path, ".gz")[start:]
    else:
        files = find_specific_files(Dataset_path, ".gz")[start:end]
    results_dict["files"] = files
    if Datatype == "MRI" or Datatype == "fMRI":
        data_list = load_tensors(files)
    elif Datatype == "MRI_Slice":
        data_list = load_tensors(files)
        data_list = MRI_to_MRI_slices(data_list)
    results_dict["shapes"] = get_shapes(data_list)
    mps_list = conv_to_mps(data_list)
    results_dict["cutoff_list"] = cutoff_list.tolist()
    print("Starting benchmark")
    ssim_list, compressionratio_list, bonddim_list = run_benchmark(mps_list, data_list, cutoff_list)
    results_dict["ssim_list"] = ssim_list.tolist()
    results_dict["compressionratio_list"] = compressionratio_list.tolist()
    results_dict["bonddim_list"] = bonddim_list
    with open("src/evaluation/results/"+result_file, 'w') as fp:
        json.dump(results_dict, fp)


def MRI_to_MRI_slices(data_list):
    img_data_list = []
    for i, data in enumerate(data_list):
        img_data_list.append(data[data.shape[0]//2, :, :])
        img_data_list.append(data[:, data.shape[1]//2, :])
        img_data_list.append(data[:, :, data.shape[2]//2])
    return img_data_list