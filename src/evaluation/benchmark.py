import os
import sys
import pickle
import json
import nibabel as nib
import numpy as np

# Get the absolute path of the current script
current_path = os.path.abspath(__file__)
project_folder = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.append(project_folder)

from src.compression.mps_ND import NDMPS
import src.compression.utils_ND as ut


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


# TODO: whats the diff of this to the next one? This came from the Jupyter notebook
def load_tensors(files):
    """
    Loads the data from the file name list.
    """
    data_list = []
    for i, file in enumerate(files):
        print(f"Loading file {i + 1}/{len(files)}")
        img = nib.load(file)
        img_data = img.get_fdata()
        data_list.append(img_data)
    return data_list


def conv_to_mps(data_list, encoding_scheme="hierarchical"):
    """
    Converts the data list to a list of MPS objects.
    """
    mps_list = []
    for i, data in enumerate(data_list):
        print(f"Converting file {i + 1}/{len(data_list)}")
        mps = NDMPS.from_tensor(data, norm=False, mode="DCT", encoding_scheme=encoding_scheme)
        mps_list.append(mps)
    return mps_list


def conv_to_tensors(mps_list):
    """
    Converts a list of MPS objects back to tensors.
    """
    data_list = []
    for i, mps in enumerate(mps_list):
        print(f"Converting file {i + 1}/{len(mps_list)}")
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
    return [mps.compression_ratio() for mps in mps_list]


def benchmark_ssim(mps_list: list, original_tensor_list: list):
    """
    Computes the Structural Similarity Index (SSIM) between compressed and original tensors.

    Args:
        mps_list (list): List of MPS objects.
        original_tensor_list (list): List of original tensor data.

    Returns:
        list: List of SSIM values.
    """
    ssim_list = []
    for i, mps in enumerate(mps_list):
        tensor = mps.to_tensor()

        # Ensure tensors have the same shape before SSIM computation
        if tensor.shape != original_tensor_list[i].shape:
            tensor = np.resize(tensor, original_tensor_list[i].shape)

        func_map = {
            4: ut.avg_SSIM_4D,
            3: ut.avg_SSIM_3D,
            2: ut.compute_ssim_2D
        }
        ssim_list.append(func_map.get(original_tensor_list[i].ndim, lambda x, y: None)(tensor, original_tensor_list[i]))
    
    return ssim_list


def get_bond_dimensions(mps_list):
    """
    Retrieves bond dimensions for a list of MPS objects.
    """
    return [mps.bond_sizes() for mps in mps_list]


def get_shapes(data_list):
    """
    Retrieves the shapes of tensors in a list.
    """
    return [data.shape for data in data_list]


def run_benchmark(mps_list: list, original_tensors_list: list, cutoff_list: np.ndarray):
    """
    Runs a benchmark test on MPS compression.
    
    Args:
        mps_list (list): List of MPS objects.
        original_tensors_list (list): List of original tensor data.
        cutoff_list (np.ndarray): List of compression cutoff values.
    
    Returns:
        tuple: Arrays of SSIM values, compression ratios, and bond dimensions.
    """
    ssim_list = [benchmark_ssim(mps_list, original_tensors_list)]
    compression_ratio_list = [calc_compression_ratio(mps_list)]
    bond_dim_list = []
    
    for i, cutoff in enumerate(cutoff_list):
        print(f"Running compression step {i + 1}/{len(cutoff_list)}")
        compress_list(mps_list, cutoff)
        bond_dim_list.append(get_bond_dimensions(mps_list))
        ssim_list.append(benchmark_ssim(mps_list, original_tensors_list))
        compression_ratio_list.append(calc_compression_ratio(mps_list))
    
    # TODO: do we need to transpose the last output also?
    return np.array(ssim_list).T, np.array(compression_ratio_list).T, np.array(bond_dim_list)


def run_full_benchmark(
        dataset_path: str, cutoff_list: np.ndarray, result_file: str, 
        datatype: str = "MRI", start: int = 0, end: int = -1
        ):
    """
    Runs a full benchmark test on MRI or fMRI datasets.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        cutoff_list (np.ndarray): Array of cutoff values for compression.
        result_file (str): Name of the output JSON file to store benchmark results.
        datatype (str, optional): Type of dataset ("MRI", "fMRI", or "MRI_Slice"). Defaults to "MRI".
        start (int, optional): Start index for selecting files. Defaults to 0.
        end (int, optional): End index for selecting files. Defaults to -1 (all files from start).
    
    Returns:
        None: The results are saved in a JSON file.
    """
    results_dict = {"Datatype": datatype}
    
    files = find_specific_files(dataset_path, ".gz")[start:] if end == -1 else find_specific_files(dataset_path, ".gz")[start:end]
    results_dict["files"] = files
    
    if datatype in ["MRI", "fMRI"]:
        data_list = load_tensors(files)
    elif datatype == "MRI_Slice":
        data_list = mri_to_mri_slices(load_tensors(files))
    
    results_dict["shapes"] = get_shapes(data_list)
    mps_list = conv_to_mps(data_list)
    results_dict["cutoff_list"] = cutoff_list.tolist()
    
    print("Starting benchmark")
    ssim_list, compression_ratio_list, bond_dim_list = run_benchmark(mps_list, data_list, cutoff_list)
    
    results_dict["ssim_list"] = ssim_list.tolist()
    results_dict["compression_ratio_list"] = compression_ratio_list.tolist()
    results_dict["bonddim_list"] = bond_dim_list
    
    with open(os.path.join("results", result_file), 'w') as fp:
        json.dump(results_dict, fp)


def mri_to_mri_slices(data_list: list):
    """
    Extracts central slices from 3D MRI scans.
    
    Args:
        data_list (list): List of 3D MRI tensors.
    
    Returns:
        list: List of extracted 2D MRI slices.
    """
    slices = []
    for data in data_list:
        slices.append(data[data.shape[0] // 2, :, :])
        slices.append(data[:, data.shape[1] // 2, :])
        slices.append(data[:, :, data.shape[2] // 2])
    return slices


def save_object(obj: object, filename: str):
    """
    Saves a Python object to a file using pickle.
    
    Args:
        obj (object): Object to be saved.
        filename (str): Path to the output file.
    
    Returns:
        None
    """
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# # Benchmark execution
# D_PATH = os.path.join(project_folder, 'Data/fMRI_Dataset')
# CUTOFF_LIST = np.linspace(0, 0.1, 10)[1:]

# run_full_benchmark(D_PATH, CUTOFF_LIST, 'results_fMRI_test.json', "fMRI", 0, 5)