"""
This file contains the functions which are used to run the benchmark and track the metrics.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import json
from copy import deepcopy

from imgcompressionmps.core.ndmps import NDMPS
from imgcompressionmps.utils.filetools import get_num_bits, find_specific_files, get_shapes, mri_to_slices
from imgcompressionmps.utils.metrics import compute_ssim_by_dim, compute_psnr, compute_overlap


def load_tensors(files, ending, shape=None):
    """
    Loads tensor data from a list of file paths using nibabel.
    
    Args:
        files (list of str): Paths to data files (.gz or .npz).
        ending (str): Expected file extension.
        shape (tuple, optional): Tuple (B, H, W) to slice the data. If None, no slicing.
    
    Returns:
        tuple:
            - data_list (list of np.ndarray): Loaded tensors.
            - bitsize_list (list of int): Corresponding data type sizes (in bits).
    """
    if not (ending.endswith(".gz") or ending.endswith(".npz")):
        raise ValueError(f"Unsupported file extension: {ending}")

    B, H, W = shape if shape else (None, None, None)

    data_list = []
    bitsize_list = []

    for i, path in enumerate(files):
        print(f"Loading file {i+1}/{len(files)}")

        if ending.endswith(".gz"):
            img = nib.load(path)
            data = img.get_fdata()
            if shape:
                data = data[:B, :H, :W]
            dtype = img.header.get_data_dtype()

        elif ending.endswith(".npz"):
            with np.load(path) as archive:
                data = archive["sequence"]
                dtype = data.dtype

        data_list.append(data)
        bitsize_list.append(get_num_bits(dtype))

    return data_list, bitsize_list


def conv_to_mps(data_list, mode = "DCT"):
    """Converts a list of tensors into a list of MPS (Matrix Product State) objects.

    This function takes a list of tensors and converts each tensor into an MPS object
    using the `NDMPS.from_tensor` method. The conversion process does not normalize
    the tensors and uses the "DCT" mode for the transformation. Progress is printed
    to the console for each tensor being converted.

    Args:
        data_list (list): A list of tensors to be converted into MPS objects.
        mode (str, optional): The mode used for preprocession. Default is "DCT".

    Returns:
        list: A list of MPS objects corresponding to the input tensors."""
    mps_list = []
    for i, data in enumerate(data_list):
        print(f"Converting file {i+1}/{len(data_list)}")
        mps = NDMPS.from_tensor(data, norm = False, mode=mode)
        mps_list.append(mps)
    return mps_list


def conv_to_tensors(mps_list):
    """
    Converts a list of Matrix Product State (MPS) objects into their corresponding tensor representations.

    Args:
        mps_list (list): A list of NDMPS objects to be converted to tensors.

    Returns:
        list: A list of tensors obtained from the MPS objects.

    Example:
        mps_list = [mps1, mps2, mps3]
        tensors = conv_to_tensors(mps_list)
        # tensors now contains the tensor representations of the MPS objects.
    """
    data_list = []
    for i, mps in enumerate(mps_list):
        print(f"Converting file {i+1}/{len(mps_list)}")
        data = mps.to_tensor()
        data_list.append(data)
    return data_list


def compress_list(mps_list, compression_factors):
    """Compresses a list of Matrix Product State (MPS) objects using specified compression factors.

    Args:
        mps_list (list): A list of MPS objects to be compressed. Each object in the list
                         should have a `compress` method that accepts compression factors.
        compression_factors (any): The compression factors to be applied to each MPS object.
                                   The type and structure of this parameter depend on the 
                                   implementation of the `compress` method in the MPS class.

    Returns:
        None: The function modifies the MPS objects in place and does not return a value."""
    for mps in mps_list:
        mps.compress(compression_factors)


def benchmark_metric(mps_list, reference_list=None, metric="compression_ratio", dtype=np.uint16):
    results = []

    metric_fn = {
        "compression_ratio": lambda mps, _: mps.compression_ratio(),
        "storage": lambda mps, _: mps.get_storage_space(dtype),
        "gzip_bytes": lambda mps, _: mps.get_bytesize_on_disk(dtype=dtype),
        "gzip_ratio": lambda mps, _: mps.compression_ratio_on_disk(dtype=dtype, replace=True),
        "ssim": lambda mps, ref: compute_ssim_by_dim(mps.to_tensor(), ref),
        "psnr": lambda mps, ref: compute_psnr(mps.to_tensor(), ref),
        "bond_dims": lambda mps, _: mps.bond_sizes(),
        "shape": lambda _, ref: ref.shape,
        "fidelity": lambda mps, ref: compute_overlap(mps, ref)
    }

    if metric not in metric_fn:
        raise ValueError(f"Unsupported metric: {metric}")

    for i, mps in enumerate(mps_list):
        ref = reference_list[i] if reference_list else None
        results.append(metric_fn[metric](mps, ref))

    return results


def run_benchmark(mps_list, original_tensors_list, cutoff_list):
    """
    Runs benchmarks across multiple compression levels.
    """

    original_mps_list = deepcopy(mps_list)

    # Define all metrics in one clean config
    metrics = [
        ("ssim", original_tensors_list),
        #("multi_ssim", original_tensors_list), TODO do we need this
        ("compression_ratio", None),
        ("bond_dims", None),
        ("psnr", original_tensors_list),
        ("fidelity", original_mps_list),
    ]

    results = {name: [] for name, _ in metrics}

    # Initial benchmarks (no compression)
    for name, ref in metrics:
        results[name].append(benchmark_metric(mps_list, ref, metric=name))

    # Run over compression cutoffs
    for i, cutoff in enumerate(cutoff_list):
        print(f"Status: {100*(i+1)/len(cutoff_list):.2f}% - Cutoff: {cutoff}")
        compress_list(mps_list, cutoff)

        for name, ref in metrics:
            results[name].append(benchmark_metric(mps_list, ref, metric=name))

    # Post-process results: transpose where applicable
    for key in results:
        if isinstance(results[key][0], (list, np.ndarray)) and np.ndim(results[key][0]) > 0:
            results[key] = np.array(results[key]).T
        # bond_dims stays as list-of-lists

    return results


def run_full_benchmark(dataset_path, cutoff_list, result_file,
                       datatype="MRI", mode="DCT", start=0, end=-1,
                       ending=".gz", shape=None):
    """
    Runs a full compression benchmark over a dataset using MPS.
    Results are saved as a JSON file.
    """
    dataset_path = Path(dataset_path)
    result_path = Path("src/evaluation/results") / result_file
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # Select files
    files = find_specific_files(dataset_path, ending)
    files = files[start:] if end == -1 else files[start:end]

    # Load tensors
    data_list, bitsize_list = load_tensors(files, ending, shape)
    if datatype == "MRI_Slice":
        data_list, bitsize_list = mri_to_slices(data_list, bitsize_list)

    # Convert to MPS
    mps_list = conv_to_mps(data_list, mode)

    # Run benchmarks
    print("Starting benchmark...")
    metrics = run_benchmark(mps_list, data_list, cutoff_list)

    # Save results
    print(f"Saving results to {result_path}")
    result_dict = {
        "datatype": datatype,
        "mode": mode,
        "files": files,
        "cutoff_list": cutoff_list.tolist(),
        "bitsize_list": bitsize_list,
        "shapes": get_shapes(data_list),
        **{k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
    }

    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
