import os
import json
import numpy as np
from pathlib import Path


def get_num_bits(dtype: np.dtype) -> int:
    """
    Return the number of bits required to store the given dtype.
    Supports integer and floating types.
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).bits
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).bits
    raise ValueError(f"Unsupported dtype {dtype!r}")


def scale_to_dtype(array: np.ndarray, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Normalize array to the range [0, max of dtype] and cast to dtype.
    """
    arr = array - np.min(array)
    arr = arr / np.max(arr)
    return (arr * np.iinfo(dtype).max).astype(dtype)


def scale_back(
    array: np.ndarray,
    arr_min: float,
    arr_max: float,
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Reconstruct original scale of a normalized and quantized array.
    """
    arr = array / np.iinfo(dtype).max
    return arr * (arr_max - arr_min) + arr_min


def mri_to_slices(data_list, bitsize_list=None):
    """
    Extract central 2D slices from 3D MRI volumes along all three axes.

    Args:
        data_list (List[np.ndarray]): List of 3D MRI volumes.
        bitsize_list (List[int], optional): Corresponding bit sizes.

    Returns:
        Tuple[List[np.ndarray], List[int]]: Always returns a tuple of (slices, bits).
    """
    slices, bits = [], []
    for i, volume in enumerate(data_list):
        if volume.ndim != 3:
            print(f"Skipping non-3D volume at index {i} with shape {volume.shape}")
            continue

        mid_slices = [
            volume[volume.shape[0] // 2, :, :],
            volume[:, volume.shape[1] // 2, :],
            volume[:, :, volume.shape[2] // 2]
        ]
        slices.extend(mid_slices)

        if bitsize_list:
            bits.extend([bitsize_list[i]] * 3)
        else:
            bits.extend([16] * 3)  # Default fallback if bits not provided

    return slices, bits


def find_project_root(marker: str = "src") -> Path:
    """
    Traverse up the directory tree to find the project root containing the given marker folder.
    """
    current = Path(__file__).resolve()
    while not (current / marker).exists() and current != current.parent:
        current = current.parent
    if not (current / marker).exists():
        raise FileNotFoundError(f"Could not find directory containing '{marker}'")
    return current


def find_specific_files(directory_path, file_extension=None):
    """
    Recursively find files in a directory, optionally filtering by extension.
    """
    files = []
    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if file_extension is None or filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files


def combine_jsons(input_file_1, input_file_2, output_file):
    """
    Merge two JSON files into one. Lists (except some keys) are concatenated.
    """
    with open(input_file_1, "r") as file1:
        data1 = json.load(file1)
    with open(input_file_2, "r") as file2:
        data2 = json.load(file2)

    combined = {}
    for key in data1:
        if key in ["cutoff_list", "mode"] or not isinstance(data1[key], list):
            combined[key] = data1[key]
        else:
            combined[key] = data1[key] + data2[key]

    with open(output_file, "w") as outfile:
        json.dump(combined, outfile, indent=4)
    print("Combined JSON created successfully!")


def get_shapes(data_list):
    """
    Return the shapes of each array in the list.
    """
    return [np.shape(data) for data in data_list]