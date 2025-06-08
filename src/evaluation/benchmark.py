
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
from copy import deepcopy

"""
This file contains the functions which are used to run the benchmark and track the metrics.
"""


def find_specific_files(directory_path, file_extension=None):
    """Recursively searches for files in a specified directory and returns a list of file paths.

    Args:
        directory_path (str): The path to the directory where the search will be performed.
        file_extension (str, optional): The file extension to filter files by. If None, all files
            will be included regardless of their extension.

    Returns:
        list: A list of file paths matching the specified criteria.

    Example:
        # Find all `.txt` files in the directory
        txt_files = find_specific_files("/path/to/directory", file_extension=".txt")

        # Find all files in the directory
        all_files = find_specific_files("/path/to/directory")
    """
    files = []
    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if file_extension is None or filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

"""def load_tensors(files):
    Loads tensor data from a list of file paths.

    Args:
        files (list of str): A list of file paths pointing to the data files to be loaded.

    Returns:
        tuple: A tuple containing:
            - data_list (list of numpy.ndarray): A list of loaded tensor data arrays.
            - bitsize_list (list of int): A list of the sizes (in bytes) of the loaded data arrays.

    Notes:
        - The function uses the `nibabel` library to load the data from the files.
        - Each file is expected to contain data compatible with `nibabel`.
        - The function prints progress information during the loading process.
    data_list = []
    bitsize_list = []
    for i, file in enumerate(files):
        print(f"Loading file {i+1}/{len(files)}")
        img = nib.load(file)
        bitpix = ut.get_num_bits(img.header.get_data_dtype())
        img_data = img.get_fdata()
        data_list.append(img_data)
        bitsize_list.append(bitpix)
    return data_list, bitsize_list"""

def load_tensors(files, ending, shape = None):
    """Loads tensor data from a list of file paths.

    Args:
        files (list of str): A list of file paths pointing to the data files to be loaded.

    Returns:
        tuple: A tuple containing:
            - data_list (list of numpy.ndarray): A list of loaded tensor data arrays.
            - bitsize_list (list of int): A list of the sizes (in bytes) of the loaded data arrays.

    Notes:
        - The function uses the `nibabel` library to load the data from the files.
        - Each file is expected to contain data compatible with `nibabel`.
        - The function prints progress information during the loading process."""
    # if len(shape) > 3:
    #     assert "UNSUPORTED TENSOR SHAPE IN benchmark.py"

    if shape is None:
        if '.gz' in ending:
            data_list = []
            bitsize_list = []
            for i, file in enumerate(files):
                print(f"Loading file {i+1}/{len(files)}")
                img = nib.load(file)
                img_data = img.get_fdata()
                data_list.append(img_data)
                bitpix = ut.get_num_bits(img.header.get_data_dtype())
                bitsize_list.append(bitpix)

        elif '.npz' in ending:
            data_list = []
            bitsize_list = []
            for i, file in enumerate(files):
                print(f"Loading file {i+1}/{len(files)}")
                img = np.load(file)
                img_data = img['sequence']
                data_list.append(img_data)
                bitsize_list.append(ut.get_num_bits(img_data.dtype))
        
        else:
            assert('Error in benchmark.py: Unsuported file ')
    
    else:
        B, H, W = shape
        if '.gz' in ending:
            data_list = []
            bitsize_list = []
            for i, file in enumerate(files):
                print(f"Loading file {i+1}/{len(files)}")
                img = nib.load(file)
                img_data = img.get_fdata()
                img_data = img_data[:B, :H, :W]
                data_list.append(img_data)
                bitpix = ut.get_num_bits(img.header.get_data_dtype())
                bitsize_list.append(bitpix)

        elif '.npz' in ending:
            data_list = []
            bitsize_list = []
            for i, file in enumerate(files):
                print(f"Loading file {i+1}/{len(files)}")
                img = np.load(file)
                img_data = img['sequence']
                img_data = img_data[:B, :H, :W]
                data_list.append(img_data)
                bitsize_list.append(ut.get_num_bits(img_data.dtype))
        else:
            assert('Error in benchmark.py: Unsuported file ')



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

def calc_compression_ratio(mps_list):
    """Calculate the compression ratios for a list of MPS objects.

    This function iterates over each MPS object in the provided mps_list and computes
    its compression ratio by calling the object's compression_ratio() method.

    Parameters:
        mps_list (list): A list of MPS objects, each of which must have a compression_ratio()
                         method that returns a compression ratio value.

    Returns:
        list: A list of compression ratio values, one for each MPS object in the list."""
    compression_ratios = []
    for i, mps in enumerate(mps_list):
        compression_ratios.append(mps.compression_ratio())
    return compression_ratios

def get_storage_space_on_disk(mps_list, dtype=np.uint16):
    """
    Calculates the storage space required on disk for a list of MPS (Matrix Product State) objects.

    Args:
        mps_list (list): A list of NDMPS objects for which the storage space is to be calculated.
        dtype (numpy.dtype, optional): The data type used for the storage space calculation. 
            This determines the number of bits per element. Defaults to numpy.uint16.

    Returns:
        list: A list of storage space values (in bytes) corresponding to each MPS object in the input list.
    """
    storage_space = []
    for i, mps in enumerate(mps_list):
        storage_space.append(mps.get_storage_space(dtype))
    return storage_space

def get_gzip_bytesize_on_disk(mps_list):
    """Computes the gzip-compressed size in bytes on disk for a list of MPS objects.

    Parameters:
        mps_list (List[NDMPS]): A list of MPS objects, each expected to implement the method get_bytesize_on_disk(), which returns the size in bytes of the object when stored using gzip compression.

    Returns:
        List[int]: A list of integers where each integer represents the gzip-compressed size in bytes for the corresponding MPS object in mps_list.
    """
    gzip_bytesize = []
    for i, mps in enumerate(mps_list):
        gzip_bytesize.append(mps.get_bytesize_on_disk())
    return gzip_bytesize

def get_compression_ratio_on_disk_with_gzip(mps_list, dtype=np.uint16):
    """Calculates the on-disk compression ratios for a list of MPS objects using gzip compression.

    This function iterates over a list of MPS objects, replacing the tensor data with
    integer-truncated representations (based on the provided data type) before computing
    the compression ratio on disk. This replacement is necessary for an accurate calculation
    of the number of bits used for storage.

    Parameters:
        mps_list (list): A list of MPS objects for which the compression ratio is to be computed.
                         Each object is expected to have a method named `compression_ratio_on_disk`.
        dtype (data-type, optional): The data type (e.g., np.uint16) used to determine the number 
                                     of bits for storage calculations. Defaults to np.uint16.

    Returns:
        list: A list of compression ratios corresponding to each MPS object in `mps_list`."""
    """
    Careful here we replace the tensor data with the integer truncated ones
    """
    compression_ratios = []
    for i, mps in enumerate(mps_list):
        compression_ratios.append(mps.compression_ratio_on_disk(dtype, replace=True))
    return compression_ratios

def benchmark_SSIM(mps_list, original_tensor_list):
    """
    Compute the average SSIM values for a list of MPS representations.

    Parameters:
        mps_list (list): List of compressed MPS objects. Each object must have a 'dim' attribute indicating its dimensions of the original tensor 
                         (2, 3, or 4) and a 'to_tensor()' method to convert the MPS to tensor form.
        original_tensor_list (list): List of original tensors corresponding to each MPS object. These are used as the
                                     reference to compute the SSIM values.

    Returns:
        list: A list of SSIM values computed for each MPS object. The function selects the appropriate SSIM computation
              method based on the dimensionality of each MPS:
                  - For mps.dim == 4, it uses ut.avg_SSIM_4D().
                  - For mps.dim == 3, it uses ut.avg_SSIM_3D().
                  - For mps.dim == 2, it uses ut.compute_ssim_2D().
    """
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
    """Retrieves the bond dimensions for a list of Matrix Product State (MPS) objects.

    Parameters:
        mps_list (list): A list of MPS objects. Each object should have a `bond_sizes()` method 
            that returns its bond dimensions.

    Returns:
        list: A list of bond dimensions corresponding to each MPS object in the input list."""
    bond_dimensions = []
    for mps in mps_list:
        bond_dimensions.append(mps.bond_sizes())
    return bond_dimensions

def get_general_PSNR(mps_list, original_tensor_list):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) for a list of MPS representations.

    Parameters:
        mps_list (list): List of compressed MPS objects. Each object must have a 'dim' attribute indicating its dimensions of the original tensor 
                         (2, 3, or 4) and a 'to_tensor()' method to convert the MPS to tensor form.
        original_tensor_list (list): List of original tensors corresponding to each MPS object. These are used as the
                                     reference to compute the PSNR values.

    Returns:
        list: A list of PSNR values computed for each MPS object.
    """
    psnr_list = []
    for i, mps in enumerate(mps_list):
        psnr_list.append(ut.calc_PSNR(mps.to_tensor(), original_tensor_list[i]))
    return psnr_list


def get_multidimensional_SSIM(mps_list, original_tensor_list):
    """
    Compute the multi-dimensional Structural Similarity (SSIM) index for a list of tensor pairs.

    This function iterates over a list of MPS objects, converts each to a tensor using its `to_tensor()` method, 
    and computes the 2D SSIM between the resulting tensor and its corresponding original tensor.

    Parameters:
        mps_list (list): A list of MPS objects, each of which must implement a `to_tensor()` method.
        original_tensor_list (list): A list of original tensors for comparison, each corresponding to an MPS in mps_list.

    Returns:
        list: A list containing the SSIM values computed for each pair of tensors.

    Raises:
        IndexError: If mps_list and original_tensor_list have different lengths.
        AttributeError: If any object in mps_list does not have a callable `to_tensor()` method.
    """
    multiD_ssim_list = []
    for i, mps in enumerate(mps_list):
        multiD_ssim_list.append(ut.compute_ssim_2D(mps.to_tensor(), original_tensor_list[i]))
    return multiD_ssim_list

def get_shapes(data_list):
    """Retrieves and returns the shapes of each tensor in the provided list.

    Parameters:
        data_list (List[Tensor]): A list of tensor objects that each have a 'shape' attribute.

    Returns:
        List[Any]: A list containing the shape of each tensor extracted from data_list."""
    shapes = []
    for data in data_list:
        shapes.append(data.shape)
    return shapes

def get_fidelity_list(mps_list, original_mps_list):
    """
    Computes the fidelity between a list of MPS objects and their corresponding original tensors.

    Parameters:
        mps_list (list): A list of MPS objects. Each object must have a 'dim' attribute indicating its dimensions of the original tensor 
                         (2, 3, or 4) and a 'to_tensor()' method to convert the MPS to tensor form.
        original_tensor_list (list): A list of original tensors corresponding to each MPS object. These are used as the
                                     reference to compute the fidelity values.

    Returns:
        list: A list of fidelity values computed for each MPS object.
    """
    fidelity_list = []
    for i in np.arange(len(mps_list)):
        fidelity_list.append(ut.calc_overlap(mps_list[i], original_mps_list[i]))
    return fidelity_list

def run_benchmark(mps_list, original_tensors_list, cutoff_list):
    """
    Runs a series of benchmarks on a list of MPS tensors by iteratively compressing all mps objects int the list with given cutoff values.
    This function computes several evaluation metrics for each compression level, including:
    - SSIM (Structural Similarity Index) between the compressed tensors and the original tensors.
    - Compression ratio based on the tensor data.
    - Bond dimensions of the MPS tensors.
    - Storage size on disk of the individual MPS tensors
    - Storage size on disk of the MPS tensors when compressed with gzip.
    - Compression ratio on disk considering gzip compression.
    Parameters:
        mps_list (list): A list of MPS tensors to be evaluated and compressed.
        original_tensors_list (list): A list of original tensors that serves as reference for SSIM computation.
        cutoff_list (list): A list of cutoff values used to iteratively compress the MPS tensors.
    Returns:
        tuple: A tuple containing:
            - ssim_vals (numpy.ndarray): Transposed array of SSIM values computed for each compression level.
            - compression_ratios (numpy.ndarray): Transposed array of compression ratios computed for each compression level.
            - bond_dims (list): A list of bond dimensions for the MPS tensors at each compression level.
            - plain_disk_sizes (numpy.ndarray): Transposed array of plain storage sizes on disk for each compression level.
            - gzip_disk_sizes (numpy.ndarray): Transposed array of gzip-compressed storage sizes on disk for each compression level.
            - disk_compression_ratios (numpy.ndarray): Transposed array of disk compression ratios when using gzip for each compression level.
    """
    original_mps_list = deepcopy(mps_list)
    ssim_list = [benchmark_SSIM(mps_list, original_tensors_list)]
    multidimensional_SSIM = [get_multidimensional_SSIM(mps_list, original_tensors_list)]
    compressionratio_list = [calc_compression_ratio(mps_list)]
    bonddim_list = [get_bond_dimensions(mps_list)]
    plain_disk_size = [get_storage_space_on_disk(mps_list)]
    gzip_disk_size = [get_gzip_bytesize_on_disk(mps_list)]
    compressionratio_list_disk = [get_compression_ratio_on_disk_with_gzip(mps_list)]
    PSNR_list_general = [get_general_PSNR(mps_list, original_tensors_list)]
    fidelity_list = [get_fidelity_list(mps_list, original_mps_list)]
    for i, cutoff in enumerate(cutoff_list):
        percent = (i + 1) / len(cutoff_list) * 100
        print(f"Status: {percent:.2f}% - Current cutoff: {cutoff}")
        compress_list(mps_list, cutoff)
        bonddim_list.append(get_bond_dimensions(mps_list))
        compressionratio_list.append(calc_compression_ratio(mps_list))
        plain_disk_size.append(get_storage_space_on_disk(mps_list))
        gzip_disk_size.append(get_gzip_bytesize_on_disk(mps_list))
        compressionratio_list_disk.append(get_compression_ratio_on_disk_with_gzip(mps_list))
        ssim_list.append(benchmark_SSIM(mps_list, original_tensors_list))
        PSNR_list_general.append(get_general_PSNR(mps_list, original_tensors_list))
        multidimensional_SSIM.append(get_multidimensional_SSIM(mps_list, original_tensors_list))
        fidelity_list.append(get_fidelity_list(mps_list, original_mps_list))
    return np.array(ssim_list).T, np.array(compressionratio_list).T, bonddim_list, np.array(plain_disk_size).T, np.array(gzip_disk_size).T, np.array(compressionratio_list_disk).T, np.array(PSNR_list_general).T, np.array(multidimensional_SSIM).T, np.array(fidelity_list).T

def run_full_benchmark(Dataset_path, cutoff_list, result_file, Datatype = "MRI", mode = "DCT" , start = 0, end=-1, ending = ".gz", shape = "None"):
    """
    Runs a full benchmark on a dataset of tensors, evaluating compression performance 
    using matrix product states (MPS) and saving the results to a JSON file.

    Args:
        Dataset_path (str): Path to the dataset directory containing the files to process.
        cutoff_list (list): List of cutoff values to use for MPS compression.
        result_file (str): Name of the JSON file where the benchmark results will be saved.
        Datatype (str, optional): Type of data in the dataset. Options are "MRI", "fMRI", or "MRI_Slice". 
                                    Defaults to "MRI".
        start (int, optional): Starting index for selecting files from the dataset. Defaults to 0.
        end (int, optional): Ending index for selecting files from the dataset. Use -1 to include all files 
                                from the starting index. Defaults to -1.

    Returns:
        None: The function saves the benchmark results to a JSON file and does not return any value.

    Notes:
        - The function processes files with the ".gz" extension.
        - For "MRI_Slice" datatype, the MRI data is converted to slices before processing.
        - The benchmark evaluates metrics such as SSIM, compression ratio, bond dimensions, 
            and disk sizes (plain and gzip-compressed).
        - Results are stored in a dictionary and serialized to a JSON file in the 
            "src/evaluation/results/" directory.

    Raises:
        FileNotFoundError: If the specified dataset path or result file path is invalid.
        ValueError: If invalid datatype is provided or if other input arguments are incorrect.
    """
    results_dict = {}
    results_dict["Datatype"] = Datatype
    if end == -1:
        files = find_specific_files(Dataset_path, ending)[start:]
    else:
        files = find_specific_files(Dataset_path, ending)[start:end]
    results_dict["files"] = files
    if Datatype == "MRI" or Datatype == "fMRI" or Datatype == "Video":
        data_list, bitsize_list = load_tensors(files, ending, shape)
    elif Datatype == "MRI_Slice":
        print("Loading MRI slices")
        data_list, bitsize_list = load_tensors(files, ending, shape)
        data_list, bitsize_list = MRI_to_MRI_slices(data_list, bitsize_list)
    results_dict["Mode"] = mode
    results_dict["bitsize_list"] = bitsize_list
    results_dict["shapes"] = get_shapes(data_list)
    mps_list = conv_to_mps(data_list, mode)
    results_dict["cutoff_list"] = cutoff_list.tolist()
    print("Starting benchmark")
    ssim_list, compressionratio_list, bonddim_list, plain_disk_size, gzip_disk_size, compressionratio_list_disk, PSNR_general_list, multidimensional_SSIM, fidelity_list = run_benchmark(mps_list, data_list, cutoff_list)
    results_dict["ssim_list"] = ssim_list.tolist()
    results_dict["multidimensional_SSIM"] = multidimensional_SSIM.tolist()
    results_dict["PSNR_list"] = PSNR_general_list.tolist()
    results_dict["fidelity_list"] = fidelity_list.tolist()
    results_dict["compressionratio_list"] = compressionratio_list.tolist()
    results_dict["bonddim_list"] = bonddim_list
    results_dict["plain_disk_size"] = plain_disk_size.tolist()
    results_dict["gzip_disk_size"] = gzip_disk_size.tolist()
    results_dict["compressionratio_list_disk"] = compressionratio_list_disk.tolist()
    print("Saving results ",result_file)
    with open("src/evaluation/results/"+result_file, 'w') as fp:
        json.dump(results_dict, fp)


def MRI_to_MRI_slices(data_list, bitsize_list=None):
    """
    Extracts central slices from 3D MRI volumes.

    This function takes a list of 3D MRI data arrays and, for each array, extracts three 2D slices:
        - The slice at the middle index along the first dimension.
        - The slice at the middle index along the second dimension.
        - The slice at the middle index along the third dimension.

    Parameters:
        data_list (List[numpy.ndarray]): A list of 3D numpy arrays representing MRI volumes.

    Returns:
        List[numpy.ndarray]: A list of 2D numpy arrays where each MRI volume contributes three slices,
        corresponding to its central slices along each dimension.
    """
    img_data_list = []
    bitsize_list_new = []
    for i, data in enumerate(data_list):
        img_data_list.append(data[data.shape[0]//2, :, :])
        bitsize_list_new.append(bitsize_list[i])
        img_data_list.append(data[:, data.shape[1]//2, :])
        bitsize_list_new.append(bitsize_list[i])
        img_data_list.append(data[:, :, data.shape[2]//2])
        bitsize_list_new.append(bitsize_list[i])

    return img_data_list, bitsize_list_new


def combine_jsons(input_file_1, input_file_2, output_file):
    """
    Combines two benchmarking JSON files into a single JSON file.

    This function reads two JSON files from the "src/evaluation/results/" directory, 
    merges their contents, and writes the combined data back to a new JSON file in the same directory. 
    For list-type values (except for the "cutoff_list" key), the lists are concatenated. For non-list values or the "cutoff_list", 
    the value from the first file is used.

    Parameters:
        input_file_1 (str): The filename of the first JSON file to be combined.
        input_file_2 (str): The filename of the second JSON file to be combined.
        output_file (str): The filename for the resulting combined JSON file.

    Raises:
        FileNotFoundError: If either of the input files cannot be found.
        json.JSONDecodeError: If an input file is not a valid JSON.
        
    Side Effects:
        Writes the combined JSON content to a file in the "src/evaluation/results/" directory.
        Prints a success message upon completion.
    """
    with open(input_file_1, "r") as file1:
        data1 = json.load(file1)

    with open(input_file_2, "r") as file2:
        data2 = json.load(file2)

    # Create a new dictionary for the combined data
    combined = {}

    # Assume that the keys in both JSONs are the same.
    for key in data1:
        # If the value is a list in both JSONs, concatenate them.
        if key == "cutoff_list":
            combined[key] = data1[key]
        elif key == "mode":
            combined[key] = data1[key]
        elif isinstance(data1[key], list) and isinstance(data2[key], list):
            combined[key] = data1[key] + data2[key]
        else:
            # Otherwise, take the value from the first file (or handle as needed)
            combined[key] = data1[key]

    # Write the combined dictionary to a new JSON file
    with open(output_file, "w") as outfile:
        json.dump(combined, outfile, indent=4)

    print("Combined JSON created successfully!")