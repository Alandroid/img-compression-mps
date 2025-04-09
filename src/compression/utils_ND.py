import numpy as np
from sympy.ntheory import factorint
from skimage.metrics import structural_similarity as ssim
from functools import reduce
from operator import mul
import time


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Time to run {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


# @time_function
def balance_factors(factor_list, target_num):
    """
    Balances the number of factors in a factor list by grouping the smallest factors
    until the list has the target number of factors.

    Arguments:
        factor_list (list of int): The list of factors for a single dimension.
        target_num (int): The desired number of factors after balancing.

    Returns:
        list of int: The balanced list of factors.
    """
    while len(factor_list) > target_num:
        # Multiply the two smallest factors
        smallest = factor_list.pop(0)
        next_smallest = factor_list.pop(0)
        grouped = smallest * next_smallest
        # Insert the grouped factor back, maintaining sorted order
        insertion_index = 0
        while insertion_index < len(factor_list) and factor_list[insertion_index] < grouped:
            insertion_index += 1
        factor_list.insert(insertion_index, grouped)
    return factor_list

def get_factorlist(shape):
    # todo rename to actually get_block_sizes
    factor_lists = []
    for dim_size in shape:
        if dim_size == 1:
            # Handle dimension of size 1 by assigning a factor of 1
            factors = [1]
        else:
            factors_dict = factorint(dim_size)
            # Extract prime factors and repeat them according to their exponents
            factors = []
            for prime, exponent in sorted(factors_dict.items()):
                factors.extend([prime] * exponent)
        # Sort the factors in ascending order
        factors_sorted = sorted(factors)
        factor_lists.append(factors_sorted)
    
    # Step 2: Balancing the number of factors across all dimensions
    # Determine the minimum number of factors among all dimensions
    min_factors = min(len(factors) for factors in factor_lists)
    
    # Balance factors by grouping the smallest factors in dimensions with more factors
    for idx, factors in enumerate(factor_lists):
        if len(factors) > min_factors:
            factor_lists[idx] = balance_factors(factors, min_factors)
        elif len(factors) < min_factors:
            # If a dimension has fewer factors, pad with 1s to reach min_factors
            # This effectively treats missing factors as trivial
            factor_lists[idx].extend([1] * (min_factors - len(factors)))
            factor_lists[idx] = sorted(factor_lists[idx])
        # If equal, do nothing
    for idx, list in enumerate(factor_lists):
        if idx%2 == 1:
            factor_lists[idx] = factor_lists[idx][::-1] 

    factor_lists = np.array(factor_lists).T
    prod_block_sizes = np.ones((len(factor_lists)+1, len(factor_lists[0])), dtype = int)
    prod_block_sizes[1:-1] = np.cumprod(factor_lists[-1:0:-1], axis =0)[::-1]
    prod_block_sizes[0] = prod_block_sizes[0] * 1e100

    return factor_lists, prod_block_sizes

#@time_function
def hierarchical_block_indexing(index, prod_block_sizes):
    return np.floor(np.mod(index.reshape([1]+list(index.shape)), prod_block_sizes[:-1].reshape(list(prod_block_sizes[:-1].shape)+[1]*(prod_block_sizes.shape[1])))/prod_block_sizes[1:].reshape(list(prod_block_sizes[1:].shape)+[1]*(prod_block_sizes.shape[1]))).astype(int)

def gen_encoding_map(shape):
    dim = len(shape)
    block_sizes, prod_blocks = get_factorlist(shape)
    indices_all = np.indices(shape)
    mapped_indexes = hierarchical_block_indexing(indices_all, prod_blocks)
    final_map = np.empty([len(block_sizes)]+list(shape))
    for i in range(len(block_sizes)):
        final_map[i] = np.ravel_multi_index(mapped_indexes[i], block_sizes[i])
    return np.prod(block_sizes, axis= 1), final_map.astype(int)

# calculates the SSIM for a 2D image
def compute_ssim_2D(original, compressed):
    # Assuming input and output images the same size
    original_arr = np.array(original)
    compressed_arr = np.array(compressed)
    compressed_arr = np.clip(compressed_arr, 0, None)
    return ssim(original_arr, compressed_arr, data_range=max(original_arr.max(), compressed_arr.max()) - min(original_arr.min(), compressed_arr.min()))


def SSIM_3D_axis(original, compressed, axis = 0):
    """
    Calculate the SSIM for a 3D image along a specified axis.
    """
    ssim_list = []
    compressed = np.clip(compressed, 0, None) # clip is necessary to avoid negative pixel values from noise
    if axis == 0:
        for i in np.arange(original.shape[0]):
            ssim_list.append(compute_ssim_2D(original[i, :, :], compressed[i, :, :]))
    
    elif axis == 1:
        for i in np.arange(original.shape[1]):
            ssim_list.append(compute_ssim_2D(original[:, i, :], compressed[:, i, :]))
    elif axis == 2:
        for i in np.arange(original.shape[2]):
            ssim_list.append(compute_ssim_2D(original[:, :, i], compressed[:, :, i]))
    return ssim_list


def avg_SSIM_3D(original, compressed):
    """
    Compute the average SSIM of a 3D image along all axes.
    """
    ssim_0 = SSIM_3D_axis(original, compressed, axis = 0)
    ssim_1 = SSIM_3D_axis(original, compressed, axis = 1)
    ssim_2 = SSIM_3D_axis(original, compressed, axis = 2)
    return (np.average(ssim_0) + np.average(ssim_1) + np.average(ssim_2))/3


def avg_SSIM_4D(orginal, compressed):
    """
    Computes the SSIM for a fMRI Image by calcualting the SSIM for each MRI image in each frame.
    """
    ssims = []
    for i in range(orginal.shape[-1]):
        ssims.append(avg_SSIM_3D(orginal[:,:,:,i], compressed[:,:,:,i]))
    return np.mean(ssims)


def calc_PSNR(original, compressed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two tensors in general.
    original: The original tensor.
    compressed: The compressed tensor.
    """
    MSE = np.mean((original - compressed) ** 2)
    if MSE == 0:
        return np.inf
    PSNR = 10 * np.log10((np.max(original) ** 2) / MSE)
    return PSNR


def scale_to_dtype(data, dtype=np.uint8):
    """
    Scale the data to dtype for and does integer truncation.
    This class only works for unsigned integer types.
    """
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * np.iinfo(dtype).max).astype(dtype)
    return data

def scale_back(data, min, max, dtype=np.uint8):
    """
    Scale the image back to its original range.
    """
    data = data / np.iinfo(dtype).max
    data = data * (max - min)
    data = data + min
    return data

def get_num_bits(dtype):
    dtype = np.dtype(dtype)  # Ensure it's a NumPy dtype
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).bits
    elif np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).bits
    else:
        raise ValueError(f"Unsupported data type: {dtype}")