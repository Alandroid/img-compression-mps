import numpy as np
import time
from sympy.ntheory import factorint
from skimage.metrics import structural_similarity as ssim
from scipy.interpolate import interp1d

encoding_map_cache = {}


def calc_mean_std(results_dict: dict):
    """
    Calculates the mean and standard deviation of SSIM values across a common compression factor range.
    
    Args:
        results_dict (dict): Dictionary containing SSIM values and compression ratios.
    
    Returns:
        tuple:
            - np.ndarray: Mean SSIM values for common compression factors.
            - np.ndarray: Standard deviation of SSIM values.
            - np.ndarray: Common compression factor values.
    """
    ssims = np.array(results_dict["ssim_list"])
    comps = np.array(results_dict["compression_ratio_list"])
    
    max_common_fac = np.min(1 / comps[:, -1])
    min_common_fac = np.max(1 / comps[:, 0])
    common_comp_facs = np.linspace(min_common_fac, max_common_fac, 20)
    
    interpolated_ssim = []
    for x, y in zip(1 / comps, ssims):
        interp_func = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
        interpolated_ssim.append(interp_func(common_comp_facs))
    
    return np.nanmean(interpolated_ssim, axis=0), np.nanstd(interpolated_ssim, axis=0), np.array(common_comp_facs)


def time_function(func):
    """
    Decorator to time function execution.
    
    Args:
        func (callable): Function to be timed.
    
    Returns:
        callable: Wrapped function with execution time logging.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Time to run {func.__name__}: {elapsed_time:.8f} seconds")
        return result
    return wrapper


def balance_factors(factor_list: list, target_num: int):
    """
    Balances the number of factors in a factor list by grouping the smallest factors
    until the list has the target number of factors.
    
    Args:
        factor_list (list[int]): The list of factors for a single dimension.
        target_num (int): The desired number of factors after balancing.
    
    Returns:
        list[int]: The balanced list of factors.
    """
    while len(factor_list) > target_num:
        smallest = factor_list.pop(0)
        next_smallest = factor_list.pop(0)
        grouped = smallest * next_smallest
        insertion_index = 0
        while insertion_index < len(factor_list) and factor_list[insertion_index] < grouped:
            insertion_index += 1
        factor_list.insert(insertion_index, grouped)
    return factor_list


def get_factorlist(shape: tuple):
    """
    Generates factor lists and cumulative block sizes for tensor reshaping.
    
    Args:
        shape (tuple): Shape of the input tensor.
    
    Returns:
        tuple: (factor_lists, prod_block_sizes)
    """
    factor_lists = []
    for dim_size in shape:
        factors_dict = factorint(dim_size)
        factors = sorted([prime for prime, exponent in factors_dict.items() for _ in range(exponent)])
        factor_lists.append(factors if factors else [1])
    
    min_factors = min(len(factors) for factors in factor_lists)
    
    for idx, factors in enumerate(factor_lists):
        if len(factors) > min_factors:
            factor_lists[idx] = balance_factors(factors, min_factors)
        elif len(factors) < min_factors:
            factor_lists[idx].extend([1] * (min_factors - len(factors)))
            factor_lists[idx] = sorted(factor_lists[idx])
        if idx % 2 == 1:
            factor_lists[idx] = factor_lists[idx][::-1]
    
    factor_lists = np.array(factor_lists).T
    prod_block_sizes = np.ones((len(factor_lists) + 1, len(factor_lists[0])), dtype=int)
    prod_block_sizes[1:-1] = np.cumprod(factor_lists[-1:0:-1], axis=0)[::-1]
    prod_block_sizes[0] = prod_block_sizes[0] * 1e100 # TODO maybe find another way for this
    
    return factor_lists, prod_block_sizes


def hierarchical_block_indexing(index: np.ndarray, prod_block_sizes: np.ndarray):
    """
    Computes hierarchical block indexing for a tensor.
    
    Args:
        index (np.ndarray): Indices of tensor elements.
        prod_block_sizes (np.ndarray): Cumulative block sizes.
    
    Returns:
        np.ndarray: Mapped hierarchical indices.
    """
    return np.floor(
        np.mod(
            index.reshape([1] + list(index.shape)), 
            prod_block_sizes[:-1].reshape(list(prod_block_sizes[:-1].shape) + [1] * prod_block_sizes.shape[1])
        ) / prod_block_sizes[1:].reshape(list(prod_block_sizes[1:].shape) + [1] * prod_block_sizes.shape[1])
    ).astype(int)


def rebuild_from_blocks(block_tensor, original_shape):
    """
    Reconstructs the original tensor from non-overlapping blocks.

    Args:
        block_tensor (np.ndarray): Block-transformed tensor.
        original_shape (tuple): Shape of the original tensor.

    Returns:
        np.ndarray: Fully reconstructed tensor.
    """
    # Get block shape
    block_shape = block_tensor.shape[2:]  # Extract actual block size

    # Determine the number of blocks in each dimension
    num_blocks = tuple(original_shape[i] // block_shape[i] for i in range(len(original_shape)))

    # Reshape into a single contiguous array with correct shape
    reconstructed_tensor = block_tensor.reshape(*num_blocks, *block_shape)

    # Swap axes to interleave blocks correctly
    axes_order = np.arange(len(original_shape) * 2).reshape(2, -1).T.flatten()
    reconstructed_tensor = reconstructed_tensor.transpose(*axes_order)

    # Merge interleaved blocks back into the full-size image
    reconstructed_tensor = reconstructed_tensor.reshape(original_shape)

    return reconstructed_tensor


def extract_blocks(tensor, block_shape):
    """
    Extracts non-overlapping blocks from an N-D tensor.
    
    Args:
        tensor (np.ndarray): The input tensor.
        block_shape (tuple): The shape of each block.
    
    Returns:
        np.ndarray: Reshaped tensor with extracted blocks.
    """
    shape = tensor.shape
    num_blocks = tuple(shape[i] // block_shape[i] for i in range(len(shape)))

    # Reshape to (num_blocks[0], num_blocks[1], block_size[0], block_size[1])
    reshaped_tensor = tensor.reshape(*num_blocks, *block_shape)

    return reshaped_tensor


def get_valid_dct_levels(shape: tuple):
    """
    Computes the valid hierarchical levels where DCT can be applied.
    
    Args:
        shape (tuple): Shape of the input tensor.
    
    Returns:
        list: Possible DCT levels (0: finest, -1: coarsest).
    """
    factor_lists, _ = get_factorlist(shape)
    num_levels = len(factor_lists)
    
    return list(range(num_levels))  # Levels are indexed from 0 (finest) to num_levels - 1 (coarsest)


def gen_encoding_map(shape: tuple, dct_level: int = -1):
    """
    Generates an encoding map for tensor reshaping and allows selecting a hierarchical level for DCT.
    
    Args:
        shape (tuple): Shape of the input tensor.
        dct_level (int, optional): The hierarchical level at which to extract block sizes for DCT.
                                   -1 (default) uses the coarsest level.
                                   0 uses the finest level.
                                   None applies DCT to the entire unpatched tensor.
    
    Returns:
        tuple: (block_sizes, encoding_map, selected_block_size)
    """
    block_sizes, prod_blocks = get_factorlist(shape)
    indices_all = np.indices(shape)
    mapped_indexes = hierarchical_block_indexing(indices_all, prod_blocks)
    final_map = np.empty([len(block_sizes)] + list(shape))

    for i in range(len(block_sizes)):
        final_map[i] = np.ravel_multi_index(mapped_indexes[i], block_sizes[i])

    # Handle case where we don't want patching (apply DCT to the whole image)
    if dct_level is None:
        selected_block_size = shape  # Use the entire image as a single block
    else:
        selected_level = dct_level if dct_level >= 0 else len(block_sizes) - 1
        selected_block_size = tuple(block_sizes[selected_level])  # Extract the corresponding block size

    return np.prod(block_sizes, axis=1), final_map.astype(int), selected_block_size


def compute_ssim(original:np.ndarray, compressed:np.ndarray):
    """
    Computes SSIM for full N-Dimensional volumes, treating (x1, x2, ..., xN) as a continuous structure.
    """
    compressed = np.clip(compressed, 0, None)
    return ssim(original, compressed, data_range=compressed.max() - compressed.min(), multichannel=False)



# TODO: old way of computing SSIM
# def compute_ssim_2D(original: np.ndarray, compressed: np.ndarray):
#     """
#     Computes the SSIM between two 2D images.
    
#     Args:
#         original (np.ndarray): Original image.
#         compressed (np.ndarray): Compressed image.
    
#     Returns:
#         float: SSIM value.
#     """
#     compressed = np.clip(compressed, 0, None)
#     return ssim(original, compressed, data_range=compressed.max() - compressed.min())


# def SSIM_3D_axis(original: np.ndarray, compressed: np.ndarray, axis: int = 0):
#     """
#     Computes the SSIM for a 3D image along a specified axis.
    
#     Args:
#         original (np.ndarray): Original 3D image.
#         compressed (np.ndarray): Compressed 3D image.
#         axis (int, optional): Axis along which to compute SSIM. Defaults to 0.
    
#     Returns:
#         list[float]: List of SSIM values for each slice along the given axis.
#     """
#     compressed = np.clip(compressed, 0, None)  # Ensure no negative pixel values
    
#     func_map = {
#         0: lambda i: compute_ssim_2D(original[i, :, :], compressed[i, :, :]),
#         1: lambda i: compute_ssim_2D(original[:, i, :], compressed[:, i, :]),
#         2: lambda i: compute_ssim_2D(original[:, :, i], compressed[:, :, i])
#     }
#     return [func_map[axis](i) for i in range(original.shape[axis])]


# def avg_SSIM_3D(original: np.ndarray, compressed: np.ndarray):
#     """
#     Computes the average SSIM for a 3D image along all axes.
    
#     Args:
#         original (np.ndarray): Original 3D image.
#         compressed (np.ndarray): Compressed 3D image.
    
#     Returns:
#         float: Average SSIM value.
#     """
#     return np.mean(
#         [np.mean(
#             [compute_ssim_2D(original.take(i, axis=a), compressed.take(i, axis=a)) for i in range(original.shape[a])]
#             ) for a in range(3)
#         ])


# def avg_SSIM_4D(original: np.ndarray, compressed: np.ndarray):
#     """
#     Computes the SSIM for a 4D fMRI image.
    
#     Args:
#         original (np.ndarray): Original 4D fMRI image.
#         compressed (np.ndarray): Compressed 4D fMRI image.
    
#     Returns:
#         float: Average SSIM value.
#     """
#     return np.mean([avg_SSIM_3D(original[..., i], compressed[..., i]) for i in range(original.shape[-1])])
