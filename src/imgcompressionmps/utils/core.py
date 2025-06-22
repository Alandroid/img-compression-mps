import numpy as np
from typing import List, Sequence, Tuple
from sympy.ntheory import factorint


def gen_encoding_map(shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the encoding map and qubit sizes for a tensor of given shape.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the input tensor.

    Returns
    -------
    qubit_sizes : np.ndarray
        Physical qubit dimension per virtual site (product of factors).
    encoding_map : np.ndarray
        Map from physical coordinates to qubit indices with shape (num_levels, *shape).
    """
    if len(shape) == 0:
        raise ValueError("Shape cannot be empty.")
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError("All dimensions must be positive integers.")

    block_sizes, prod_blocks = get_factorlist(shape)
    idx_grid = np.indices(shape)
    mapped = hierarchical_block_indexing(idx_grid, prod_blocks)

    enc_map = np.empty((len(block_sizes), *shape), dtype=int)
    for lvl in range(len(block_sizes)):
        enc_map[lvl] = np.ravel_multi_index(mapped[lvl], block_sizes[lvl])

    return np.prod(block_sizes, axis=1), enc_map.astype(int)


def balance_factors(factors: List[int], target_num: int) -> List[int]:
    """
    Balance a list of factors to match a target number of entries.

    Parameters
    ----------
    factors : List[int]
        Prime factor list for one tensor dimension.
    target_num : int
        Desired final length of the factor list.

    Returns
    -------
    List[int]
        Balanced factor list of length target_num.

    Raises
    ------
    ValueError
        If target_num is negative or zero when factors are non-empty.
    """
    if target_num < 0:
        raise ValueError("target_num must be non-negative.")
    if target_num == 0 and len(factors) > 0:
        raise ValueError("Cannot reduce non-empty factor list to length zero.")

    factors = sorted(factors)

    if len(factors) > target_num:
        while len(factors) > target_num:
            smallest = factors.pop(0)
            next_smallest = factors.pop(0)
            factors.insert(0, smallest * next_smallest)
            factors.sort()
    elif len(factors) < target_num:
        factors += [1] * (target_num - len(factors))

    return sorted(factors)


def get_factorlist(shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced prime factor lists and cumulative products per tensor dimension.

    Parameters
    ----------
    shape : Sequence[int]
        Tensor shape.

    Returns
    -------
    factor_lists : np.ndarray
        Array of shape (num_levels, ndim), where each column is the factor list of a dimension.
    prod_block_sizes : np.ndarray
        Cumulative product arrays used for hierarchical indexing.
    """
    if len(shape) == 0:
        raise ValueError("Shape cannot be empty.")
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError("All dimensions must be positive integers.")

    factor_lists: List[List[int]] = []

    for dim in shape:
        if dim == 1:
            factors = [1]
        else:
            factors = []
            for prime, exp in sorted(factorint(dim).items()):
                factors.extend([prime] * exp)
        factor_lists.append(sorted(factors))

    min_len = min(len(f) for f in factor_lists)
    for i, factors in enumerate(factor_lists):
        factor_lists[i] = balance_factors(factors, min_len)

    for i, factors in enumerate(factor_lists):
        if i % 2 == 1:
            factor_lists[i] = factors[::-1]

    factor_arr = np.array(factor_lists).T  # Shape: (num_levels, ndim)

    prod = np.ones((len(factor_arr) + 1, factor_arr.shape[1]), dtype=int)
    if len(factor_arr) > 1:
        prod[1:-1] = np.cumprod(factor_arr[-1:0:-1], axis=0)[::-1]
    prod[0] = np.iinfo(np.int64).max

    return factor_arr, prod


def hierarchical_block_indexing(
    index: np.ndarray,
    prod_block_sizes: np.ndarray
) -> np.ndarray:
    """
    Map absolute indices to hierarchical block indices via integer division.

    Parameters
    ----------
    index : np.ndarray
        Index grid (from np.indices), shape (ndim, *shape).
    prod_block_sizes : np.ndarray
        Cumulative products from get_factorlist, shape (num_levels + 1, ndim).

    Returns
    -------
    np.ndarray
        Hierarchical indices with shape (num_levels, ndim, *original_shape).

    Raises
    ------
    ValueError
        If index and prod_block_sizes have mismatched dimensions.
    """
    ndim = index.shape[0]
    if prod_block_sizes.ndim != 2 or prod_block_sizes.shape[1] != ndim or prod_block_sizes.shape[0] < 2:
        raise ValueError("prod_block_sizes must be of shape (num_levels + 1, ndim) with ndim matching index.")

    n_levels = prod_block_sizes.shape[0] - 1
    reshape_tail = [1] * ndim

    numer = np.mod(
        index.reshape([1] + list(index.shape)),
        prod_block_sizes[:-1].reshape(list(prod_block_sizes[:-1].shape) + reshape_tail),
    )
    denom = prod_block_sizes[1:].reshape(
        list(prod_block_sizes[1:].shape) + reshape_tail
    )

    return np.floor(numer / denom).astype(int)