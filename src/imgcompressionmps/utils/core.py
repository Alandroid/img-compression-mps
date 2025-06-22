import numpy as np
from typing import List, Sequence, Tuple
from sympy.ntheory import factorint


def gen_encoding_map(shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the encoding map and qubit sizes for a tensor of *shape*.

    Returns
    -------
    qubit_sizes : np.ndarray
        Physical qubit dimension per virtual site (product of factors).
    encoding_map : np.ndarray
        Map from physical coordinates → qubit indices
        with shape ``(num_levels, *shape)``.
    """
    block_sizes, prod_blocks = get_factorlist(shape)
    idx_grid = np.indices(shape)
    mapped = hierarchical_block_indexing(idx_grid, prod_blocks)

    enc_map = np.empty((len(block_sizes), *shape), dtype=int)
    for lvl in range(len(block_sizes)):
        enc_map[lvl] = np.ravel_multi_index(mapped[lvl], block_sizes[lvl])

    return np.prod(block_sizes, axis=1), enc_map.astype(int)


def balance_factors(factors: List[int], target_num: int) -> List[int]:
    """
    Reduce *factors* to *target_num* elements by repeatedly multiplying the two
    smallest factors.

    Parameters
    ----------
    factors
        List of prime factors for one dimension.
    target_num
        Desired final length of the factor list.

    Returns
    -------
    list[int]
        Balanced list with length == *target_num*.
    """
    factors = sorted(factors)  # ensure ascending
    while len(factors) > target_num:
        smallest = factors.pop(0)
        next_smallest = factors.pop(0)
        grouped = smallest * next_smallest

        # re-insert while keeping ascending order
        insert_at = 0
        while insert_at < len(factors) and factors[insert_at] < grouped:
            insert_at += 1
        factors.insert(insert_at, grouped)

    return factors


def get_factorlist(shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced prime factors for each tensor dimension and their cumulative
    products (used for hierarchical indexing).

    Parameters
    ----------
    shape
        Tensor shape.

    Returns
    -------
    factor_lists : np.ndarray
        Shape = (num_levels, ndim).  Each column holds factors for that dimension.
    prod_block_sizes : np.ndarray
        Cumulative products used by :func:`hierarchical_block_indexing`.
    """
    # Step 1 – prime-factor decomposition for each dimension
    factor_lists: List[List[int]] = []
    for dim in shape:
        if dim == 1:
            factors = [1]
        else:
            factors = []
            for prime, exp in sorted(factorint(dim).items()):
                factors.extend([prime] * exp)
        factor_lists.append(sorted(factors))

    # Step 2 – balance number of factors across dimensions
    min_len = min(len(f) for f in factor_lists)
    for i, factors in enumerate(factor_lists):
        if len(factors) > min_len:
            factor_lists[i] = balance_factors(factors, min_len)
        elif len(factors) < min_len:
            factors.extend([1] * (min_len - len(factors)))
            factor_lists[i] = sorted(factors)

    # Optional reversal every second dimension (snake pattern)
    for i, factors in enumerate(factor_lists):
        if i % 2 == 1:
            factor_lists[i] = factors[::-1]

    factor_arr = np.array(factor_lists).T  # shape → (levels, ndim)

    # Cumulative products for hierarchical indexing
    prod = np.ones((len(factor_arr) + 1, factor_arr.shape[1]), dtype=int)
    prod[1:-1] = np.cumprod(factor_arr[-1:0:-1], axis=0)[::-1]
    prod[0] *= 1_000_000  # sentinel for the top level

    return factor_arr, prod


def hierarchical_block_indexing(
    index: np.ndarray, prod_block_sizes: np.ndarray
) -> np.ndarray:
    """
    Map absolute indices → hierarchical block indices.

    Parameters
    ----------
    index
        `np.indices(shape)` style array.
    prod_block_sizes
        Output of :func:`get_factorlist`.

    Returns
    -------
    np.ndarray
        Hierarchical indices with shape
        ``(num_levels, …original index shape…)``.
    """
    # Reshape helpers for broadcast-compatible division / modulo
    n_levels, ndim = prod_block_sizes.shape
    reshape_tail = [1] * ndim

    numer = np.mod(
        index.reshape([1] + list(index.shape)),
        prod_block_sizes[:-1].reshape(list(prod_block_sizes[:-1].shape) + reshape_tail),
    )
    denom = prod_block_sizes[1:].reshape(
        list(prod_block_sizes[1:].shape) + reshape_tail
    )
    return np.floor(numer / denom).astype(int)
