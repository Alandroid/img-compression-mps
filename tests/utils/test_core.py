import numpy as np
import pytest

# If you're testing locally, replace this import with the correct relative path.
from imgcompressionmps.utils.core import (
    balance_factors,
    get_factorlist,
    gen_encoding_map,
    hierarchical_block_indexing
)

# ---------------------- balance_factors Tests ----------------------

def test_balance_factors_preserves_product():
    """Ensure product of factors remains unchanged after balancing."""
    original = [2, 2, 3, 3]
    result = balance_factors(original.copy(), 2)
    assert np.prod(result) == np.prod(original)
    assert len(result) == 2

def test_balance_factors_no_change_if_target_equal():
    """Return unchanged if target equals current length."""
    original = [2, 3]
    result = balance_factors(original.copy(), 2)
    assert result == original

def test_balance_factors_returns_sorted():
    """Balanced list should be sorted."""
    result = balance_factors([7, 3, 2, 2], 3)
    assert result == sorted(result)

def test_balance_factors_with_ones():
    """Handle list of all ones correctly."""
    result = balance_factors([1, 1, 1, 1], 2)
    assert result == [1, 1]

def test_balance_factors_large_input():
    """Handle compression of large uniform input."""
    original = [2] * 10
    result = balance_factors(original.copy(), 2)
    assert len(result) == 2
    assert np.prod(result) == 1024


def test_balance_factors_empty_input():
    """Empty input with zero target returns empty list."""
    result = balance_factors([], 0)
    assert result == []

def test_balance_factors_invalid_target():
    """Negative target should raise error."""
    with pytest.raises(ValueError):
        balance_factors([2, 3], -1)

def test_balance_factors_target_zero_nonempty_input():
    """Target zero with nonempty input should raise error."""
    with pytest.raises(ValueError):
        balance_factors([2, 3], 0)

def test_balance_factors_identity_case():
    """Single element should return same if target is 1."""
    result = balance_factors([6], 1)
    assert result == [6]

# ---------------------- get_factorlist Tests ----------------------

def test_get_factorlist_shapes_match():
    shape = (30, 24)
    factors, prods = get_factorlist(shape)
    assert factors.shape[1] == len(shape)
    assert prods.shape[1] == len(shape)
    assert prods.shape[0] == factors.shape[0] + 1
    assert np.all(np.prod(factors, axis=0) == (30,24)) == True # checks if all factors multiplied give again the same shape

def test_synthetic_data_get_factorlist():
    shape = (256,128)
    factors, prods = get_factorlist(shape)
    true_factors = np.array([[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[2, 2],[4, 2]])
    true_prods = np.array([[9223372036854775807, 9223372036854775807],
       [                128,                  64],
       [                 64,                  32],
       [                 32,                  16],
       [                 16,                   8],
       [                  8,                   4],
       [                  4,                   2],
       [                  1,                   1]])
    assert np.array_equal(factors, true_factors)
    assert np.array_equal(prods, true_prods)

def test_get_factorlist_with_ones():
    factors, prods = get_factorlist((1, 1))
    assert np.all(factors == 1)
    assert np.all(prods >= 1)

def test_get_factorlist_snake_pattern():
    shape = (4, 6)
    factors, _ = get_factorlist(shape)
    # Last column should decrease (snake down)
    assert np.all(np.diff(factors[:, 1])[::-1] <= 0)

def test_get_factorlist_single_dim():
    shape = (8,)
    factors, prods = get_factorlist(shape)
    assert factors.shape[1] == 1
    assert prods.shape[1] == 1

def test_get_factorlist_with_zero_dim():
    with pytest.raises(ValueError):
        get_factorlist((0, 4))

def test_get_factorlist_high_dimensional():
    shape = (30, 40, 50)
    factors, prods = get_factorlist(shape)
    true_factors = np.array([[2, 5, 2],
        [3, 4, 5],
        [5, 2, 5]])
    true_prods = np.array([[9223372036854775807, 9223372036854775807, 9223372036854775807],
        [                 15,                   8,                  25],
        [                  5,                   2,                   5],
        [                  1,                   1,                   1]])
    assert np.array_equal(factors, true_factors)
    assert np.array_equal(prods, true_prods)
    

# ---------------------- gen_encoding_map Tests ----------------------

def test_gen_encoding_map_output_shapes():
    shape = (3, 3)
    qsize, encmap = gen_encoding_map(shape)
    assert encmap.shape[1:] == shape
    assert len(qsize) == encmap.shape[0]

def test_gen_encoding_map_value_bounds():
    shape = (2, 2)
    _, encmap = gen_encoding_map(shape)
    assert np.all(encmap >= 0)

def test_gen_encoding_map_dim_one():
    shape = (1, 4)
    qsize, encmap = gen_encoding_map(shape)
    assert encmap.shape == (len(qsize), 1, 4)

def test_gen_encoding_map_empty_shape():
    with pytest.raises(ValueError):
        gen_encoding_map(())

def test_gen_encoding_map_invalid_shape_type():
    with pytest.raises(ValueError):
        gen_encoding_map(("a", "b"))

def test_synthetic_gen_encoding_map():
    shape = (8, 9)
    qubit_sizes, encoding_map = gen_encoding_map(shape)
    true_qubit_sizes = np.array([6, 12])
    true_encoding_map = np.array([[[ 0,  0,  0,  1,  1,  1,  2,  2,  2],
         [ 0,  0,  0,  1,  1,  1,  2,  2,  2],
         [ 0,  0,  0,  1,  1,  1,  2,  2,  2],
         [ 0,  0,  0,  1,  1,  1,  2,  2,  2],
         [ 3,  3,  3,  4,  4,  4,  5,  5,  5],
         [ 3,  3,  3,  4,  4,  4,  5,  5,  5],
         [ 3,  3,  3,  4,  4,  4,  5,  5,  5],
         [ 3,  3,  3,  4,  4,  4,  5,  5,  5]],
 
        [[ 0,  1,  2,  0,  1,  2,  0,  1,  2],
         [ 3,  4,  5,  3,  4,  5,  3,  4,  5],
         [ 6,  7,  8,  6,  7,  8,  6,  7,  8],
         [ 9, 10, 11,  9, 10, 11,  9, 10, 11],
         [ 0,  1,  2,  0,  1,  2,  0,  1,  2],
         [ 3,  4,  5,  3,  4,  5,  3,  4,  5],
         [ 6,  7,  8,  6,  7,  8,  6,  7,  8],
         [ 9, 10, 11,  9, 10, 11,  9, 10, 11]]])
    assert np.array_equal(qubit_sizes, true_qubit_sizes)
    assert np.array_equal(encoding_map, true_encoding_map)

# ---------------------- hierarchical_block_indexing Tests ----------------------

def test_hierarchical_block_indexing_shapes():
    shape = (4, 6)
    idx = np.indices(shape)
    _, prods = get_factorlist(shape)
    result = hierarchical_block_indexing(idx, prods)
    assert result.shape == (prods.shape[0] - 1, len(shape), *shape)

def test_hierarchical_block_indexing_trivial_case():
    shape = (1, 1)
    idx = np.indices(shape)
    _, prods = get_factorlist(shape)
    result = hierarchical_block_indexing(idx, prods)
    assert np.all(result == 0)

def test_hierarchical_block_indexing_large_shape():
    shape = (8, 4)
    idx = np.indices(shape)
    _, prods = get_factorlist(shape)
    result = hierarchical_block_indexing(idx, prods)
    assert result.shape == (prods.shape[0] - 1, len(shape), *shape)

def test_hierarchical_block_indexing_mismatched_prods():
    shape = (4, 4)
    idx = np.indices(shape)
    wrong_prods = np.array([[1, 1]])
    with pytest.raises(ValueError):
        hierarchical_block_indexing(idx, wrong_prods)

def test_hierarchical_block_indexing_3d_tensor():
    shape = (2, 2, 2)
    idx = np.indices(shape)
    _, prods = get_factorlist(shape)
    result = hierarchical_block_indexing(idx, prods)
    assert result.shape == (prods.shape[0] - 1, len(shape), *shape)

def test_synthetic_hierarchical_block_indexing():
    shape = (4, 6)
    idx = np.indices(shape)
    _, prods = get_factorlist(shape)
    result = hierarchical_block_indexing(idx, prods)
    true_result = np.array([[[[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1]],

        [[0, 0, 1, 1, 2, 2],
         [0, 0, 1, 1, 2, 2],
         [0, 0, 1, 1, 2, 2],
         [0, 0, 1, 1, 2, 2]]],


       [[[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1]],

        [[0, 1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1]]]])
    assert np.array_equal(result, true_result)