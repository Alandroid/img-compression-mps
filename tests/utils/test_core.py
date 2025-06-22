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

def test_balance_factors_target_larger_than_input():
    """Extend list correctly when target length > input."""
    result = balance_factors([2, 3], 5)
    assert np.prod(result) == 6
    assert len(result) == 5

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
    shape = (6, 9)
    factors, prods = get_factorlist(shape)
    assert factors.shape[1] == len(shape)
    assert prods.shape[1] == len(shape)
    assert prods.shape[0] == factors.shape[0] + 1

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
    shape = (2, 3, 4)
    factors, prods = get_factorlist(shape)
    assert factors.shape[1] == 3
    assert prods.shape[1] == 3

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