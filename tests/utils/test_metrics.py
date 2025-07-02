import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

# =============================================================================
# Tests for compute_ssim_2d
# =============================================================================

def compute_ssim_2d(img1, img2, data_range=1.0):
    """Compute SSIM between two 2D images."""
    return ssim(img1, img2, data_range=data_range)

def test_compute_ssim_2d_perfect_match():
    """SSIM should be 1.0 for identical 2D images."""
    img = np.ones((32, 32), dtype=np.float32)
    result = compute_ssim_2d(img, img, data_range=1.0)
    assert np.isclose(result, 1.0)

def test_compute_ssim_2d_zeros():
    """SSIM should be 1.0 for two zero arrays."""
    img = np.zeros((32, 32), dtype=np.float32)
    result = compute_ssim_2d(img, img, data_range=1.0)
    assert np.isclose(result, 1.0)

def test_compute_ssim_2d_mismatched_shapes():
    """SSIM should raise ValueError for different shapes."""
    img1 = np.ones((32, 32), dtype=np.float32)
    img2 = np.ones((16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_ssim_2d(img1, img2)

# =============================================================================
# Tests for ssim_3d_axis
# =============================================================================

def ssim_3d_axis(img1, img2, axis=0, data_range=1.0):
    """Compute SSIM along a given axis in 3D arrays."""
    if axis >= img1.ndim:
        raise IndexError("Axis out of range")
    return np.mean([
        ssim(a, b, data_range=data_range)
        for a, b in zip(np.moveaxis(img1, axis, 0), np.moveaxis(img2, axis, 0))
    ])

def test_ssim_3d_axis_perfect_match():
    """SSIM should be 1.0 for identical 3D stacks."""
    img = np.ones((4, 32, 32), dtype=np.float32)
    result = ssim_3d_axis(img, img, axis=0, data_range=1.0)
    assert np.isclose(result, 1.0)

def test_ssim_3d_axis_invalid_axis():
    """Should raise IndexError for invalid axis."""
    img = np.ones((4, 32, 32), dtype=np.float32)
    with pytest.raises(IndexError):
        ssim_3d_axis(img, img, axis=3, data_range=1.0)

def test_ssim_3d_axis_upper_bound_axis():
    """SSIM should still work at upper valid axis."""
    # Axis 2 means slicing over 32x32
    img = np.ones((32, 32, 32), dtype=np.float32)
    result = ssim_3d_axis(img, img, axis=2, data_range=1.0)
    assert np.isclose(result, 1.0)

def test_ssim_3d_axis_negative_axis():
    """SSIM should support negative axis indexing."""
    # Axis -1 = axis 2, still safe if shape is (32, 32, 32)
    img = np.ones((32, 32, 32), dtype=np.float32)
    result = ssim_3d_axis(img, img, axis=-1, data_range=1.0)
    assert np.isclose(result, 1.0)

# =============================================================================
# Tests for avg_ssim_3d
# =============================================================================

def avg_ssim_3d(img1, img2, data_range=1.0):
    """Average SSIM over a 3D volume interpreted as slices."""
    if img1.shape != img2.shape:
        raise ValueError("Shape mismatch between 3D volumes.")
    return np.mean([
        ssim(a, b, data_range=data_range) for a, b in zip(img1, img2)
    ])

def test_avg_ssim_3d_perfect_match():
    """Should return 1.0 for identical volumes."""
    img = np.ones((4, 32, 32), dtype=np.float32)
    result = avg_ssim_3d(img, img, data_range=1.0)
    assert np.isclose(result, 1.0)

def test_avg_ssim_3d_low_contrast():
    """Should return value < 1.0 for distinct volumes."""
    img1 = np.ones((4, 32, 32), dtype=np.float32)
    img2 = np.zeros((4, 32, 32), dtype=np.float32)
    result = avg_ssim_3d(img1, img2, data_range=1.0)
    assert 0 <= result < 1.0

def test_avg_ssim_3d_mismatched_shapes():
    """Should raise ValueError for different volume shapes."""
    img1 = np.ones((4, 32, 32), dtype=np.float32)
    img2 = np.ones((5, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        avg_ssim_3d(img1, img2, data_range=1.0)

# =============================================================================
# Tests for avg_ssim_4d
# =============================================================================

def avg_ssim_4d(img1, img2, data_range=1.0):
    """Average SSIM over a 4D tensor (batch of volumes)."""
    if img1.shape != img2.shape:
        raise ValueError("Shape mismatch between 4D volumes.")
    return np.mean([
        avg_ssim_3d(a, b, data_range=data_range) for a, b in zip(img1, img2)
    ])

def test_avg_ssim_4d_perfect_match():
    """Should return 1.0 for identical 4D tensors."""
    img = np.ones((2, 4, 32, 32), dtype=np.float32)
    result = avg_ssim_4d(img, img, data_range=1.0)
    assert np.isclose(result, 1.0)

def test_avg_ssim_4d_mismatched_volume_count():
    """Should raise ValueError for different number of 3D blocks."""
    img1 = np.ones((2, 4, 32, 32), dtype=np.float32)
    img2 = np.ones((2, 5, 32, 32), dtype=np.float32)  # mismatch in time dim
    with pytest.raises(ValueError):
        avg_ssim_4d(img1, img2, data_range=1.0)

# =============================================================================
# Tests for compute_mean_std
# =============================================================================

def compute_mean_std(data, num_common_points):
    """Compute mean and std of SSIM over last N compression ratios."""
    cr = np.concatenate(data["compressionratio_list_disk"])
    ssim_vals = np.concatenate(data["ssim_list"])

    if len(cr) < num_common_points:
        return np.nan, np.nan, np.full(num_common_points, np.nan)

    indices = np.argsort(cr)[-num_common_points:]
    return np.mean(ssim_vals[indices]), np.std(ssim_vals[indices]), cr[indices]

def test_compute_mean_std_prime_shape_filtered():
    """Return finite values even for prime-only shapes."""
    d = {
        "compressionratio_list_disk": [np.array([0.25, 0.5, 1.0])],
        "ssim_list": [np.array([0.8, 0.85, 0.9])],
        "shapes": [(3, 5)],
    }
    mean, std, common = compute_mean_std(d, num_common_points=3)
    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert np.all(np.isfinite(common))

def test_compute_mean_std_too_few_points():
    """Return NaNs if not enough points for sampling."""
    d = {
        "compressionratio_list_disk": [np.array([0.1])],
        "ssim_list": [np.array([0.9])],
        "shapes": [(4, 4)],
    }
    mean, std, common = compute_mean_std(d, num_common_points=3)
    assert np.isnan(mean)
    assert np.isnan(std)
    assert np.all(np.isnan(common))

def test_compute_mean_std_zero_variance():
    """Standard deviation should be 0 if all values are the same."""
    d = {
        "compressionratio_list_disk": [np.array([0.25, 0.5, 1.0])],
        "ssim_list": [np.array([0.8, 0.8, 0.8])],
        "shapes": [(4, 4)],
    }
    mean, std, common = compute_mean_std(d, num_common_points=3)
    assert np.isclose(std, 0.0)
