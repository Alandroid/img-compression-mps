import numpy as np
from typing import List
from skimage.metrics import structural_similarity as ssim
from scipy.interpolate import interp1d
from sympy import isprime

# Vectorized version of isprime for performance
vectorized_isprime = np.vectorize(isprime)


def compute_ssim_2d(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute SSIM (Structural Similarity Index) for two 2D arrays.

    Args:
        original: Ground truth image.
        compressed: Compressed or reconstructed image.

    Returns:
        SSIM value (float) between original and compressed images.
    """
    original = np.asarray(original)
    compressed = np.clip(np.asarray(compressed), 0, None)
    data_range = max(original.max(), compressed.max()) - min(original.min(), compressed.min())

    # Ensure small arrays can be processed
    min_dim = min(original.shape)
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1

    return ssim(original, compressed, data_range=data_range, win_size=win_size)


def ssim_3d_axis(
    original: np.ndarray, compressed: np.ndarray, axis: int = 0
) -> List[float]:
    """
    Compute slice-wise SSIM scores along a given axis of a 3D volume.

    Args:
        original: Ground truth 3D tensor.
        compressed: Compressed or reconstructed 3D tensor.
        axis: Axis along which to slice (0, 1, or 2).

    Returns:
        List of SSIM values for each 2D slice along the selected axis.
    """
    if original.shape != compressed.shape:
        raise ValueError("Shape mismatch between 3D arrays.")
    if axis >= original.ndim or axis < -original.ndim:
        raise ValueError(f"Invalid axis {axis} for 3D SSIM.")

    comp = np.clip(compressed, 0, None)
    scores = []

    for i in range(original.shape[axis]):
        if axis == 0:
            scores.append(compute_ssim_2d(original[i], comp[i]))
        elif axis == 1:
            scores.append(compute_ssim_2d(original[:, i, :], comp[:, i, :]))
        elif axis == 2:
            scores.append(compute_ssim_2d(original[:, :, i], comp[:, :, i]))

    return scores


def avg_ssim_3d(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute average SSIM across all three axes for a 3D tensor.

    Args:
        original: Ground truth 3D tensor.
        compressed: Compressed or reconstructed 3D tensor.

    Returns:
        Mean SSIM over axis-wise SSIMs.
    """
    if original.shape != compressed.shape:
        raise ValueError("Shape mismatch between 3D volumes.")

    return np.mean([
        np.mean(ssim_3d_axis(original, compressed, axis))
        for axis in range(3)
    ])


def avg_ssim_4d(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute average SSIM over all time frames in a 4D tensor.

    Args:
        original: Ground truth 4D tensor (e.g., fMRI data).
        compressed: Compressed 4D tensor.

    Returns:
        Mean SSIM across all frames.
    """
    if original.shape != compressed.shape:
        raise ValueError("Shape mismatch between 4D volumes.")

    return np.mean([
        avg_ssim_3d(original[..., t], compressed[..., t])
        for t in range(original.shape[-1])
    ])


def compute_ssim_by_dim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute SSIM for 2D, 3D, or 4D arrays by dispatching to the appropriate function.

    Args:
        a: Ground truth tensor.
        b: Compressed or reconstructed tensor.

    Returns:
        SSIM score.

    Raises:
        ValueError: If the tensor dimensionality is not supported.
    """
    if a.ndim == 4:
        return avg_ssim_4d(a, b)
    elif a.ndim == 3:
        return avg_ssim_3d(a, b)
    elif a.ndim == 2:
        return compute_ssim_2d(a, b)
    else:
        raise ValueError(f"Unsupported tensor dimension for SSIM: {a.ndim}")


def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two arrays.

    Args:
        original: Ground truth tensor.
        compressed: Compressed or reconstructed tensor.

    Returns:
        PSNR value in dB.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10((np.max(original) ** 2) / mse)


def compute_overlap(mps1, mps2) -> float:
    """
    Compute normalized overlap (fidelity) between two MPS states.

    Args:
        mps1: First MPS object (with `.mps` and `.norm_value` attributes).
        mps2: Second MPS object.

    Returns:
        Overlap fidelity between normalized states.
    """
    return (mps1.mps @ mps2.mps) / (mps1.norm_value * mps2.norm_value)


def compute_mean_std(
    dict, num_common_points, key_x="compressionratio_list_disk", key_y="ssim_list"
):
    """
    Interpolates SSIM values across a common compression factor grid and returns
    the mean and std across samples that are not excluded for having only prime-shaped inputs.

    Args:
        dict: Dictionary containing compression ratios, SSIMs, and input shapes.
        num_common_points: Number of interpolation points for alignment.
        key_x: Key name for compression ratio list in dict.
        key_y: Key name for SSIM list in dict.

    Returns:
        mean: Mean SSIM curve across valid shapes.
        std: Std deviation SSIM curve across valid shapes.
        common_comp_facs: Interpolated compression factor grid.
    """
    values_x = np.array(dict[key_x])
    values_y = np.array(dict[key_y])
    shapes = np.array(dict["shapes"])

    max_common_fac = np.min(1 / values_x[:, -1])
    min_common_fac = np.max(1 / values_x[:, 0])
    common_comp_facs = np.linspace(min_common_fac, max_common_fac, num_common_points)

    interpolated_ssim = []
    for x, y, s in zip(1 / values_x, values_y, shapes):
        if not np.all(vectorized_isprime(s)):
            interp_func = interp1d(x, y, kind="linear", bounds_error=False)
            interpolated_ssim.append(interp_func(common_comp_facs))

    if len(interpolated_ssim) == 0:
        return np.nan, np.nan, common_comp_facs

    return (
        np.mean(interpolated_ssim, axis=0),
        np.std(interpolated_ssim, axis=0),
        np.array(common_comp_facs),
    )