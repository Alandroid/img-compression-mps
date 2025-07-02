import numpy as np
from imgcompressionmps.core.ndmps import NDMPS
import pytest
import math

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(2025)


@pytest.fixture(
    params=[
        (512, 680),            # 2-D image-like
        (8, 512, 680),         # small RGB block
       # (12, 8, 512, 680),      # 4-D tensor
    ],
    ids=lambda s: f"shape={s}",
)
def tensor(request, rng) -> np.ndarray:
    """Random real tensor of the parametrised shape."""
    return rng.random(request.param)


@pytest.fixture(params=["Std", "DCT"])
def mode(request):
    return request.param


@pytest.fixture
def ndmps_obj(tensor, mode) -> NDMPS:
    """NDMPS instance built from *tensor* in the requested mode."""
    return NDMPS.from_tensor(tensor, norm=False, mode=mode)


def test_roundtrip_exact(ndmps_obj, tensor):
    """from_tensor → to_tensor should be (nearly) lossless."""
    out = ndmps_obj.to_tensor()
    assert np.allclose(out, tensor, atol=1e-10), "Round-trip mismatch"


def test_norm_option(tensor):
    """If norm=True the stored norm_value should be ~1.0."""
    obj = NDMPS.from_tensor(tensor, norm=True)
    assert math.isclose(obj.norm_value, 1.0, rel_tol=1e-12)

def test_compression_reduces_elements(ndmps_obj):
    before = ndmps_obj.number_elements_in_MPS()
    ndmps_obj.compress(cutoff=0.1)
    after = ndmps_obj.number_elements_in_MPS()
    assert after < before, "Compress did not shrink MPS"


def test_boundary_and_norm_refresh(ndmps_obj):
    # Manually modify first tensor
    ndmps_obj.mps.arrays[0][:] *= 10
    ndmps_obj.update_boundary_list()
    ndmps_obj.update_norm()
    # min/max should reflect scaling
    new_min, new_max = ndmps_obj.boundary_list[0]
    assert new_min <= np.min(ndmps_obj.mps.arrays[0]) and new_max >= np.max(
        ndmps_obj.mps.arrays[0]
    )
    # norm_value matches explicit inner product
    assert math.isclose(
        ndmps_obj.norm_value**2, ndmps_obj.mps @ ndmps_obj.mps, rel_tol=1e-12
    )


def test_disk_compression_ratio(ndmps_obj):
    ndmps_obj.compress(cutoff=0.4) # I HAVE TO PUT THIS RELATIVELY HIGH... IS THIS OKAY??
    r = ndmps_obj.compression_ratio_on_disk(dtype=np.uint16, replace=False)
    assert 0 < r < 1, "Disk ratio should be strictly between 0 and 1"


def test_continuous_compress_prints(ndmps_obj, capsys):
    ndmps_obj.continuous_compress(cutoff=0.05, print_ratio=True)
    captured = capsys.readouterr().out
    # We asked for 20 steps, each prints a line with "Compression ratio at"
    assert captured.count("Compression ratio at") == 20
