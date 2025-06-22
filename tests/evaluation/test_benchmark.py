# tests/evaluation/test_benchmark.py

"""
Full unit test coverage for imgcompressionmps.evaluation.benchmark
"""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Correct import based on folder structure
bm = importlib.import_module("imgcompressionmps.evaluation.benchmark")

# --------------------------------------------------------------------------- #
#                                GLOBAL PATCHES                               #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _patch_benchmark_module(monkeypatch):
    """Stubs all external dependencies for fast testing."""
    monkeypatch.setattr(
        bm,
        "get_num_bits",
        lambda dtype: int(np.dtype(dtype).itemsize * 8),
        raising=True,
    )
    monkeypatch.setattr(
        bm,
        "compute_ssim_by_dim",
        lambda a, b: 1.0,
        raising=True,
    )
    monkeypatch.setattr(
        bm,
        "compute_psnr",
        lambda a, b: 42.0,
        raising=True,
    )
    monkeypatch.setattr(
        bm,
        "compute_overlap",
        lambda mps, ref: 0.999,
        raising=True,
    )
    monkeypatch.setattr(
        bm,
        "get_shapes",
        lambda tensors: [t.shape for t in tensors],
        raising=True,
    )
    monkeypatch.setattr(
        bm,
        "mri_to_slices",
        lambda tensors, bits: (tensors, bits),
        raising=True,
    )

    class _FakeNDMPS:
        def __init__(self, data):
            self._data = np.asarray(data)
            self._compressed = False

        @classmethod
        def from_tensor(cls, tensor, norm=False, mode="DCT"):
            return cls(tensor)

        def to_tensor(self):
            return self._data.copy()

        def compress(self, _factors):
            self._compressed = True

        def compression_ratio(self):
            return 0.5 if self._compressed else 1.0

        def get_storage_space(self, dtype=np.uint16):
            factor = 2 if self._compressed else 1
            return int(self._data.size * np.dtype(dtype).itemsize / factor)

        def get_bytesize_on_disk(self, dtype=np.uint16):
            return self.get_storage_space(dtype) // 2

        def compression_ratio_on_disk(self, dtype=np.uint16, replace=True):
            return 0.25 if self._compressed else 1.0

        def bond_sizes(self):
            return [1, 2, 3]

    monkeypatch.setattr(bm, "NDMPS", _FakeNDMPS, raising=True)

    class _FakeHeader:
        def __init__(self, dtype):
            self._dtype = dtype

        def get_data_dtype(self):
            return self._dtype

    class _FakeNibImg:
        def __init__(self, data):
            self._data = data
            self.header = _FakeHeader(self._data.dtype)

        def get_fdata(self):
            return self._data

    def _fake_nib_load(_path):
        dummy = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        return _FakeNibImg(dummy)

    monkeypatch.setattr(bm.nib, "load", _fake_nib_load, raising=True)

    yield


# --------------------------------------------------------------------------- #
#                             INDIVIDUAL TEST CASES                           #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_array():
    return np.arange(24, dtype=np.uint8).reshape(2, 3, 4)


def test_load_tensors_npz(tmp_path, sample_array):
    f = tmp_path / "a.npz"
    np.savez_compressed(f, sequence=sample_array)
    tensors, bits = bm.load_tensors([str(f)], ".npz")
    assert np.array_equal(tensors[0], sample_array)
    assert bits == [8]


def test_load_tensors_npz_with_shape(tmp_path, sample_array):
    f = tmp_path / "b.npz"
    np.savez_compressed(f, sequence=sample_array)
    tensors, _ = bm.load_tensors([str(f)], ".npz", shape=(1, 2, 2))
    assert tensors[0].shape == sample_array.shape


def test_load_tensors_invalid_suffix():
    with pytest.raises(ValueError):
        bm.load_tensors(["dummy.foo"], ".foo")


def test_load_tensors_gz():
    tensors, bits = bm.load_tensors(["fake.nii.gz"], ".gz")
    assert tensors[0].shape == (2, 2, 2)
    assert bits == [32]


def test_conv_roundtrip(sample_array):
    tensors = [sample_array]
    mps_list = bm.conv_to_mps(tensors)
    back = bm.conv_to_tensors(mps_list)
    assert np.array_equal(tensors[0], back[0])


def test_compress_list_changes_state(sample_array):
    mps = bm.conv_to_mps([sample_array])[0]
    assert mps.compression_ratio() == 1.0
    bm.compress_list([mps], 0.5)
    assert mps.compression_ratio() == 0.5


@pytest.mark.parametrize("metric", [
    "compression_ratio", "storage", "gzip_bytes", "gzip_ratio",
    "ssim", "psnr", "bond_dims", "shape", "fidelity"
])
def test_benchmark_metric_all(metric, sample_array):
    mps = bm.conv_to_mps([sample_array])
    ref = [sample_array]
    result = bm.benchmark_metric(mps, ref, metric=metric)
    assert len(result) == 1


def test_benchmark_metric_invalid(sample_array):
    mps = bm.conv_to_mps([sample_array])
    with pytest.raises(ValueError):
        bm.benchmark_metric(mps, metric="invalid_metric")


def test_run_benchmark_shapes(sample_array):
    mps_list = bm.conv_to_mps([sample_array, sample_array + 1])
    cutoffs = np.array([0.8, 0.5])
    res = bm.run_benchmark(mps_list, [sample_array, sample_array + 1], cutoffs)
    assert set(res.keys()).issuperset({"ssim", "compression_ratio", "psnr", "fidelity", "bond_dims"})
    assert res["ssim"].shape == (2, 3)  # 2 files, 3 cutoffs (incl. no compression)


def test_load_tensors_empty_list():
    tensors, bits = bm.load_tensors([], ".npz")
    assert tensors == []
    assert bits == []

def test_conv_to_mps_empty_list():
    assert bm.conv_to_mps([]) == []

def test_conv_to_tensors_empty_list():
    assert bm.conv_to_tensors([]) == []

def test_compress_list_invalid_input(sample_array):
    mps = bm.conv_to_mps([sample_array])[0]
    with pytest.raises(Exception):
        bm.compress_list([mps], None)

def test_benchmark_metric_length_mismatch(sample_array):
    mps_list = bm.conv_to_mps([sample_array])
    refs = [sample_array, sample_array + 1]  # Too many refs
    with pytest.raises(IndexError):
        bm.benchmark_metric(mps_list, refs, metric="ssim")

def test_run_benchmark_empty_lists():
    cutoffs = np.array([0.5])
    res = bm.run_benchmark([], [], cutoffs)
    for v in res.values():
        assert v == []

def test_run_benchmark_empty_cutoffs(sample_array):
    mps_list = bm.conv_to_mps([sample_array])
    res = bm.run_benchmark(mps_list, [sample_array], [])
    for k in ("ssim", "compression_ratio", "psnr", "fidelity", "bond_dims"):
        assert k in res
        assert isinstance(res[k], (list, np.ndarray))

def test_run_full_benchmark_invalid_path(tmp_path):
    with pytest.raises(Exception):
        bm.run_full_benchmark(
            dataset_path=tmp_path / "nonexistent",
            cutoff_list=np.array([0.5]),
            result_file="should_fail.json",
            datatype="MRI",
            ending=".npz"
        )

def test_run_full_benchmark_datatype_modes(tmp_path, monkeypatch, sample_array):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    f = data_dir / "scan.npz"
    np.savez_compressed(f, sequence=sample_array)

    monkeypatch.setattr(bm, "find_specific_files", lambda _, __: [str(f)], raising=True)
    monkeypatch.chdir(tmp_path)

    bm.run_full_benchmark(
        dataset_path=data_dir,
        cutoff_list=np.array([0.9]),
        result_file="mri.json",
        datatype="MRI",
        ending=".npz"
    )

    out = Path("src/evaluation/results/mri.json")
    assert out.exists()


# ---------------------- Specific Output Value Checks ----------------------- #
def test_benchmark_metric_output_values(sample_array):
    mps = bm.conv_to_mps([sample_array])
    ref = [sample_array]
    out = bm.benchmark_metric(mps, ref, metric="ssim")
    assert out == [1.0]  # from monkeypatch

    out = bm.benchmark_metric(mps, ref, metric="psnr")
    assert out == [42.0]  # from monkeypatch

    out = bm.benchmark_metric(mps, ref, metric="fidelity")
    assert out == [0.999]  # from monkeypatch

    out = bm.benchmark_metric(mps, ref, metric="compression_ratio")
    assert out == [1.0]

    bm.compress_list(mps, 0.5)
    out = bm.benchmark_metric(mps, ref, metric="compression_ratio")
    assert out == [0.5]


def test_conv_to_mps_mode_passthrough(sample_array):
    # Just check no crash when passing another mode
    bm.NDMPS.from_tensor(sample_array, mode="None")


# ------------------------ run_benchmark Order Tests ------------------------ #
def test_run_benchmark_unsorted_cutoffs(sample_array):
    mps_list = bm.conv_to_mps([sample_array])
    cutoffs = np.array([0.9, 0.2, 0.5])  # intentionally unordered
    results = bm.run_benchmark(mps_list, [sample_array], cutoffs)
    assert all(key in results for key in ["ssim", "compression_ratio"])
    assert results["ssim"].shape == (1, 4)


# ----------------------- result_file Path Creation ------------------------- #
def test_run_full_benchmark_nested_path(tmp_path, monkeypatch, sample_array):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    f = data_dir / "scan1.npz"
    np.savez_compressed(f, sequence=sample_array)

    monkeypatch.setattr(bm, "find_specific_files", lambda root, ending: [str(f)], raising=True)
    monkeypatch.chdir(tmp_path)

    nested_path = Path("src/evaluation/results/sub/folder/output.json")

    bm.run_full_benchmark(
        dataset_path=data_dir,
        cutoff_list=np.array([0.8]),
        result_file=str(nested_path),
        datatype="MRI_Slice",
        ending=".npz",
    )

    assert nested_path.exists()

    
def test_run_full_benchmark_end_to_end(tmp_path, monkeypatch, sample_array):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    f = data_dir / "scan1.npz"
    np.savez_compressed(f, sequence=sample_array)

    monkeypatch.setattr(
        bm,
        "find_specific_files",
        lambda root, ending: [str(f)],
        raising=True,
    )

    monkeypatch.chdir(tmp_path)

    cutoffs = np.array([0.7])
    bm.run_full_benchmark(
        dataset_path=str(data_dir),
        cutoff_list=cutoffs,
        result_file="result.json",
        datatype="MRI_Slice",
        mode="DCT",
        ending=".npz",
    )

    res_file = Path("src/evaluation/results/result.json")
    assert res_file.exists()

    with res_file.open() as f:
        results = json.load(f)

    assert "ssim" in results
    assert results["cutoff_list"] == [0.7]