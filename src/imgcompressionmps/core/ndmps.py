import io
import gzip
import numpy as np
import quimb.tensor as qtn
from scipy.fftpack import dct, idct

from imgcompressionmps.utils.core import gen_encoding_map
from imgcompressionmps.utils.filetools import get_num_bits, scale_to_dtype, scale_back


class NDMPS:
    """
    Class for storing and compressing N-dimensional tensors using MPS.
    """

    def __init__(
        self,
        mps: qtn.MatrixProductState = None,
        qubit_size: tuple = None,
        encoding_map: np.ndarray = None,
        boundary_list: list = None,
        norm: bool = True,
        norm_value=None,
        mode: str = "Std",
        dim: int = None,
    ):
        self.qubit_size = qubit_size
        self.encoding_map = encoding_map
        self.mps = mps
        self.dim = dim
        self.norm = norm
        self.norm_value = norm_value
        self.mode = mode
        self.boundary_list = np.array(boundary_list)

    @classmethod
    def from_tensor(
        cls, tensor: np.ndarray, norm: bool = False, mode: str = "Std"
    ) -> "NDMPS":
        """
        Create an NDMPS instance from a tensor with encoding and optional normalization.

        Parameters
        ----------
        tensor : np.ndarray
            Input tensor.
        norm : bool
            Normalize the input tensor by L2 norm.
        mode : str
            "Std" for raw encoding or "DCT" for DCT-based preprocessing.

        Returns
        -------
        NDMPS
        """
        tensor = tensor.astype(np.float64)
        qubit_size, encoding_map = gen_encoding_map(tensor.shape)
        encoding_map = np.moveaxis(encoding_map, 0, -1)

        if norm:
            tensor /= np.linalg.norm(tensor)
        if mode == "DCT":
            tensor = dct(tensor, norm="ortho")

        # Map data into encoding structure
        contracted_tensor = np.empty(shape=tuple(qubit_size), dtype=tensor.dtype)
        k = encoding_map.shape[-1]
        flat_data = tensor.flatten()
        flat_new_indices = encoding_map.reshape(-1, k).astype(int)
        indices = tuple(flat_new_indices[:, dim] for dim in range(k))
        contracted_tensor[indices] = flat_data

        # Construct MPS
        mps = qtn.MatrixProductState.from_dense(contracted_tensor, dims=tuple(qubit_size))
        boundary_list = [[np.min(arr), np.max(arr)] for arr in mps.arrays]
        norm_value = np.sqrt(mps @ mps)

        return cls(mps, qubit_size, encoding_map, boundary_list, norm, norm_value, mode, tensor.ndim)

    def update_boundary_list(self):
        """Recompute min/max boundaries for each MPS tensor."""
        self.boundary_list = np.array([[np.min(t), np.max(t)] for t in self.mps.arrays])

    def update_norm(self):
        """Update stored norm of the current MPS."""
        self.norm_value = np.sqrt(self.mps @ self.mps)

    def compression_ratio(self):
        """Compute compression ratio: MPS elements / original tensor elements."""
        initial_N = np.prod(self.qubit_size)
        compressed_N = self.number_elements_in_MPS()
        return compressed_N / initial_N

    def compress(self, cutoff: float):
        """
        Compress MPS by truncating bonds with a relative cutoff.

        Parameters
        ----------
        cutoff : float
            Relative singular value cutoff.
        """
        for i in range(1, len(self.mps.sites)):
            t1 = self.mps[i - 1]
            t2 = self.mps[i]
            qtn.tensor_compress_bond(t1, t2, cutoff=cutoff, cutoff_mode="rel")
        self.update_boundary_list()
        self.update_norm()

    def continuous_compress(self, cutoff: float, print_ratio: bool = True):
        """
        Apply compression across a range of cutoff values.

        Parameters
        ----------
        cutoff : float
            Max cutoff to sweep over.
        print_ratio : bool
            Whether to print compression ratio at each step.
        """
        compress_list = np.linspace(0, 1, 20) * cutoff
        for c in compress_list:
            self.compress(c)
            if print_ratio:
                print(f"Compression ratio at {c}: {self.compression_ratio()}")

    def number_elements_in_MPS(self) -> int:
        """Return the total number of elements in all MPS tensors."""
        return sum(t.size for t in self.mps)

    def to_tensor(self) -> np.ndarray:
        """
        Convert MPS back to tensor format (with optional inverse DCT).

        Returns
        -------
        np.ndarray
            Reconstructed tensor.
        """
        contracted = self.mps ^ ...
        for i in range(len(contracted.inds)):
            contracted.moveindex("k" + str(i), i, inplace=True)

        k = self.encoding_map.shape[-1]
        recovered = np.empty(self.encoding_map.shape)
        recovered = contracted.data[
            tuple(self.encoding_map[..., dim] for dim in range(k))
        ]

        if self.mode == "Std":
            return recovered
        if self.mode == "DCT":
            return idct(recovered, norm="ortho")

    def show(self):
        """Display the MPS tensor network diagram."""
        self.mps.show()

    def bond_sizes(self):
        """Return the bond dimensions of the MPS."""
        return self.mps.bond_sizes()

    def replace_tensordata(self, tensorlist: list[np.ndarray]):
        """
        Replace internal tensors in the MPS with externally provided ones.

        Parameters
        ----------
        tensorlist : list of np.ndarray
            Replacement tensor list.
        """
        for i in range(len(self.mps.arrays)):
            assert self.mps.arrays[i].shape == tensorlist[i].shape
            self.mps.arrays[i][:] = tensorlist[i]
        self.update_boundary_list()
        self.update_norm()

    def return_tensors_data(self) -> list[np.ndarray]:
        """Return internal MPS tensor list."""
        return [t for t in self.mps.arrays]

    def compress_to_dtype(
        self, dtype=np.uint16, replace: bool = False
    ) -> list[np.ndarray]:
        """
        Integer-truncate each MPS tensor to given dtype.

        Parameters
        ----------
        dtype : np.dtype
            Unsigned int type (e.g. np.uint8, np.uint16).
        replace : bool
            If True, replaces original data with truncated version.

        Returns
        -------
        list of np.ndarray
            List of dtype-cast tensors (scaled to original range).
        """
        tensor_int_list = [scale_to_dtype(t, dtype) for t in self.mps.arrays]
        scaled_back = [
            scale_back(t, b[0], b[1], dtype)
            for t, b in zip(tensor_int_list, self.boundary_list)
        ]
        if replace:
            self.replace_tensordata(scaled_back)
        return tensor_int_list

    def get_bytesize_on_disk(self, dtype=np.uint16, replace: bool = False) -> int:
        """
        Estimate gzipped bytesize of MPS (optionally after dtype compression).

        Parameters
        ----------
        dtype : np.dtype
            Unsigned int type for truncation.
        replace : bool
            If True, apply truncation in-place.

        Returns
        -------
        int
            Total bytes after gzip compression.
        """
        tensor_int_list = self.compress_to_dtype(dtype, replace)
        total_bytes = 0

        for arr in tensor_int_list:
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(arr.tobytes())
            total_bytes += len(buf.getvalue())

        return total_bytes

    def compression_ratio_on_disk(
        self, dtype=np.uint16, replace: bool = False
    ) -> float:
        """
        Compute ratio: compressed size (gzipped) / uncompressed original size.

        Parameters
        ----------
        dtype : np.dtype
            Target dtype (e.g., np.uint16).
        replace : bool
            Whether to apply dtype conversion in-place.

        Returns
        -------
        float
            Disk compression ratio.
        """
        original_size = np.prod(self.qubit_size) * get_num_bits(dtype) / 8.
        compressed_size = self.get_bytesize_on_disk(dtype, replace)
        return compressed_size / original_size

    def get_storage_space(self, dtype=np.uint16, verbose: bool = False) -> float:
        """
        Estimate uncompressed storage in bytes using given dtype.

        Parameters
        ----------
        dtype : np.dtype
            Target dtype.
        verbose : bool
            If True, prints storage in KB.

        Returns
        -------
        float
            Storage in bytes.
        """
        size_bytes = self.number_elements_in_MPS() * get_num_bits(dtype) / 8
        if verbose:
            print(f"The storage space is approximately: {size_bytes / 1024:.2f} KB")
        return size_bytes