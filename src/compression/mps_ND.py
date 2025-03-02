import os
import sys
# Get the absolute path of the current script
current_path = os.path.abspath(__file__)
project_folder = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.append(project_folder)
import numpy as np
import quimb.tensor as qtn
from scipy.fftpack import dct, idct

from compression.utils_ND import *

class NDMPS:
    def __init__(self, mps: qtn.MatrixProductState = None, qubit_size: tuple = None, encoding_map: np.ndarray = None, 
                 norm: bool = True, mode: str = "Std", encoding_scheme: str = "hierarchical", dim: int = None):
        """
        Initializes the NDMPS class.
        
        Args:
            mps (qtn.MatrixProductState, optional): MPS representation.
            qubit_size (tuple, optional): Size of the qubit encoding.
            encoding_map (np.ndarray, optional): Encoding map.
            norm (bool, optional): Whether to normalize data. Defaults to True.
            mode (str, optional): Compression mode, "Std" or "DCT". Defaults to "Std".
            encoding_scheme (str, optional): Encoding method: "rowmajor", "snake", or "hierarchical". Defaults to "hierarchical".
            dim (int, optional): Dimensionality of input tensor.
        """
        self.qubit_size = qubit_size
        self.encoding_map = encoding_map
        self.mps = mps
        self.norm = norm
        self.mode = mode
        self.encoding_scheme = encoding_scheme
        self.dim = dim
    
    @classmethod
    def from_tensor(cls, tensor: np.ndarray, norm: bool = False, mode: str = "Std", encoding_scheme: str = "hierarchical") -> "NDMPS":
        """
        Creates an NDMPS instance from a tensor with specified encoding scheme.
        
        Args:
            tensor (np.ndarray): Input tensor.
            norm (bool, optional): Whether to normalize the tensor. Defaults to False.
            mode (str, optional): Compression mode, "Std" or "DCT". Defaults to "Std".
            encoding_scheme (str, optional): Encoding method: "rowmajor", "snake", or "hierarchical". Defaults to "hierarchical".
        
        Returns:
            NDMPS: Instance of NDMPS class.
        """
        if encoding_scheme == "hierarchical":
            qubit_size, encoding_map = gen_encoding_map(tensor.shape)                

            encoding_map = np.moveaxis(encoding_map, 0, -1)

            # TODO: try to apply the dct to the individual blocks for the highest structure

            if norm:
                tensor /= np.linalg.norm(tensor)
            if mode == "DCT":
                tensor = dct(tensor, norm="ortho")

            # Ensure contracted_tensor has the correct shape
            contracted_tensor = np.zeros(shape=tuple(qubit_size), dtype=tensor.dtype)

            k = encoding_map.shape[-1]
            flat_data = tensor.flatten()
            flat_new_indices = encoding_map.reshape(-1, k).astype(int)

            # Convert indices to tuple format compatible with numpy array assignment
            indices = tuple(flat_new_indices[:, dim] for dim in range(k))
            contracted_tensor[indices] = flat_data  # This should now work correctly


        else: # Valid for both rowmajor and snake encodings - TODO: if we add another one, change here
            # Create an encoding map
            encoding_map = np.zeros(tensor.shape, dtype=int)
        
            num_elements = np.prod(tensor.shape)
            encoding_map.flat = np.arange(num_elements).reshape(tensor.shape)
            qubit_size = [2] * np.prod(tensor.shape)

            # TODO: is this the idea?

            if encoding_scheme == "snake":
                for i in range(tensor.shape[0]):  # Iterate over rows
                    if i % 2 == 1:  # Reverse order for every other row
                        encoding_map[i, :] = encoding_map[i, ::-1]
                #[:] = encoding_map # TODO: check dimensions here

            contracted_tensor = encoding_map

        mps = qtn.MatrixProductState.from_dense(contracted_tensor, dims=tuple(qubit_size))

        return cls(mps, qubit_size, encoding_map, norm, mode, encoding_scheme, len(tensor.shape))
    
    # @staticmethod
    # def generate_encoding_map(shape: tuple, encoding_scheme: str):
    #     """
    #     Generates an encoding map based on the specified encoding scheme, including hierarchical encoding.

    #     Args:
    #         shape (tuple): Shape of the input tensor.
    #         encoding_scheme (str): Encoding method: "rowmajor", "snake", or "hierarchical".

    #     Returns:
    #         tuple: (qubit_size, encoding_map)
    #     """
    #     num_elements = np.prod(shape)
    #     qubit_size = shape

    #     if encoding_scheme == "hierarchical":
    #         hierarchical_return = gen_encoding_map(shape)
    #         print(hierarchical_return)
    #         return hierarchical_return #gen_encoding_map(shape)  # Uses existing hierarchical method

    #     # Create an encoding map
    #     encoding_map = np.zeros(shape, dtype=int)

    #     if encoding_scheme == "rowmajor":
    #         # Linear row-major order assignment
    #         encoding_map.flat = np.arange(num_elements).reshape(shape)

    #     elif encoding_scheme == "snake":
    #         # Apply a zigzag/snake-like traversal pattern
    #         linear_indices = np.arange(num_elements).reshape(shape)
    #         for i in range(shape[0]):  # Iterate over rows
    #             if i % 2 == 1:  # Reverse order for every other row
    #                 linear_indices[i, :] = linear_indices[i, ::-1]
    #         encoding_map[:] = linear_indices

    #     # Convert encoding map to match hierarchical format
    #     encoding_map = np.moveaxis(np.indices(shape), 0, -1)
        
    #     return qubit_size, encoding_map
    
    def compression_ratio(self):
        """
        Computes the compression ratio.
        
        Returns:
            float: Compression ratio.
        """
        initial_N = np.prod(self.qubit_size)
        compressed_N = self.number_elements_in_MPS()
        return compressed_N / initial_N
    
    def compress(self, cutoff: float):
        """
        Compresses the MPS representation with a given cutoff.
        
        Args:
            cutoff (float): Relative cutoff value for bond compression.
        """
        size = len(self.mps.sites)
        for i in range(1, size):
            t1 = self.mps[i - 1]
            t2 = self.mps[i]
            qtn.tensor_compress_bond(t1, t2, cutoff=cutoff, cutoff_mode="rel")
    
    def continuous_compress(self, cutoff: float, print_ratio: bool = True):
        """
        Performs continuous compression at different cutoff levels.
        
        Args:
            cutoff (float): Base cutoff value.
            print_ratio (bool, optional): Whether to print compression ratio. Defaults to True.
        """
        compress_list = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1]) * cutoff
        for c in compress_list:
            self.compress(c)
            if print_ratio:
                print(f"Compression ratio at {c}: {self.compression_ratio()}")
    
    def number_elements_in_MPS(self):
        """
        Returns the number of tensor elements in the MPS.
        
        Returns:
            int: Total number of elements in MPS.
        """
        return sum(t.size for t in self.mps)
    
    def to_tensor(self):
        """
        Converts the MPS representation back to a tensor.
        
        Returns:
            np.ndarray: Reconstructed tensor.
        """
        contracted_mps = self.mps ^ ...
        for i in range(len(contracted_mps.inds)):
            contracted_mps.moveindex(f"k{i}", i, inplace=True)
        
        k = self.encoding_map.shape[-1]
        recovered_tensor = np.empty(self.encoding_map.shape)
        contracted_mps = contracted_mps.data
        recovered_tensor = contracted_mps[tuple(self.encoding_map[..., dim] for dim in range(k))]
        
        return recovered_tensor if self.mode == "Std" else idct(recovered_tensor, norm="ortho")
    
    def show(self):
        """
        Displays the MPS structure.
        """
        self.mps.show()
    
    def bond_sizes(self):
        """
        Returns the bond dimensions of the MPS.
        
        Returns:
            list: Bond dimensions.
        """
        return self.mps.bond_sizes()
