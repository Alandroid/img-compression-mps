import os
import sys
# Get the absolute path of the current script
current_path = os.path.abspath(__file__)
project_folder = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.append(project_folder)
import numpy as np
import quimb.tensor as qtn
from scipy.fftpack import dct, dctn, idct, idctn

from compression.utils_ND import *

class NDMPS:
    def __init__(self, mps: qtn.MatrixProductState = None, qubit_size: tuple = None, encoding_map: np.ndarray = None, 
                 norm: bool = True, mode: str = "Std", encoding_scheme: str = "hierarchical", dim: int = None, 
                 dct_level: int = None, dct_block_size: int = None, original_shape: tuple = (1,1)):
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
        self.dct_level = dct_level
        self.dct_block_size = dct_block_size
        self.original_shape = original_shape
    
    @classmethod
    def from_tensor(cls, tensor: np.ndarray, norm: bool = True, mode: str = "Std", encoding_scheme: str = "hierarchical", dct_level: int = -1) -> "NDMPS":
        """
        Creates an NDMPS instance from a tensor with specified encoding scheme, applying DCT at a chosen hierarchical level.

        Args:
            tensor (np.ndarray): Input tensor.
            norm (bool, optional): Whether to normalize the tensor. Defaults to True.
            mode (str, optional): Compression mode, "Std" or "DCT". Defaults to "Std".
            encoding_scheme (str, optional): Encoding method: "rowmajor", "snake", or "hierarchical". Defaults to "hierarchical".
            dct_level (int, optional): The hierarchical level at which to apply the DCT.
                                    -1 (default) applies at the coarsest level.
                                    0 applies at the finest level.
                                    None applies DCT to the entire tensor.

        Returns:
            NDMPS: Instance of NDMPS class.
        """
        original_shape = tensor.shape  # ✅ Store the original shape

        valid_dct_levels = get_valid_dct_levels(tensor.shape)

        print("Valid DCT levels:", valid_dct_levels)

        if encoding_scheme == "hierarchical":
            qubit_size, encoding_map, dct_block_size = gen_encoding_map(tensor.shape, dct_level=dct_level)
            encoding_map = np.moveaxis(encoding_map, 0, -1)

            if norm:
                tensor /= np.linalg.norm(tensor)

            if mode == "DCT":
                if dct_level is None:
                    # Apply DCT to the entire tensor (no patching)
                    tensor = dctn(tensor, norm="ortho")  
                else:
                    # Extract non-overlapping blocks correctly
                    block_tensor = extract_blocks(tensor, dct_block_size)

                    # Apply DCT per block
                    transformed_blocks = np.array([[dctn(block, norm="ortho") for block in row] for row in block_tensor])

                    # Reconstruct tensor properly
                    tensor = transformed_blocks.reshape(tensor.shape)

            # Ensure contracted_tensor has the correct shape
            contracted_tensor = np.zeros(shape=tuple(qubit_size), dtype=tensor.dtype)

            k = encoding_map.shape[-1]
            flat_data = tensor.flatten()
            flat_new_indices = encoding_map.reshape(-1, k).astype(int)

            # Convert indices to tuple format compatible with numpy array assignment
            indices = tuple(flat_new_indices[:, dim] for dim in range(k))
            contracted_tensor[indices] = flat_data  # Assign correctly

        else:  # Valid for both rowmajor and snake encodings
            encoding_map = np.zeros(tensor.shape, dtype=int)
            num_elements = np.prod(tensor.shape)
            encoding_map.flat = np.arange(num_elements).reshape(tensor.shape)
            qubit_size = [2] * np.prod(tensor.shape)

            if encoding_scheme == "snake":
                for i in range(tensor.shape[0]):  # Iterate over rows
                    if i % 2 == 1:  # Reverse order for every other row
                        encoding_map[i, :] = encoding_map[i, ::-1]

            contracted_tensor = encoding_map

        mps = qtn.MatrixProductState.from_dense(contracted_tensor, dims=tuple(qubit_size))

        return cls(mps, qubit_size, encoding_map, norm, mode, encoding_scheme, len(tensor.shape), dct_level, dct_block_size, original_shape)

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
        Converts the MPS representation back to a tensor, correctly handling hierarchical encoding 
        and DCT preprocessing with a unique `dct_block_size`.

        Returns:
            np.ndarray: Reconstructed tensor.
        """
        # Step 1: Decode tensor from MPS (hierarchical structure, NO DCT handling yet)
        contracted_mps = self.mps ^ ...
        for i in range(len(contracted_mps.inds)):
            contracted_mps.moveindex(f"k{i}", i, inplace=True)

        k = self.encoding_map.shape[-1]
        recovered_tensor = np.empty(self.encoding_map.shape[:-1])  # Remove last dimension
        contracted_mps = contracted_mps.data
        recovered_tensor = contracted_mps[tuple(self.encoding_map[..., dim] for dim in range(k))]

        # Step 2: If no DCT was applied, return reconstructed tensor
        if self.mode != "DCT":
            return recovered_tensor

        # Step 3: If DCT was applied globally, undo it at once
        if self.dct_level is None:
            return idctn(recovered_tensor, norm="ortho")

        # Step 4: If DCT was applied at a specific scale, apply IDCT per block
        if self.dct_block_size is not None:
            # Compute the number of blocks
            num_blocks = tuple(recovered_tensor.shape[i] // self.dct_block_size[i] for i in range(len(self.dct_block_size)))

            # Ensure block structure before applying IDCT
            reshaped_tensor = recovered_tensor.reshape(*num_blocks, *self.dct_block_size)

            # Apply IDCT per block
            inverse_transformed_blocks = np.array([
                [idctn(block, norm="ortho") for block in row] for row in reshaped_tensor
            ])
            # TODO: check if it works for 3D and also if we really need to store the original_shape
            reconstructed_tensor = inverse_transformed_blocks.reshape(self.original_shape)

        return reconstructed_tensor

    # TODO: old version - dct the whole image
    # def to_tensor(self):
    #     """
    #     Converts the MPS representation back to a tensor.
        
    #     Returns:
    #         np.ndarray: Reconstructed tensor.
    #     """
    #     contracted_mps = self.mps ^ ...
    #     for i in range(len(contracted_mps.inds)):
    #         contracted_mps.moveindex(f"k{i}", i, inplace=True)
        
    #     k = self.encoding_map.shape[-1]
    #     recovered_tensor = np.empty(self.encoding_map.shape)
    #     contracted_mps = contracted_mps.data
    #     recovered_tensor = contracted_mps[tuple(self.encoding_map[..., dim] for dim in range(k))]
        
    #     return recovered_tensor if self.mode == "Std" else idctn(recovered_tensor, norm="ortho") # TODO: or idct? !!!
    
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
